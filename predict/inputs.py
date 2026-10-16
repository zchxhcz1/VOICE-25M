"""Read cell crops and boundaries from a slide image or a prepared slide directory."""
from __future__ import annotations
import os
import numpy as np

from predict.geometry import polygon_mask, crop_px_for, OUTPUT_SIZE, TOKEN_GRID, MPP_REF


class WSISlide:
    """Crops cut on demand from a whole-slide image, using segmentations from a cells.npz."""

    kind = "wsi"

    def __init__(self, image, cells, mpp=None):
        from predict.slide_io import SlideReader
        self.image = str(image)
        z = np.load(cells, allow_pickle=True)
        self.pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)
        self.poly = ((z["indptr"].astype(np.int64), z["vertex_x"].astype(np.float32),
                      z["vertex_y"].astype(np.float32))
                     if "indptr" in z.files and "vertex_x" in z.files else None)
        self.ids = (z["cell_id"].astype(str) if "cell_id" in z.files
                    else np.array([f"cell_{i}" for i in range(len(self.pos))]))
        rd = SlideReader(self.image)
        self.mpp = mpp if mpp is not None else rd.mpp
        if self.mpp is None and "mpp" in z.files and np.isfinite(float(z["mpp"])):
            self.mpp = float(z["mpp"])
        if self.mpp is None or not np.isfinite(self.mpp) or self.mpp <= 0:
            raise ValueError("Physical image resolution is required: pass --mpp in um/pixel")
        self.width, self.height, self.backend = rd.width, rd.height, rd.backend
        rd.close()
        self.crop_px = crop_px_for(self.mpp)
        self._rd = None

    def __len__(self): return len(self.pos)

    def describe(self):
        fov = self.crop_px * (self.mpp or MPP_REF)
        return (f"{os.path.basename(self.image)} {self.width}x{self.height} via {self.backend} | "
                f"mpp={self.mpp if self.mpp else 'unknown'} -> crop {self.crop_px:.1f} px = {fov:.1f} um | "
                f"polygons: {'yes' if self.poly else 'NO (centre-token mask)'}")

    def _reader(self):
        if self._rd is None:                      # opened per worker: slide handles are not fork-safe
            from predict.slide_io import SlideReader
            self._rd = SlideReader(self.image)
        return self._rd

    def batch(self, idx):
        idx = np.asarray(idx)
        cy = self.pos[idx, 0]; cx = self.pos[idx, 1]
        imgs = self._reader().crops(cx, cy, self.crop_px, OUTPUT_SIZE)
        W = np.empty((len(idx), TOKEN_GRID, TOKEN_GRID), np.float32)
        side = int(round(self.crop_px))
        half = side // 2
        s = OUTPUT_SIZE / side            # slide px -> crop frame
        for j, c in enumerate(idx):
            if self.poly is None:
                W[j] = polygon_mask(np.empty(0), np.empty(0))
            else:
                ip, vx, vy = self.poly
                a, b = int(ip[c]), int(ip[c + 1])
                W[j] = polygon_mask((vx[a:b] - (int(round(cx[j])) - half)) * s, (vy[a:b] - (int(round(cy[j])) - half)) * s)
        return imgs, W


class PreparedSlide:
    """Crops read from a pre-built slide directory (the training / benchmark layout)."""

    kind = "prepared"

    def __init__(self, directory, mpp=None):
        self.dir = str(directory)
        bz = np.load(os.path.join(self.dir, "patch_cell_boundaries.npz"), allow_pickle=True)
        self.er = bz["expr_rows"].astype(np.int64)
        self.ip = bz["indptr"].astype(np.int64)
        self.vx = bz["vertex_x_patch"].astype(np.float32)
        self.vy = bz["vertex_y_patch"].astype(np.float32)
        self.output_size = int(bz["output_size"])
        self.crop_px = float(bz["crop_px"]) if "crop_px" in bz.files else float(self.output_size)
        self.pos = np.stack([bz["y_pixel"], bz["x_pixel"]], 1).astype(np.float32)
        self.mpp = mpp

        import pandas as pd
        man = (pd.read_csv(os.path.join(self.dir, "manifest.csv.gz"), usecols=["expr_row", "patch_path"])
                 .drop_duplicates("expr_row").set_index("expr_row")["patch_path"])
        self.paths = [os.path.join(self.dir, str(man.loc[int(e)])) for e in self.er]
        self.ids = np.array([str(e) for e in self.er])
        if len(self.paths) != len(self.pos):
            raise ValueError(f"{self.dir}: manifest has {len(self.paths)} crops but boundaries have {len(self.pos)} cells")

    def __len__(self): return len(self.pos)

    def describe(self):
        return (f"{os.path.basename(self.dir)} | {len(self)} pre-cut crops at {self.output_size} px "
                f"(source crop {self.crop_px:.0f} px) | polygons: yes")

    def batch(self, idx):
        from PIL import Image
        idx = np.asarray(idx)
        imgs = np.empty((len(idx), 3, OUTPUT_SIZE, OUTPUT_SIZE), np.uint8)
        W = np.empty((len(idx), TOKEN_GRID, TOKEN_GRID), np.float32)
        for j, c in enumerate(idx):
            with Image.open(self.paths[c]) as original:
                im = original.convert("RGB")
            if im.size != (OUTPUT_SIZE, OUTPUT_SIZE):
                im = im.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)
            imgs[j] = np.ascontiguousarray(np.asarray(im, np.uint8).transpose(2, 0, 1))
            a, b = int(self.ip[c]), int(self.ip[c + 1])
            # vertices are already in the crop frame; rescale only if the stored crop is not 224 px
            sx = OUTPUT_SIZE / self.output_size
            W[j] = polygon_mask(self.vx[a:b] * sx, self.vy[a:b] * sx)
        return imgs, W

    def expression(self, head):
        """(Y_log1p [N, G], global gene ids [G]) over genes present in `head` (symbol -> global id); (None, None)
        if the slide carries no expression. Only used to SCORE a prediction."""
        e = os.path.join(self.dir, "expression.npz"); g = os.path.join(self.dir, "genes.tsv")
        if not (os.path.exists(e) and os.path.exists(g)):
            return None, None
        import pandas as pd
        from scipy import sparse
        syms = pd.read_csv(g, sep="\t").iloc[:, 0].astype(str).tolist()
        cols, hidx, seen = [], [], set()
        for k, sym in enumerate(syms):
            if sym in head and sym not in seen:
                seen.add(sym); cols.append(k); hidx.append(head[sym])
        X = sparse.load_npz(e).tocsr()[self.er][:, np.asarray(cols, np.int64)]
        return np.log1p(np.asarray(X.todense(), np.float32)), np.asarray(hidx, np.int64)


def open_slide(image=None, cells=None, prepared=None, mpp=None):
    """Pick the source from whichever arguments were given."""
    if prepared:
        return PreparedSlide(prepared, mpp)
    if image and cells:
        return WSISlide(image, cells, mpp)
    raise ValueError("give either --prepared <dir>, or both --image and --cells")
