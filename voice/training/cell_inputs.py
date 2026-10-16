import os
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
def cells_of(d):
    """Every cell of the object: crop path, 16x16 cmask weights, expr_row, source-pixel position."""
    z = np.load(f"{d}/patch_cell_boundaries.npz", allow_pickle=True)
    er = z["expr_rows"].astype(np.int64); ip = z["indptr"]
    vx, vy = z["vertex_x_patch"], z["vertex_y_patch"]
    osize = int(z["output_size"]); tok = osize // 16
    man = (pd.read_csv(f"{d}/manifest.csv.gz", usecols=["expr_row", "patch_path"])
           .drop_duplicates("expr_row").set_index("expr_row")["patch_path"])
    from PIL import ImageDraw
    paths, Ws = [], np.zeros((len(er), 16, 16), np.float32)
    for i in range(len(er)):
        a, b = int(ip[i]), int(ip[i + 1]); w = np.zeros((16, 16), np.float32)
        if b - a >= 3:
            im = Image.new("L", (osize, osize), 0)
            ImageDraw.Draw(im).polygon(list(zip(vx[a:b].tolist(), vy[a:b].tolist())), fill=1)
            w = np.asarray(im, np.float32).reshape(16, tok, 16, tok).mean((1, 3))
        if w.sum() <= 1e-6:
            w[8, 8] = 1.0
        Ws[i] = w / w.sum()
        paths.append(os.path.join(d, str(man.loc[int(er[i])])))
    pos = np.stack([z["y_pixel"], z["x_pixel"]], 1).astype(np.float32)
    return paths, Ws, er, pos
