"""The crop / mask geometry the model was trained under. Every number here is a contract, not a preference.

VOICE was trained on a fixed PHYSICAL field of view. Per cell:

    fov         = 55 um  square, centred on the cell centroid (identical on every training slide)
    crop_px     = 55 um / (um per pixel of the slide)   e.g. 201 px at 0.2738 um/px, 259 px at 0.2124 um/px
    output_size = 224    the crop is resized to 224 x 224 before the encoder (=> 0.2455 um/px seen by the model)
    token grid  = 16x16  UNI2-h at 224 px with patch 14 emits 16*16 = 256 patch tokens
    mask        = the cell's polygon rasterised in the 224-px frame, area-pooled to 16x16, summing to 1

The training corpus was cut at 55.0 um on every slide, with the source pixel count following the slide's own
resolution (0.137-0.364 um/px -> 402-151 px). What has to match on YOUR slide is that PHYSICAL size, not a pixel
count: pass --mpp and crop_px = 55 / mpp. When the resolution is unknown the fallback is 201 px, which is 55 um
only at 0.2738 um/px (the resolution of most training H&E). Feeding the model a 201-px crop from a 0.2125 um/px
scan shows it 43 um of tissue instead of 55, and the predictions degrade quietly rather than failing.

The pooled feature is a cell-mask weighted average over the 256 patch tokens (`voice.encoder.pooled_feat`),
so the mask decides which tokens speak for the cell. A degenerate polygon (fewer than 3 vertices, or zero area
after rasterisation) falls back to the single centre token, matching training.
"""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

FOV_UM = 55.0              # THE contract: physical crop side on every training slide (dataset.json patch_size_um)
MPP_REF = 0.2738           # um/px of most training H&E; only used to turn FOV_UM into a pixel fallback
CROP_PX_REF = 201.0        # = round(FOV_UM / MPP_REF): the fallback when a slide's resolution is unknown
OUTPUT_SIZE = 224          # encoder input side
TOKEN_GRID = 16            # UNI2-h patch-token grid at 224 px
MPP_OUT = FOV_UM / OUTPUT_SIZE     # 0.2455 um/px, what the encoder actually sees


def crop_px_for(mpp: float | None) -> float:
    """Source-pixel crop side that covers FOV_UM (55 um) at YOUR resolution; 201 px when the resolution is unknown."""
    if mpp is None:
        return CROP_PX_REF
    return FOV_UM / float(mpp)


def polygon_mask(vx, vy, output_size: int = OUTPUT_SIZE, grid: int = TOKEN_GRID) -> np.ndarray:
    """Rasterise one cell polygon (crop-local pixel coords) and area-pool it to [grid, grid], summing to 1.

    Identical to the training-time construction: fill the polygon at full crop resolution, average within each
    token block, then normalise. Degenerate polygons fall back to the centre token.
    """
    tok = output_size // grid
    w = np.zeros((grid, grid), np.float32)
    if len(vx) >= 3:
        im = Image.new("L", (output_size, output_size), 0)
        ImageDraw.Draw(im).polygon(list(zip(np.asarray(vx).tolist(), np.asarray(vy).tolist())), fill=1)
        w = np.asarray(im, np.float32).reshape(grid, tok, grid, tok).mean((1, 3))
    if w.sum() <= 1e-6:
        w[grid // 2, grid // 2] = 1.0
    return w / w.sum()


def tile(pos: np.ndarray, patch_size: int = 256):
    """Group cells into non-overlapping square tiles so the SE(2) decoder sees a local neighbourhood.

    pos is [N,2] = (y, x) in slide pixels. Every cell lands in exactly one tile, so every cell is predicted
    exactly once -- the same tiling the cross-slide evaluation uses.
    """
    cy, cx = pos[:, 0], pos[:, 1]
    H, W = cy.max() + patch_size, cx.max() + patch_size
    out = []
    for y0 in np.arange(0, H, patch_size):
        rm = (cy >= y0) & (cy < y0 + patch_size)
        if not rm.any():
            continue
        cyr = np.where(rm)[0]; cxr = cx[rm]
        for x0 in np.arange(0, W, patch_size):
            cm = (cxr >= x0) & (cxr < x0 + patch_size)
            if cm.any():
                out.append(cyr[cm])
    return out
