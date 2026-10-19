"""Cell-centered 55-micron crops and normalized 16-by-16 cell masks."""
from __future__ import annotations
import numpy as np
from PIL import Image, ImageDraw

FOV_UM = 55.0              # THE contract: physical crop side on every training slide (dataset.json patch_size_um)
OUTPUT_SIZE = 224          # encoder input side
TOKEN_GRID = 16            # UNI2-h patch-token grid at 224 px
MPP_OUT = FOV_UM / OUTPUT_SIZE     # 0.2455 um/px, what the encoder actually sees


def crop_px_for(mpp: float) -> float:
    """Source pixels covering the 55-micron training field of view."""
    if mpp is None or not np.isfinite(mpp) or mpp <= 0:
        raise ValueError('A positive physical image resolution is required')
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
