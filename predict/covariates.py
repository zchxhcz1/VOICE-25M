import os, json
import numpy as np
from predict.geometry import OUTPUT_SIZE, MPP_OUT
log=print
def polygon_areas_um2(ip, vx, vy, um_per_px):
    out = np.zeros(len(ip) - 1, np.float32)
    for i in range(len(out)):
        a, b = int(ip[i]), int(ip[i + 1])
        if b - a >= 3:
            x, y = vx[a:b], vy[a:b]
            out[i] = 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return out * um_per_px ** 2

def geometry(src, mpp_arg):
    """(pos_um [N,2], area_abs, area_rel, mpp) -- the covariates expression_head.cell_areas / LiveStore build."""
    if src.kind == "prepared":
        dj = os.path.join(src.dir, "dataset.json")
        mpp = mpp_arg or (float(json.load(open(dj))["um_per_pixel_source"]) if os.path.exists(dj) else None)
        if mpp is None:
            raise SystemExit("prepared slide without dataset.json: pass --mpp (um per SOURCE pixel of x_pixel/y_pixel)")
        if not np.isfinite(mpp) or mpp <= 0:
            raise ValueError("Physical resolution must be positive and finite")
        area = polygon_areas_um2(src.ip, src.vx * (OUTPUT_SIZE / src.output_size), src.vy * (OUTPUT_SIZE / src.output_size), MPP_OUT)
    else:
        mpp = mpp_arg or src.mpp
        if mpp is None:
            raise SystemExit("no resolution: pass --mpp")
        if src.poly is None:
            log("[warn] cells.npz has no polygons: centre-token masks and a constant cell area are used")
            area = np.zeros(len(src), np.float32)
        else:
            area = polygon_areas_um2(*src.poly, mpp)
    med = float(np.median(area[area > 0])) if (area > 0).any() else 60.0
    area = np.where(area > 0, area, med).astype(np.float32)
    return (src.pos * mpp).astype(np.float32), np.log(area + 1.0).astype(np.float32), \
           (np.log(area + 1.0) - np.log(med + 1.0)).astype(np.float32), float(mpp)
