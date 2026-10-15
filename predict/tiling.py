"""Density-adapted spatial tiles, preserving canonical 25M tiling and complete coverage."""
import numpy as np


def _grid(pos, side):
    bins = np.floor((pos[:, 0].astype(np.float64) - pos[:, 0].min()) / side).astype(np.int64)
    result = []
    for y in np.unique(bins):
        rows = np.flatnonzero(bins == y)
        x = pos[rows, 1].astype(np.float64)
        xb = np.floor((x-x.min()) / side).astype(np.int64)
        result.extend(rows[xb == b] for b in np.unique(xb))
    return result


def tiles_for(pos, target=128, cap=512):
    pos = np.asarray(pos, np.float32)
    if pos.ndim != 2 or pos.shape[1] != 2 or not len(pos) or not np.isfinite(pos).all():
        raise ValueError('Expected nonempty, finite [N,2] cell coordinates')
    if target < 1 or cap < 1:
        raise ValueError('Tile sizes must be positive')
    side = float(np.sqrt(target * float(np.ptp(pos, axis=0).prod()) / len(pos)))
    if side <= 0:  # a single cell, coincident cells, or cells on a straight line
        side = max(1., float(np.ptp(pos, axis=0).max()))
    result = []
    for tile in _grid(pos, side):
        if len(tile) <= cap:
            result.append(tile)
        else:
            sub = [tile[u] for u in _grid(pos[tile], side/2)]
            result.extend(sub if all(len(u) <= cap for u in sub)
                          else [tile[i:i+cap] for i in range(0, len(tile), cap)])
    counts = np.bincount(np.concatenate(result), minlength=len(pos))
    if not np.all(counts == 1):
        raise RuntimeError('Spatial partition does not cover each cell exactly once')
    return side, result
