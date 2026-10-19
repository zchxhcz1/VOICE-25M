"""Fit reference-derived gene fusion weights and combine direct and retrieved scores."""
from __future__ import annotations
import numpy as np

GRID = np.linspace(0, 1, 41).astype(np.float32)
EPS = 1e-8


def _validate_arrays(A, R, Y=None):
    if A.ndim != 2 or R.shape != A.shape or (Y is not None and Y.shape != A.shape):
        raise ValueError("A, R and Y must have matching [cells, genes] shapes")
    if not len(A) or not np.isfinite(A).all() or (Y is not None and not np.isfinite(Y).all()):
        raise ValueError("A and measured Y must be nonempty and finite")


def _fit_eligible(A, R, Y):
    return (np.isfinite(R).all(0) & (Y.std(0) > EPS)
            & ((A.std(0) > EPS) | (np.nan_to_num(R, nan=0., posinf=0., neginf=0.).std(0) > EPS)))


def _centred(X):
    X = np.asarray(X, np.float32)
    return X - X.mean(0, keepdims=True)


def per_gene_pcc_blend(Ac, Rc, Yc, Yn, beta):
    """Pearson per gene of `beta*A + (1-beta)*R`, with A, R, Y already centred and |Y| precomputed."""
    b = beta if np.isscalar(beta) else np.asarray(beta, np.float32)[None, :]
    P = b * Ac + (1.0 - b) * Rc
    return (P * Yc).sum(0) / (np.sqrt((P ** 2).sum(0)) * Yn + EPS)


def _fit_gene_chunk(A, R, Y, covered=None, grid=GRID):
    """Fit one beta per gene on a slide whose labels you may read.

    A, R : [N, G] predictions. R may contain NaN where retrieval has no coverage.
    Y    : [N, G] measured log1p expression.
    Returns (beta [G] float32, pcc_at_beta [G] float32).

    A 41-point grid, not a closed form: the objective is a correlation of a blend, which is smooth but not
    concave in beta, and a grid is both robust and cheap at this size. Uncovered genes get beta = 1.
    """
    A = np.asarray(A, np.float32); R = np.asarray(R, np.float32); Y = np.asarray(Y, np.float32)
    _validate_arrays(A, R, Y)
    if covered is None:
        covered = np.isfinite(R).all(0)
    covered = np.asarray(covered, bool) & _fit_eligible(A, R, Y)
    if covered.shape != (A.shape[1],):
        raise ValueError("covered must have one entry per gene")
    G = A.shape[1]
    Rz = np.nan_to_num(R, nan=0., posinf=0., neginf=0.)
    Ac, Rc, Yc = _centred(A), _centred(Rz), _centred(Y)
    Yn = np.sqrt((Yc ** 2).sum(0))

    best = np.full(G, -np.inf, np.float32)
    beta = np.ones(G, np.float32)
    for b in grid:
        p = per_gene_pcc_blend(Ac, Rc, Yc, Yn, float(b))
        m = (p > best) | ((p == best) & (b > beta))
        best[m] = p[m]; beta[m] = b
    beta[~covered] = 1.0                                   # no retrieval -> direct branch only
    best[~covered] = per_gene_pcc_blend(Ac, Rc, Yc, Yn, 1.0)[~covered]
    return beta, np.nan_to_num(best).astype(np.float32)


def fit_beta(A, R, Y, covered=None, grid=GRID):
    """Fit the 41-point correlation grid in gene chunks to bound temporary memory."""
    A = np.asarray(A, np.float32)
    R = np.asarray(R, np.float32)
    Y = np.asarray(Y, np.float32)
    _validate_arrays(A, R, Y)
    if covered is not None and np.asarray(covered).shape != (A.shape[1],):
        raise ValueError('covered must have one entry per gene')
    beta, score = np.empty(A.shape[1], np.float32), np.empty(A.shape[1], np.float32)
    for start in range(0, A.shape[1], 128):
        stop = min(start+128, A.shape[1])
        mask = None if covered is None else np.asarray(covered)[start:stop]
        beta[start:stop], score[start:stop] = _fit_gene_chunk(
            A[:,start:stop].copy(), R[:,start:stop].copy(), Y[:,start:stop].copy(), mask, grid)
    return beta, score


def apply_gate(A, R, panel, beta_by_gene, default=1.0):
    """Fuse on a target slide. Nothing here reads the target's labels.

    A, R  : [N, G] target predictions; NaN in R means no retrieval coverage
    panel : [G] global gene ids
    Returns (fused [N, G] float32, beta_used [G] float32).
    """
    A = np.asarray(A, np.float32); R = np.asarray(R, np.float32)
    _validate_arrays(A, R)
    panel = np.asarray(panel, np.int64)
    if panel.shape != (A.shape[1],):
        raise ValueError("Target panel must have one global ID per gene")
    covered = np.isfinite(R).all(0)
    beta = np.array([beta_by_gene.get(int(g), default) for g in panel], np.float32)
    # A gene the references never fitted arrives as NaN (or is simply absent); either way it falls back to the
    # direct branch. Letting a NaN through would silently blank that gene's whole column.
    beta[~np.isfinite(beta)] = 1.0
    if np.any((beta < 0) | (beta > 1)):
        raise ValueError("Finite gate beta must be in [0, 1]")
    beta[~covered] = 1.0                                   # transferred beta cannot apply where R is absent
    Rz = np.nan_to_num(R, nan=0., posinf=0., neginf=0.)
    return (beta[None, :] * A + (1.0 - beta[None, :]) * Rz).astype(np.float32), beta
