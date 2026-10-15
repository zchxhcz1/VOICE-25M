"""Stage 3, part two: the per-gene gate that fuses the two branches.

    pred_g = beta_g * A_g + (1 - beta_g) * R_g,        beta_g in [0, 1]

One weight per gene, not per cell. That is a deliberate limit, and it was tested: a learned router that also
conditions on the cell (`beta_{i,g} = sigmoid(u_g + s_g * z_i)`) does not reliably beat this constant, and hard
per-cell routing is actively worse — the metric is a correlation computed ACROSS cells for each gene, so mixing
two prediction scales cell-by-cell within one gene's column corrupts that column even when every cell was
individually sent to the better branch.

**Where beta is fitted is the whole game.** Fitting it on the target slide needs that slide's labels, which is
the thing being predicted; such a gate is an oracle and an upper bound, not a method. The deployable gate fits
beta on *reference* slides of the same tissue — slides that have measured expression — and transfers it
unchanged. `fit_beta` does the fitting, `apply_gate` the transferring, and they are separate functions so it
stays obvious which slides' labels were read.

The reference slides must be scored the way the target will be: each reference's own R must come from a bank
that excludes it. A gate fitted against self-retrieval sees an R that is far too good and transfers a beta that
trusts retrieval much more than it should.

Genes with no retrieval coverage keep beta = 1 (direct branch only) rather than being dropped, so the fused
prediction always spans the full panel.
"""
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


def fit_beta(A, R, Y, covered=None, grid=GRID):
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


def fit_beta_reference(references, grid=GRID):
    """Fit beta on several reference slides and average, keyed by GLOBAL gene id.

    `references` is an iterable of (A, R, Y, panel_global_ids). Averaging over references rather than
    concatenating their cells keeps a large slide from dominating a small one.

    Returns {global_gene_id: beta}. A gene absent from every reference simply has no entry, and
    `apply_gate` then falls back to the direct branch for it.
    """
    acc, cnt = {}, {}
    for A, R, Y, panel in references:
        A, R, Y = (np.asarray(x, np.float32) for x in (A, R, Y))
        _validate_arrays(A, R, Y)
        panel = np.asarray(panel, np.int64)
        if panel.shape != (A.shape[1],) or len(set(panel.tolist())) != len(panel):
            raise ValueError("Reference panel must contain one unique global ID per gene")
        eligible = _fit_eligible(A, R, Y)
        beta, _p = fit_beta(A, R, Y)
        for g, b in zip(panel[eligible], beta[eligible]):
            if not np.isfinite(b):
                continue
            g = int(g)
            acc[g] = acc.get(g, 0.0) + float(b)
            cnt[g] = cnt.get(g, 0) + 1
    return {g: acc[g] / cnt[g] for g in acc}


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


def oracle_beta(A, R, Y, covered=None):
    """Beta fitted on the target itself. An UPPER BOUND, not a method: it reads the labels being predicted.

    Report it to show how much room a per-gene scheme had, never as a result.
    """
    return fit_beta(A, R, Y, covered)
