"""scGPT (whole-human) cell embeddings for every object of the corpus -- the Stage-1 target for
`clip_muon_ab.py --target scgpt` (docs/scgpt_variant.md).

Uses scGPT's own code, vendored unmodified in third_party/scgpt (v0.2.4), exactly as
`scgpt.tasks.cell_emb.get_batch_cell_embeddings` does it:
  * genes outside scGPT's vocabulary are dropped; each cell's tokens are its non-zero genes, `<cls>` prepended with
    value pad_value (-2);
  * `DataCollator(do_binning=True, max_length=1200, sampling=True, keep_first_n_tokens=1)`: per-cell binning of raw
    counts into 51 bins, random subsampling to 1,200 tokens for cells with more non-zero genes (CosMx);
  * fp16 autocast, the `<cls>` position of `TransformerModel._encode`, L2-normalised.
Cells are batched by length (the official loader is sequential); each cell is still encoded on its own, so this
changes speed, not values, apart from the binning tie-break / subsampling RNG, seeded per object with crc32(name) so reruns are bit-identical. The model is
built as in `embed_data` with use_fast_transformer=False (no flash-attn), Wqkv -> in_proj as scgpt.utils does.

Output: $VOICE_SCGPT_EMB/<obj>.npy, [n_expression_rows, 512] float16, ROW-ALIGNED with expression.npz (the
scf_cellemb.npy convention). Cells with no in-vocabulary non-zero gene get a zero row; clip_muon_ab drops those.
Per-object stats go to <obj>.json. Resumable: finished objects are skipped.

    python scgpt_cellemb.py [--only a,b] [--shard k --nshard n]
"""
from __future__ import annotations
import os, sys, json, time, glob, types, zlib, logging, argparse
from contextlib import nullcontext
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, str(__import__("pathlib").Path(HERE).parent))
import numpy as np, pandas as pd, scipy.sparse as sp, torch

from voice.scgpt_backend.model import TransformerModel        # noqa: E402
from voice.scgpt_backend.data_collator import DataCollator          # noqa: E402

DATASET = os.environ.get("VOICE_DATASET", "data")
SCGPT_DIR = os.environ.get("VOICE_SCGPT_DIR", os.path.join(os.path.dirname(HERE), "weights", "scgpt_human"))
OUT = os.environ.get("VOICE_SCGPT_EMB", "")
log = lambda *x: print(*x, flush=True)


def load_model(dev):
    cfg = json.load(open(f"{SCGPT_DIR}/args.json"))
    vocab = json.load(open(f"{SCGPT_DIR}/vocab.json"))
    for s in ("<pad>", "<cls>", "<eoc>"):
        assert s in vocab, s
    model = TransformerModel(ntoken=len(vocab), d_model=cfg["embsize"], nhead=cfg["nheads"], d_hid=cfg["d_hid"],
                             nlayers=cfg["nlayers"], nlayers_cls=cfg["n_layers_cls"], n_cls=1, vocab=vocab,
                             dropout=cfg["dropout"], pad_token=cfg["pad_token"], pad_value=cfg["pad_value"],
                             do_mvc=True, do_dab=False, use_batch_labels=False, domain_spec_batchnorm=False,
                             explicit_zero_prob=False, use_fast_transformer=False,
                             fast_transformer_backend="flash", pre_norm=False)
    sd = torch.load(f"{SCGPT_DIR}/best_model.pt", map_location="cpu", weights_only=False)
    sd = {k.replace("Wqkv.", "in_proj_"): v for k, v in sd.items()}
    own = model.state_dict()
    used = {k: v for k, v in sd.items() if k in own and v.shape == own[k].shape}
    model.load_state_dict(used, strict=False)
    enc_keys = [k for k in own if k.startswith(("encoder.", "value_encoder.", "transformer_encoder."))]
    missing = [k for k in enc_keys if k not in used]
    assert not missing, f"encoder weights not loaded: {missing[:5]}"
    log(f"[scgpt] {SCGPT_DIR}: {len(used)}/{len(sd)} tensors; every encoder tensor present ({len(enc_keys)})")
    return model.to(dev).eval(), vocab, cfg


class CellDS(torch.utils.data.Dataset):
    """get_batch_cell_embeddings' Dataset, on a CSR matrix restricted to in-vocabulary genes."""
    def __init__(self, X, gene_ids, cls_id, pad_value):
        self.X, self.g, self.cls, self.pv = X, gene_ids, cls_id, pad_value
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, i):
        a, b = self.X.indptr[i], self.X.indptr[i + 1]
        cols = self.X.indices[a:b]; vals = self.X.data[a:b].astype(np.float32)
        nz = vals > 0
        return {"genes": torch.from_numpy(np.insert(self.g[cols[nz]], 0, self.cls)).long(),
                "expressions": torch.from_numpy(np.insert(vals[nz], 0, self.pv)).float()}


def embed_object(o, d, model, vocab, cfg, dev, max_tokens=98304, workers=8):
    X = sp.load_npz(f"{d}/expression.npz").tocsr()
    syms = pd.read_csv(f"{d}/genes.tsv", sep="\t").iloc[:, 0].astype(str).tolist()
    assert len(syms) == X.shape[1], (o, len(syms), X.shape)
    vid = np.array([vocab.get(s, -1) for s in syms], np.int64)
    keep = np.where(vid >= 0)[0]
    Xk = X[:, keep].tocsr(); Xk.eliminate_zeros(); Xk.sort_indices()
    nnz = np.diff(Xk.indptr); n = Xk.shape[0]
    log(f"  [{o[:40]:40s}] {n:,} cells | {len(keep):,}/{len(syms):,} genes in vocab | non-zero/cell "
        f"median {int(np.median(nnz)) if n else 0}, max {int(nnz.max()) if n else 0}, empty {int((nnz == 0).sum()):,}")
    np.random.seed(zlib.crc32(o.encode()))          # stable per object (Python's str hash is salted per process)
    torch.manual_seed(0)
    ds = CellDS(Xk, vid[keep], vocab["<cls>"], cfg["pad_value"])
    col = DataCollator(do_padding=True, pad_token_id=vocab[cfg["pad_token"]], pad_value=cfg["pad_value"],
                       do_mlm=False, do_binning=True, max_length=1200, sampling=True, keep_first_n_tokens=1)
    order = np.argsort(nnz, kind="stable"); order = order[nnz[order] > 0]
    batches, cur, cur_len = [], [], 0
    for i in order:                                   # token-budget batches over length-sorted cells
        L = min(int(nnz[i]), 1199) + 1
        if cur and (len(cur) + 1) * max(cur_len, L) > max_tokens:
            batches.append(cur); cur, cur_len = [], 0
        cur.append(int(i)); cur_len = max(cur_len, L)
    if cur:
        batches.append(cur)
    dl = torch.utils.data.DataLoader(ds, batch_sampler=batches, collate_fn=col, num_workers=workers,
                                     pin_memory=dev.type == "cuda", prefetch_factor=4 if workers else None)
    out = np.zeros((n, cfg["embsize"]), np.float32); t0 = time.time(); done = 0
    with torch.no_grad(), (torch.autocast("cuda", dtype=torch.float16) if dev.type == "cuda" else nullcontext()):
        for k, bd in enumerate(dl):                   # DataCollator drops ids; batches arrive in sampler order
            ids = np.asarray(batches[k], np.int64)
            g = bd["gene"].to(dev, non_blocking=True)
            assert len(ids) == g.shape[0]
            e = model._encode(g, bd["expr"].to(dev, non_blocking=True),
                              src_key_padding_mask=g.eq(vocab[cfg["pad_token"]]))[:, 0, :]
            out[ids] = e.float().cpu().numpy(); done += len(ids)
    nrm = np.linalg.norm(out, axis=1, keepdims=True)
    out = np.where(nrm > 0, out / np.maximum(nrm, 1e-12), 0.0).astype(np.float16)
    dt = time.time() - t0
    log(f"      embedded {done:,} cells in {dt:.0f}s ({done / max(dt, 1e-9):.0f} cells/s)")
    return out, dict(n_cells=n, n_genes_vocab=int(len(keep)), n_genes_panel=len(syms), n_empty=int((nnz == 0).sum()),
                     nnz_median=float(np.median(nnz)) if n else 0.0, n_subsampled=int((nnz > 1199).sum()), sec=dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--scgpt-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument("--only", default="", help="comma-separated objects (default: every object in the corpus)")
    ap.add_argument("--shard", type=int, default=0); ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=98304)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    global DATASET, SCGPT_DIR, OUT
    DATASET, SCGPT_DIR, OUT = a.dataset, a.scgpt_dir, a.out
    assert OUT, "set VOICE_SCGPT_EMB to the output directory"
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for split in ("training_set",):
        for plat in ("xenium", "cosmx"):
            for d in sorted(glob.glob(f"{DATASET}/{split}/{plat}/*/*/")):
                d = d.rstrip("/")
                if os.path.exists(f"{d}/expression.npz"):
                    jobs.append((os.path.basename(d), d))
    if a.only:
        want = [x for x in a.only.split(",") if x]
        jobs = [j for j in jobs if j[0] in want]
        assert {j[0] for j in jobs} == set(want), f"unknown objects: {sorted(set(want) - {j[0] for j in jobs})}"
    jobs = jobs[a.shard::a.nshard]
    log(f"[plan] shard {a.shard}/{a.nshard}: {len(jobs)} objects -> {OUT}")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, vocab, cfg = load_model(dev)
    for o, d in jobs:
        p = f"{OUT}/{o}.npy"
        if os.path.exists(p):
            log(f"  [{o}] cached"); continue
        out, m = embed_object(o, d, model, vocab, cfg, dev, a.max_tokens, a.workers)
        tmp = p + ".tmp.npy"; np.save(tmp, out); os.replace(tmp, p)
        json.dump(m, open(f"{OUT}/{o}.json", "w"), indent=2)
    log("[done]")


if __name__ == "__main__":
    main()
