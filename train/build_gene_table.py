#!/usr/bin/env python
"""Extract normalized scGPT gene embeddings for an explicit ordered gene list."""
from pathlib import Path
import argparse
import json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--genes',required=True,help='One RNA gene symbol per line; fixes output head order')
    p.add_argument('--scgpt-dir',required=True,help='best_model.pt and vocab.json')
    p.add_argument('--out',required=True)
    a=p.parse_args()
    import numpy as np
    import torch
    genes=[g.strip() for g in Path(a.genes).read_text().splitlines() if g.strip()]
    if not genes or len(set(genes))!=len(genes):raise ValueError('Genes must be unique and nonempty')
    root=Path(a.scgpt_dir)
    state=torch.load(root/'best_model.pt',map_location='cpu',weights_only=False)
    vocab=json.loads((root/'vocab.json').read_text())
    w=state['encoder.embedding.weight'].float()
    norm=torch.nn.LayerNorm(w.shape[1]);norm.weight.data=state['encoder.enc_norm.weight'].float()
    norm.bias.data=state['encoder.enc_norm.bias'].float()
    upper={k.upper():v for k,v in vocab.items()}
    ids=[vocab.get(g,upper.get(g.upper(),-1)) for g in genes]
    covered=np.array([i>=0 for i in ids]);emb=np.zeros((len(genes),w.shape[1]),np.float32)
    with torch.no_grad():emb[covered]=norm(w[torch.tensor([i for i in ids if i>=0])]).numpy()
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.out,genes=np.array(genes),emb=emb,covered=covered)


if __name__=='__main__':main()
