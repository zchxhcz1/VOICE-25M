#!/usr/bin/env python
"""Check model/gene/bank/gate consistency before processing images."""
from pathlib import Path
import argparse
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',required=True)
    p.add_argument('--direct-only',action='store_true')
    p.add_argument('--device',default='cpu')
    a=p.parse_args()
    from voice.release import Release
    from voice.retrieval import CommunityR100
    from voice.gate_rules import validate_gate_genes
    import numpy as np
    r=Release(a.release,not a.direct_only)
    genes=np.load(r.paths['gene_space'],allow_pickle=False)['genes'].astype(str).tolist()
    if not a.direct_only:
        validate_gate_genes(r.gate,genes)
        bank=CommunityR100(r.paths['bank'],a.device,verify_hashes=True)
        print(f'{len(genes)} genes; {len(bank.banks)} objects; {bank.n_communities} clusters; comm100/K50/V3')
    print('Release validation passed')


if __name__=='__main__':main()
