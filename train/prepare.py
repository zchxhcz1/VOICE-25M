#!/usr/bin/env python
"""Create portable gene-space tables from a prepared corpus and explicit gene embeddings."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True)
    p.add_argument('--work-dir',required=True)
    p.add_argument('--gene-table',required=True,help='NPZ: genes (ordered symbols), emb [G,D]; scGPT gene embeddings')
    a=p.parse_args()
    import numpy as np
    import pandas as pd
    from voice.training_contract import dataset_contract
    root=Path(a.dataset).resolve(); work=Path(a.work_dir).resolve()/'direct'
    work.mkdir(parents=True,exist_ok=True)
    contract=dataset_contract(root)
    if contract is None:
        raise ValueError('Supply the documented split files and MANIFEST.csv')
    table=np.load(a.gene_table,allow_pickle=False)
    genes=table['genes'].astype(str); emb=table['emb'].astype(np.float32)
    if emb.ndim!=2 or len(emb)!=len(genes) or len(set(genes))!=len(genes) or not np.isfinite(emb).all():
        raise ValueError('Invalid gene embedding table')
    objects=[]; platforms=[]; splits=[]; masks=[]; xenium=set()
    from voice.tissue import object_directory
    for obj in contract['training_set']+contract['cross_slide_test_set']:
        d=object_directory(root,obj)
        meta=json.loads((d/'dataset.json').read_text())
        if int(meta['patch_size'])!=224 or float(meta['patch_size_um'])!=55.:
            raise ValueError(f'{obj}: crops must be 224px and 55um')
        symbols=set(pd.read_csv(d/'genes.tsv',sep='\t').iloc[:,0].astype(str))
        platform=0 if d.parent.parent.name=='xenium' else 1
        if obj in contract['training_set'] and platform==0:
            xenium |= symbols
        objects.append(obj);platforms.append(platform);splits.append(0 if obj in contract['training_set'] else 1)
        masks.append([g in symbols for g in genes])
    np.savez_compressed(work/'gene_space_all.npz',genes=genes,gene_emb=emb,objects=np.array(objects),
                        platform=np.array(platforms,np.int8),split=np.array(splits,np.int8),
                        panel_mask=np.asarray(masks,bool),um_per_px=np.float64(55/224))
    np.savez_compressed(work/'gene_space_xen.npz',genes=np.array([g for g in genes if g in xenium]))
    np.savez_compressed(work/'gene_emb_scgpt.npz',genes=genes,emb=emb)
    print(f'{work}: {len(genes)} genes; {len(objects)} objects')


if __name__=='__main__':main()
