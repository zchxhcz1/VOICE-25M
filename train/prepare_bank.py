#!/usr/bin/env python
"""Encode reference slides and prepare features and expression for bank construction."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',required=True,help='Direct release from export_release.py (bank/gate not needed)')
    p.add_argument('--dataset',required=True)
    p.add_argument('--references',required=True,help='One training reference object per line')
    p.add_argument('--out',required=True)
    p.add_argument('--device',default='cuda')
    a=p.parse_args()
    import h5py
    import numpy as np
    import pandas as pd
    from scipy import sparse
    from voice.release import Release
    from voice.bank_io import FEATURE_PROTOCOL,VALUE_PROTOCOL,write_json
    from voice.training_contract import read_ids,dataset_contract
    from voice.tissue import object_directory
    release=Release(a.release,retrieval=False);out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=True)
    genes=np.load(release.paths['gene_space'],allow_pickle=False)['genes'].astype(str)
    lookup={g:i for i,g in enumerate(genes)}
    refs=read_ids(a.references);contract=dataset_contract(a.dataset)
    if contract and not set(refs)<=set(contract['training_set']):raise ValueError('Bank references must be training samples')
    objects=[]
    for sample in refs:
        d=object_directory(a.dataset,sample);work=out/sample;work.mkdir(exist_ok=True)
        symbols=pd.read_csv(d/'genes.tsv',sep='\t').iloc[:,0].astype(str).tolist()
        cols=[i for i,g in enumerate(symbols) if g in lookup]
        if not cols:raise ValueError(f'{sample}: no genes match release')
        names=[symbols[i] for i in cols]
        if len(set(names))!=len(names):raise ValueError('Duplicate measured gene symbols')
        (work/'genes.txt').write_text('\n'.join(names)+'\n')
        prediction=work/'direct.h5ad'
        if not prediction.exists():
            subprocess.run([sys.executable,str(Path(__file__).resolve().parent.parent/'predict/predict.py'),
                            '--prepared',str(d),'--release',str(release.root),'--out',str(prediction),
                            '--genes',str(work/'genes.txt'),'--direct-only','--save-embeddings','--device',a.device],check=True)
        with h5py.File(prediction,'r') as h:
            if not h.attrs.get('complete') or h['uns/voice25m/stage1_sha256'].asstr()[()]!=release.hashes['stage1']:
                raise ValueError('Cached reference predictions do not match this release')
            er=h['obs/cell_id'].asstr()[:].astype(np.int64)
            np.save(work/'E.npy',h['obsm/X_voice_phase1'][:])
        matrix=sparse.load_npz(d/'expression.npz').tocsr()[er][:,cols]
        # Match the original bank: population z-score log1p counts, sd floor 1e-3, fp16 before cluster means.
        log=matrix.astype(np.float32);log.data=np.log1p(log.data)
        mean=np.asarray(log.mean(0)).ravel().astype(np.float32)
        variance=np.maximum(np.asarray(log.power(2).mean(0)).ravel()-mean*mean,0)
        sd=np.maximum(np.sqrt(variance),1e-3)
        values=np.lib.format.open_memmap(work/'Y.npy',mode='w+',dtype='float16',shape=matrix.shape)
        for begin in range(0,len(er),4096):
            values[begin:begin+4096]=(log[begin:begin+4096].toarray()-mean)/sd
        values.flush();del values
        np.save(work/'panel.npy',np.array([lookup[g] for g in names],np.int64))
        write_json(work/'feature_meta.json',{'checkpoint_sha256':release.hashes['stage1'],
                   'feature_protocol_id':FEATURE_PROTOCOL,'dim':1536,'output_field':'phase1'})
        objects.append({'object':sample,'split':'train','n_cells':len(er),'feature_meta':str(work/'feature_meta.json'),
                        **{k:{'path':str(work/f'{k}.npy')} for k in ['E','Y','panel']}})
    write_json(out/'manifest.json',{'checkpoint':str(release.paths['stage1']),
               'gene_space':str(release.paths['gene_space']),'feature_protocol_id':FEATURE_PROTOCOL,
               'value_protocol_id':VALUE_PROTOCOL,'objects':objects})
    print(out/'manifest.json')


if __name__=='__main__':main()
