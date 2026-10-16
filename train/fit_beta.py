#!/usr/bin/env python
"""Fit gene-wise fusion weights on measured reference slides."""
from pathlib import Path
import argparse
import json
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['release','bank','dataset','references','reference-inputs','out']:
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--batch-size',type=int,default=256)
    p.add_argument('--max-cells',type=int,default=20000,help='Deterministic calibration subsample per reference; 0 uses all cells')
    p.add_argument('--seed',type=int,default=1)
    a=p.parse_args()
    if a.batch_size<1 or a.max_cells<0:raise ValueError('Invalid fitting batch or sample size')
    import numpy as np
    import h5py
    from voice.release import Release
    from voice.retrieval import ReferenceBank,sha
    from voice.gate import fit_beta,_fit_eligible
    from voice.training_contract import read_ids,dataset_contract,gene_hash
    from voice.tissue import reference_exclusions,tissue_of_object
    release=Release(a.release,retrieval=False)
    genes=np.load(release.paths['gene_space'],allow_pickle=False)['genes'].astype(str).tolist()
    model=json.loads((Path(a.bank)/'model.json').read_text())
    if model['checkpoint_sha256']!=release.hashes['stage1'] or model['gene_space_sha256']!=release.hashes['gene_space']:
        raise ValueError('Bank and model identities differ')
    refs=read_ids(a.references);contract=dataset_contract(a.dataset)
    if contract and not set(refs)<=set(contract['gate_fit_references']):
        raise ValueError('Gate fitting references must belong to the approved training split')
    grouped={};all_fits=[];exclusions={}
    for sample in refs:
        work=Path(a.reference_inputs)/sample
        panel=np.load(work/'panel.npy')
        # Gate labels are measured log1p counts, whereas Y.npy is bank z-expression.
        from scipy import sparse
        import pandas as pd
        from voice.tissue import object_directory
        d=object_directory(a.dataset,sample)
        symbols=pd.read_csv(d/'genes.tsv',sep='\t').iloc[:,0].astype(str).tolist()
        columns=[symbols.index(genes[int(g)]) for g in panel]
        with h5py.File(work/'direct.h5ad','r') as h:
            from anndata.io import read_elem
            info=read_elem(h['uns/voice25m'])
            if info['stage1_sha256']!=release.hashes['stage1'] or info['stage2_sha256']!=release.hashes['stage2']:
                raise ValueError('Reference direct predictions belong to another model')
            er=read_elem(h['obs']).index.to_numpy().astype(np.int64)
            rng=np.random.default_rng(a.seed)
            take=(np.sort(rng.choice(len(er),a.max_cells,replace=False))
                  if a.max_cells and len(er)>a.max_cells else np.arange(len(er)))
            er=er[take]
            A=h['layers/A'][take].astype(np.float32);H=h['obsm/X_voice_phase1'][take]
            if not np.array_equal(read_elem(h['var'])['global_gene_id'].to_numpy(),panel):
                raise ValueError('Reference prediction columns do not match panel')
        counts=sparse.load_npz(d/'expression.npz').tocsr()[er][:,columns].astype(np.float32)
        counts.data=np.log1p(counts.data);Y=counts.toarray()
        excluded=reference_exclusions(a.dataset,sample,model['objects']);exclusions[sample]=excluded
        bank=ReferenceBank(a.bank,a.device,exclude=excluded);bank.configure(panel)
        R=np.empty_like(A)
        for start in range(0,len(H),a.batch_size):
            R[start:start+a.batch_size]=bank.retrieve(H[start:start+a.batch_size],50)[0]
        eligible=_fit_eligible(A,R,Y);beta,_=fit_beta(A,R,Y)
        fitted={int(g):float(b) for g,b in zip(panel[eligible],beta[eligible])}
        key=f'{tissue_of_object(a.dataset,sample)}|'+('wide' if len(panel)>=1000 else 'narrow')
        grouped.setdefault(key,[]).append(fitted);all_fits.append(fitted)
        del bank
        import gc;gc.collect()
        print(f'[beta] {sample}',flush=True)
    def aggregate(fits):
        acc={}
        for fit in fits:
            for g,b in fit.items():acc.setdefault(g,[]).append(b)
        return {'n_refs':len(fits),'beta':{str(g):float(np.mean(v)) for g,v in acc.items()}}
    gate={'version':'per_gene', 'fallback':'same_tissue','tissue_casefold':True,'panel_cut':1000,'min_refs':2,
          'strata_order':sorted(grouped),'strata':{k:aggregate(grouped[k]) for k in sorted(grouped)},
          'global':aggregate(all_fits),'fit_signature':{'genes':gene_hash(genes)},
          'branches':{'A':'log1p(mu)','R':'per-bank-slide z','fusion':'beta*A+(1-beta)*R'}}
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(gate,indent=2)+'\n')
    meta={'gate_sha256':sha(out),'checkpoint_sha256':release.hashes['stage1'],
          'stage2_checkpoint_sha256':release.hashes['stage2'],'gene_space_sha256':release.hashes['gene_space'],
          'cache_fingerprint':model['cache_fingerprint'],'retrieval_method':'cluster_retrieval',
          'K':50,'tau':0.03,'target_size':100,
          'reference_objects':refs,'reference_exclusions':exclusions,'fallback':'same_tissue',
          'max_calibration_cells':a.max_cells,'sampling_seed':a.seed,
          'tissue_casefold':True,'min_refs':2,'panel_cut':1000}
    out.with_suffix('.meta.json').write_text(json.dumps(meta,indent=2)+'\n')


if __name__=='__main__':main()
