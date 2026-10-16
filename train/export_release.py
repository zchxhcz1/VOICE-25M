#!/usr/bin/env python
"""Export an independent model release outside the source-code repository."""
from pathlib import Path
import argparse
import json
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ['uni2h','stage1','stage2','space','out']:
        p.add_argument('--'+key,required=True)
    p.add_argument('--bank');p.add_argument('--beta','--gate',dest='gate');p.add_argument('--xenium-space')
    a=p.parse_args()
    from voice.retrieval import sha
    out=Path(a.out).resolve()
    out.mkdir(parents=True,exist_ok=True)
    if any(out.iterdir()):raise ValueError('Export directory must be empty')
    config={'schema':'voice25m-release-v1','uni2h':'uni2h','stage1':'stage1.pt',
            'stage2':'stage2.pt','gene_space':'gene_space.npz'}
    for key,source in [('stage1',a.stage1),('stage2',a.stage2),('gene_space',a.space)]:
        shutil.copy2(source,out/config[key])
    shutil.copytree(a.uni2h,out/'uni2h')
    if bool(a.bank)!=bool(a.gate):raise ValueError('Supply both --bank and --beta')
    if a.bank:
        shutil.copytree(a.bank,out/'bank',symlinks=False)
        shutil.copy2(a.gate,out/'beta.json')
        shutil.copy2(Path(a.gate).with_suffix('.meta.json'),out/'beta.meta.json')
        config.update(bank='bank',gate='beta.json')
    if a.xenium_space:
        import numpy as np
        genes=np.load(a.xenium_space,allow_pickle=False)['genes'].astype(str).tolist()
        (out/'xenium_genes.tsv').write_text('\n'.join(genes)+'\n')
        config['xenium_genes']='xenium_genes.tsv'
    config['sha256']={k:sha(out/config[k]) for k in ['stage1','stage2','gene_space']}
    (out/'release.json').write_text(json.dumps(config,indent=2)+'\n')
    from voice.release import Release
    Release(out,retrieval=bool(a.bank))
    print(out)


if __name__=='__main__':main()
