#!/usr/bin/env python
"""Predict single-cell expression from H&E with automatic segmentation and reference retrieval."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--image', help='H&E WSI, TIFF, PNG or JPEG; segmented automatically if --cells is omitted')
    source.add_argument('--prepared', help='Prepared cell crops and patch_cell_boundaries.npz')
    p.add_argument('--cells', help='Optional existing segmentation in source image pixels')
    p.add_argument('--mpp', type=float, help='Microns per source pixel; required if absent from slide metadata')
    p.add_argument('--release', required=True, help='Portable model directory containing release.json')
    p.add_argument('--out', required=True, help='Output .h5ad; matrices are streamed to disk')
    p.add_argument('--tissue', default='', help='Tissue key for fusion weights; omitted or unknown uses global weights')
    p.add_argument('--genes', default='all', help='all, xenium, or a text file of gene symbols (one per line)')
    p.add_argument('--direct-only', action='store_true', help='Run A without bank or gate')
    p.add_argument('--device', default='cuda')
    p.add_argument('--amp', choices=['bf16', 'fp16', 'fp32'], default='bf16')
    p.add_argument('--enc-chunk', type=int, default=64)
    p.add_argument('--query-batch', type=int, default=256)
    p.add_argument('--target-cells', type=int, default=128)
    p.add_argument('--save-embeddings', action='store_true')
    p.add_argument('--segment-window', type=int, default=2048)
    p.add_argument('--segment-overlap', type=int, default=128)
    p.add_argument('--segment-model', help='Local Cellpose-SAM weights; default downloads official cpsam')
    p.add_argument('--exclude', nargs='*', default=[], help='Reference objects from this donor/block to exclude')
    p.add_argument('--sample-id', help='Target identity for self-reference exclusion; defaults to input basename')
    p.add_argument('--overwrite', action='store_true')
    return p


def main(argv=None):
    a = parser().parse_args(argv)
    if a.mpp is not None and (not 0 < a.mpp < 100):
        raise ValueError('--mpp must be positive microns per pixel')
    if min(a.enc_chunk, a.query_batch, a.target_cells) < 1:
        raise ValueError('Batch and tile sizes must be positive')
    if a.prepared and a.cells:
        raise ValueError('--cells applies only to --image')
    out = Path(a.out).resolve()
    partial = out.with_name(out.name+'.partial')
    if not a.overwrite and (out.exists() or partial.exists()):
        raise FileExistsError(f'Output/partial exists: {out}; choose a new path or --overwrite')
    out.parent.mkdir(parents=True, exist_ok=True)
    from voice.release import Release
    release = Release(a.release, retrieval=not a.direct_only)
    if a.image and not a.cells:
        from predict.slide_io import SlideReader
        reader = SlideReader(a.image)
        mpp = a.mpp if a.mpp is not None else reader.mpp
        reader.close()
        if mpp is None:
            raise ValueError('Image has no physical resolution metadata. Supply --mpp (um/pixel).')
        a.mpp = mpp
        cells = out.with_name(out.stem+'.cells.npz')
        if cells.exists() and not a.overwrite:
            raise FileExistsError(f'{cells} exists. Pass it explicitly with --cells, or choose another output.')
        command = [sys.executable, str(Path(__file__).with_name('segment.py')), '--image', a.image,
                   '--out', str(cells), '--mpp', str(mpp), '--window', str(a.segment_window),
                   '--overlap', str(a.segment_overlap)]
        if a.segment_model:
            command += ['--model', a.segment_model]
        if a.device == 'cpu':
            command += ['--cpu']
        subprocess.run(command, check=True)
        a.cells = str(cells)
    import gc
    import h5py
    import numpy as np
    import pandas as pd
    import torch
    from anndata.io import write_elem
    from predict.model import Voice25M
    from predict.inputs import open_slide
    from predict.covariates import geometry
    from predict.tiling import tiles_for
    from voice.gate import apply_gate
    from voice.gate_rules import select_gate, validate_gate_genes
    from voice.retrieval import ReferenceBank
    src = open_slide(a.image, a.cells, a.prepared, a.mpp)
    if not len(src):
        raise ValueError('No cells in input')
    if len(set(src.ids.tolist())) != len(src):
        raise ValueError('Cell identifiers must be unique')
    pos_um, aa, ar, mpp = geometry(src, a.mpp)
    model = Voice25M(release, a.device, a.amp)
    genes = model.genes
    if a.genes == 'all':
        names = genes.tolist()
    elif a.genes == 'xenium':
        if 'xenium_genes' not in release.config:
            raise ValueError('Release does not provide xenium_genes; use --genes all or a gene list')
        names = (release.root/release.config['xenium_genes']).read_text().splitlines()
    else:
        names = [g.strip() for g in Path(a.genes).read_text().splitlines() if g.strip()]
    head_of = {g:i for i,g in enumerate(genes)}
    if not names or len(set(names)) != len(names) or any(g not in head_of for g in names):
        raise ValueError('Requested genes must be nonempty, unique, and present in the model gene head')
    panel = np.array([head_of[g] for g in names], np.int64)
    n, g = len(src), len(panel)
    side, tiles = tiles_for(pos_um, a.target_cells)
    source, beta_map = 'direct', {}
    if not a.direct_only:
        validate_gate_genes(release.gate, genes.tolist())
        source, beta_map = select_gate(release.gate, a.tissue, g)
    var = pd.DataFrame({'global_gene_id': panel}, index=pd.Index(names, name='gene'))
    obs = pd.DataFrame({'y_pixel':src.pos[:,0], 'x_pixel':src.pos[:,1],
                        'y_um':pos_um[:,0], 'x_um':pos_um[:,1]},
                       index=pd.Index(src.ids.astype(str),name='cell_id'))
    info = {'model':'VOICE-25M-scGPT', 'stage1_sha256':release.hashes['stage1'],
            'stage2_sha256':release.hashes['stage2'], 'gene_space_sha256':release.hashes['gene_space'],
            'mpp':mpp, 'crop_um':55., 'tile_side_um':side, 'gate_source':source,
            'direct_only':a.direct_only, 'tissue':a.tissue, 'source':src.kind, 'amp':a.amp, 'spatial_order':'x,y in source pixels',
            'A_scale':'log1p(mu)', 'R_scale':'per-reference population z score of log1p(counts)',
            'X_scale':'A' if a.direct_only else 'beta*A+(1-beta)*R; per-gene scores, not counts'}
    print(f'[predict] {n:,} cells; {g:,} genes; {len(tiles):,} tiles', flush=True)
    tensor = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=model.device)
    hi = tensor(panel)
    with h5py.File(partial, 'w') as h:
        h.attrs.update({'encoding-type':'anndata','encoding-version':'0.1.0','complete':False})
        for key, value in [('obs',obs),('var',var),('obsm',{}),('layers',{}),('uns',{}),
                           ('varm',{}),('obsp',{}),('varp',{})]:
            write_elem(h,key,value)
        def matrix(parent,key,shape,dtype='float32'):
            ds = parent.create_dataset(key,shape=shape,dtype=dtype,
                                       chunks=(min(n,128),min(shape[1],512)),compression='lzf')
            ds.attrs.update({'encoding-type':'array','encoding-version':'0.2.0'})
            return ds
        A = matrix(h['layers'],'A',(n,g))
        H1 = matrix(h['obsm'],'X_voice_phase1',(n,1536),'float16') if not a.direct_only or a.save_embeddings else None
        X = matrix(h,'X',(n,g))
        if a.save_embeddings:
            H2 = matrix(h['obsm'],'X_voice_stage2',(n,1536),'float16')
            Z = matrix(h['obsm'],'X_voice_se2',(n,model.args['d_model']),'float16')
        for ti, indices in enumerate(tiles):
            indices = np.sort(indices)
            images, masks = src.batch(indices)
            images, masks = tensor(images), tensor(masks)
            f2 = model.encode(images,masks,'stage2',a.enc_chunk)
            direct,z = model.decode(f2,tensor(pos_um[indices]),tensor(aa[indices]),tensor(ar[indices]),hi)
            direct = direct.cpu().numpy()
            if not np.isfinite(direct).all():
                raise RuntimeError('Nonfinite direct prediction')
            A[indices] = direct
            if a.direct_only:
                X[indices] = direct
            if a.save_embeddings:
                H2[indices] = f2.cpu().numpy(); Z[indices] = z.cpu().numpy()
            if H1 is not None:
                H1[indices] = model.encode(images,masks,'phase1',a.enc_chunk).cpu().numpy()
            if (ti+1)%20 == 0 or ti+1 == len(tiles):
                print(f'[A] tiles {ti+1}/{len(tiles)}',flush=True)
        del model, images, masks, f2, direct, z, hi
        if getattr(src,'_rd',None) is not None:
            src._rd.close(); src._rd = None
        gc.collect()
        if a.device.startswith('cuda'):
            torch.cuda.empty_cache()
        beta = np.ones(g,np.float32)
        covered = np.zeros(g,bool)
        if not a.direct_only:
            target_id = a.sample_id or Path(a.prepared or a.image).stem
            excluded = set(a.exclude) | {target_id}
            bank = ReferenceBank(release.paths['bank'],device=a.device,exclude=excluded)
            bank.configure(panel)
            R = matrix(h['layers'],'R',(n,g))
            covered = bank.covered
            for start in range(0,n,a.query_batch):
                end = min(start+a.query_batch,n)
                retrieved,_ = bank.retrieve(H1[start:end],50)
                fused,beta = apply_gate(A[start:end],retrieved,panel,beta_map)
                R[start:end] = retrieved; X[start:end] = fused
                print(f'[retrieval + fusion] {end:,}/{n:,} cells',flush=True)
            info.update({'K':50,'tau':0.03,'cluster_target_size':100,
                         'bank_fingerprint':bank.model['cache_fingerprint'],
                         'excluded_objects':sorted(excluded & set(bank.model['objects']))})
        var['beta'] = beta; var['covered'] = covered
        del h['var']; write_elem(h,'var',var)
        write_elem(h['obsm'],'spatial',src.pos[:,::-1].copy())
        if H1 is not None and not a.save_embeddings:
            del h['obsm']['X_voice_phase1']
        write_elem(h['uns'],'voice25m',info)
        h.attrs['complete'] = True
        h.flush()
    os.replace(partial,out)
    print(f'[done] {out}',flush=True)


if __name__ == '__main__':
    main()
