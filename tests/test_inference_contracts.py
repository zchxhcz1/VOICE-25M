"""Numerical and portable-input regressions for the public inference path."""
from pathlib import Path
import argparse
import json
import numpy as np
import pytest
from voice.gate import apply_gate,fit_beta_reference
from voice.gate_rules import select_gate
from voice.bank_io import FEATURE_PROTOCOL,VALUE_PROTOCOL,write_json
from voice.bank_builder import build
from voice.retrieval import CommunityR100,sha
from predict.tiling import tiles_for
from predict.inputs import WSISlide
from PIL import Image


def test_v3_same_tissue_other_width_then_global():
    gate={'version':'v3','tissue_casefold':True,'min_refs':2,'panel_cut':1000,
          'strata_order':['lung|narrow','lung|wide'],
          'strata':{'lung|narrow':{'n_refs':2,'beta':{'0':.25}},
                    'lung|wide':{'n_refs':1,'beta':{'0':.8}}},'global':{'beta':{'0':.5}}}
    assert select_gate(gate,'LUNG',18580)==('lung|narrow',{0:.25})
    assert select_gate(gate,'unknown',18580)==('global',{0:.5})


def test_uncovered_and_unfitted_genes_keep_direct():
    A=np.arange(12,dtype=np.float32).reshape(4,3)
    R=np.ones_like(A)*2;R[:,1]=np.nan
    fused,beta=apply_gate(A,R,np.arange(3),{0:.25,1:0.})
    np.testing.assert_allclose(fused[:,0],.25*A[:,0]+1.5)
    np.testing.assert_array_equal(fused[:,1:],A[:,1:])
    np.testing.assert_array_equal(beta,[.25,1.,1.])


@pytest.mark.parametrize('pos',[np.zeros((1,2)),np.zeros((900,2)),np.column_stack([np.arange(1000),np.zeros(1000)]),
                                   np.random.default_rng(4).normal(size=(2000,2))])
def test_every_cell_is_predicted_once(pos):
    _,tiles=tiles_for(pos)
    np.testing.assert_array_equal(np.sort(np.concatenate(tiles)),np.arange(len(pos)))
    assert max(map(len,tiles))<=512


def test_image_resolution_from_cells_and_rounded_crop(tmp_path):
    img=np.arange(99*99*3,dtype=np.uint8).reshape(99,99,3)
    Image.fromarray(img).save(tmp_path/'image.png')
    np.savez(tmp_path/'cells.npz',y_pixel=[45.4],x_pixel=[47.6],mpp=.3,cell_id=['a'],
             indptr=[0,4],vertex_x=[45,51,51,45],vertex_y=[42,42,48,48])
    slide=WSISlide(tmp_path/'image.png',tmp_path/'cells.npz')
    assert slide.mpp==.3
    crop,mask=slide.batch([0]);assert crop.shape==(1,3,224,224)
    np.testing.assert_allclose(mask.sum(),1.)


def test_missing_resolution_rejected(tmp_path):
    Image.new('RGB',(20,20)).save(tmp_path/'image.png')
    np.savez(tmp_path/'cells.npz',y_pixel=[1],x_pixel=[1])
    with pytest.raises(ValueError,match='resolution'):
        WSISlide(tmp_path/'image.png',tmp_path/'cells.npz')


@pytest.fixture
def bank(tmp_path):
    rng=np.random.default_rng(12)
    np.savez(tmp_path/'space.npz',genes=np.array(['a','b','unmeasured']))
    (tmp_path/'stage1.pt').write_bytes(b'test-stage1')
    objects=[]
    for i in range(2):
        root=tmp_path/f'slide{i}';root.mkdir()
        E=rng.normal(size=(100,1536)).astype(np.float16)
        # Each reference measures a different gene, so their values must never be zero-filled together.
        np.save(root/'E.npy',E);np.save(root/'Y.npy',np.full((100,1),i+2,dtype=np.float16))
        np.save(root/'panel.npy',np.array([i],np.int64))
        write_json(root/'feature.json',{'checkpoint_sha256':sha(tmp_path/'stage1.pt'),
                   'feature_protocol_id':FEATURE_PROTOCOL,'dim':1536,'output_field':'phase1'})
        objects.append({'object':f'slide{i}','split':'train','feature_meta':str(root/'feature.json'),
                        **{k:{'path':str(root/f'{k}.npy')} for k in ['E','Y','panel']}})
    write_json(tmp_path/'manifest.json',{'objects':objects,'checkpoint':str(tmp_path/'stage1.pt'),
               'gene_space':str(tmp_path/'space.npz'),'feature_protocol_id':FEATURE_PROTOCOL,
               'value_protocol_id':VALUE_PROTOCOL})
    build(argparse.Namespace(device='cpu',backend='torch',manifest=str(tmp_path/'manifest.json'),
                             expected_objects=None,output=str(tmp_path/'bank')))
    return tmp_path/'bank'


def test_bank_coverage_and_exclusion(bank):
    query=np.random.default_rng(4).normal(size=(3,1536)).astype(np.float16)
    model=CommunityR100(bank,device='cpu');model.configure(np.arange(3))
    R,_=model.retrieve(query,50)
    np.testing.assert_allclose(R[:,:2],np.tile([2.,3.],(3,1)),atol=1e-5)
    assert np.isnan(R[:,2]).all()
    model=CommunityR100(bank,device='cpu',exclude=['slide0']);model.configure(np.arange(3))
    R,_=model.retrieve(query,50)
    assert np.isnan(R[:,0]).all();np.testing.assert_allclose(R[:,1],3.,atol=1e-5)


def test_bank_corruption_rejected(bank):
    p=bank/'summaries/object_00_comm100/values.npy'
    with p.open('r+b') as handle:handle.seek(-1,2);handle.write(b'!')
    with pytest.raises(ValueError,match='Corrupt'):
        CommunityR100(bank,device='cpu')
