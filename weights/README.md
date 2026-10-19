# Model files

Download the model from [Hugging Face](https://huggingface.co/zchx/VOICE-25M). Obtain `uni2h/pytorch_model.bin` separately through [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h), using an account with approved access.

The complete directory used by `--release` contains:

```text
voice25m/
  release.json
  stage1.pt
  stage2.pt
  gene_space.npz
  xenium_genes.tsv
  beta.json
  beta.meta.json
  uni2h/pytorch_model.bin
  bank/
    model.json
    bank_complete.json
    summaries/object_00/...
    ...
```

`release.json` supplies relative paths to the model files. Keep the checkpoints, gene information, fusion weights, and bank together. `genes.tsv` and `xenium_genes.tsv` in this source repository list model gene symbols, one per line without a header.

To package a newly trained model:

```bash
python train/export_release.py \
  --uni2h /models/uni2h \
  --stage1 /runs/voice/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/voice/checkpoints/expression/stage2_last.pt \
  --space /runs/voice/direct/gene_space_all.npz \
  --xenium-space /runs/voice/direct/gene_space_xen.npz \
  --bank /runs/bank --beta /runs/beta.json \
  --out /models/voice25m
```

The destination must be empty. Omit both `--bank` and `--beta` when exporting the intermediate checkpoints used by reference preparation during model training. Follow the UNI2-h access and distribution terms when sharing model files.
