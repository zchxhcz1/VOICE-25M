# VOICE-25M

Predict single-cell gene expression from an H&E slide using UNI2-h, scGPT-aligned LoRA, an SE(2) spatial decoder, and k-means reference retrieval.

This release uses **comm100 / K50 / V3**: approximately 100 cells per k-means cluster, 50 retrieved clusters per gene coverage group, temperature 0.03, and V3 per-gene fusion weights refitted for that bank. The gene head contains 18,580 genes. The `25M` name refers to the training-cell scale.

[中文说明](docs/README_CN.md) · [Training](docs/training.md) · [Model artifacts](weights/README.md) · [Input/output formats](docs/formats.md) · [Validation](docs/validation.md)

## Resources

- **Model:** [zchx/VOICE-25M](https://huggingface.co/zchx/VOICE-25M)

## Install

Python 3.10 or later; Python 3.11 is the tested environment. Install a CUDA-enabled PyTorch build appropriate for your machine, then:

```bash
pip install -r requirements.txt
```

For SVS/NDPI/MRXS, install the OpenSlide system library and `pip install openslide-python`. For very large TIFFs, use an OpenSlide-readable pyramidal TIFF; the array fallback loads TIFF/PNG/JPEG into host memory.

UNI2-h weights require access to [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h). The automatic segmentation uses the original `cpsam` model described in the [Cellpose documentation](https://cellpose.readthedocs.io/en/latest/models.html); it downloads that checkpoint on first use. `--segment-model` selects a local checkpoint for offline use.

## Predict from one H&E image

Obtain a compatible model release following [weights/README.md](weights/README.md). The source repository contains code and the calibrated beta JSON; large neural weights and the reference bank are separate artifacts. No target RNA, cell coordinates, masks, precomputed features, or training corpus are required.

Download the VOICE checkpoints, V3 beta and matching bank from [zchx/VOICE-25M](https://huggingface.co/zchx/VOICE-25M), then obtain the original UNI2-h weights from its official gated repository:

```bash
hf auth login
python - <<'PYTHON'
from huggingface_hub import snapshot_download, hf_hub_download
release_dir = "/models/voice25m"
snapshot_download(repo_id="zchx/VOICE-25M", local_dir=release_dir)
hf_hub_download("MahmoodLab/UNI2-h", "pytorch_model.bin", local_dir=f"{release_dir}/uni2h")
PYTHON
```

Use your own Hugging Face account with approved UNI2-h access. The original UNI2-h base weights are not redistributed in the VOICE repository. The downloaded release contains relative paths and is ready for inference once `uni2h/pytorch_model.bin` has been added.

```bash
python predict/predict.py \
  --image /data/my_slide.svs \
  --release /models/voice25m \
  --out /results/my_slide.h5ad
```

The command reads physical resolution from the WSI metadata, segments nuclei, constructs 55 µm crops and mask weights, runs the direct branch and comm100/K50 retrieval, applies V3 beta, and writes AnnData. The segmentation is also saved next to the output as `my_slide.cells.npz`.

For PNG/JPEG or an image without physical resolution metadata, supply the actual microns per pixel:

```bash
python predict/predict.py --image /data/he.png --mpp 0.25 \
  --release /models/voice25m --out /results/he.h5ad
```

`0.25` is an example, not an assumed scanner resolution. Unknown resolution produces an actionable error. Automatic segmentation estimates nuclei; supplied cell boundaries can better match the cell masks used in training.

Useful options:

- `--tissue xenium_lung`: select tissue-specific beta; omitted or unrecognized tissue uses global beta. Keys are listed in `weights/gate_v3.json`.
- `--genes all` (default), `--genes xenium`, or `--genes /data/genes.txt`: select the output panel. V3 panel width is the number of requested output genes.
- `--cells /data/cells.npz`: use existing segmentation, bypassing automatic segmentation.
- `--prepared /data/prepared_slide`: use prepared cell crops instead of `--image`.
- `--direct-only`: run the direct branch without a bank or gate.
- `--save-embeddings`: retain Stage-1/Stage-2 features and decoder embeddings.
- `--enc-chunk 32 --query-batch 64`: lower transient GPU memory use. The complete bank still needs to fit on the selected device.
- `--sample-id SAMPLE --exclude RELATED_SAMPLE`: exclude the target and related donor/block references. Supply identity explicitly when filenames differ from bank object IDs.

The default device is CUDA. CPU inference is available with `--device cpu --amp fp32` and can be slow. Output matrices are written in batches, so a whole slide's dense expression matrices do not have to fit in RAM. Disk use scales with cells × requested genes; each float32 matrix needs about 4 × N × G bytes before compression.

## Outputs and fusion

```python
import anndata as ad
prediction = ad.read_h5ad('/results/my_slide.h5ad')
print(prediction.shape)
print(prediction.obsm['spatial'])  # x,y in source-image pixels
```

| Field | Meaning |
| --- | --- |
| `X` | V3 fused gene scores; direct predictions in direct-only mode |
| `layers['A']` | Direct `log1p(mu)` predictions |
| `layers['R']` | Retrieved per-reference z scores; NaN for unsupported genes |
| `var['beta']` | Applied weight per gene |
| `var['covered']` | Whether the eligible bank measures the gene |
| `obsm['spatial']` | Cell centroids in source-image pixels, x then y |
| `uns['voice25m']` | Model hashes, bank fingerprint, geometry and fusion metadata |

`X_g = beta_g * A_g + (1-beta_g) * R_g`. Missing retrieval coverage or an unfitted beta uses A. A and R retain their original scales. Fused values describe within-gene cell rankings and spatial patterns; they are not absolute molecule counts. Use A when the direct expression scale is needed.

V3 first selects the same tissue and panel width (wide ≥1,000 genes), then the first eligible stratum of the same tissue with another width, then global beta. This version case-folds tissue keys, requires at least two references per stratum, and averages gene-wise beta over references with equal reference weight. Model/bank/gene/gate identities are checked before fused inference.

## Training and building your own release

[docs/training.md](docs/training.md) covers scGPT target preparation, the two training phases, bank construction, reference-only V3 fitting, and portable release export. Training needs paired H&E and measured spatial expression; prediction of a new slide does not.

```text
voice/      model, retrieval, fusion, training utilities
train/      preparation, Stage-1, Stage-2, beta fitting, release export
predict/    image reading, automatic segmentation, inference, bank construction
configs/    portable release example
weights/    calibrated V3 beta and artifact instructions
tests/     numerical and input-contract regression tests
```

Downstream analysis, clinical prediction, paper plots, benchmark exports, training datasets and local scheduler scripts are excluded.

## Citation

The code follows [VOICE_HE_ST_Pred](https://github.com/Luosanj/VOICE_HE_ST_Pred). See [provenance](docs/provenance.md) for this 25M version.

```bibtex
@article{luo2026voice,
  title={VOICE: A Vision-Omics Foundation Model Integrating Direct and Retrieval-Based Prediction of In-situ Single-Cell Gene Expression},
  author={Luo, Xin and Tao, Yicheng and Zeng, Haoxuan and Wang, Suyuan and Ouyang, Chenzi and Zhu, Meiqi and Liu, Kai and Chen, Shuibing and Liu, Jie},
  journal={arXiv preprint arXiv:2608.08366},
  year={2026}
}
```
