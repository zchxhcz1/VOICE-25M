# VOICE-25M

Predict spatial single-cell gene expression from H&E images using an image encoder, a spatial decoder, and a reference bank of cells with measured expression.

## Resources

- **Model:** [Hugging Face](https://huggingface.co/zchx/VOICE-25M)
- **Paper:** [arXiv:2608.08366](https://arxiv.org/abs/2608.08366)

## Installation

Use Python 3.10 or later and install a PyTorch build suitable for your hardware.

```bash
git clone https://github.com/zchxhcz1/VOICE-25M.git
cd VOICE-25M
pip install -r requirements.txt
```

For SVS, NDPI, or MRXS images, install the OpenSlide system library and `openslide-python`. PNG, JPEG, and TIFF images are also supported. Large slide images should use an OpenSlide-readable format to avoid loading the entire image into memory.

## Download the model

Download the checkpoints, fusion weights, gene information, and reference bank from Hugging Face. UNI2-h requires separate access through [MahmoodLab/UNI2-h](https://huggingface.co/MahmoodLab/UNI2-h).

After receiving UNI2-h access, authenticate with your own Hugging Face account:

```bash
hf auth login
python - <<'PYTHON'
from huggingface_hub import snapshot_download, hf_hub_download

model_dir = "models/voice25m"
snapshot_download("zchx/VOICE-25M", local_dir=model_dir)
hf_hub_download("MahmoodLab/UNI2-h", "pytorch_model.bin", local_dir=f"{model_dir}/uni2h")
PYTHON
```

See [model files](weights/README.md) for the directory structure.

## Prediction

Choose your own model, image and output locations in the commands below. Run prediction from a single H&E image:

```bash
python predict/predict.py \
  --image /path/to/slide.svs \
  --release models/voice25m \
  --out results/prediction.h5ad
```

The command segments nuclei, extracts cell-centered image crops, predicts expression, retrieves reference expression, and combines both predictions. It also saves the segmentation beside the output. The model download includes the calibrated fusion weights and reference bank used by this command. The first segmentation run downloads the Cellpose model; use `--segment-model` to provide local segmentation weights.

Spatial prediction uses up to 128 target cells and 64 neighboring context cells per block, with context selected within 137 µm. All cells in the block participate in attention; only target predictions are saved. Every cell is output once. Partitioning is deterministic for a given sample ID and fixed seed.

If the image does not contain physical resolution metadata, supply its actual microns per pixel:

```bash
python predict/predict.py --image /path/to/image.png --mpp 0.25 \
  --release models/voice25m --out results/prediction.h5ad
```

`0.25` is an example. Use the resolution of your image. Prediction requires no measured expression or precomputed image features for the target slide.

| Option | Purpose |
| --- | --- |
| `--cells cells.npz` | Use existing cell boundaries instead of automatic segmentation |
| `--prepared /path/to/slide` | Read prepared cell crops and boundaries instead of `--image` |
| `--tissue xenium_lung` | Select tissue-specific fusion weights; omitted or unknown tissue uses global weights |
| `--genes all` | Predict all 18,580 model genes; this is the default |
| `--genes genes.txt` | Predict an ordered list of gene symbols, one per line |
| `--save-embeddings` | Save image and spatial features |
| `--workers 4` | Set CPU workers for loading cell crops |
| `--enc-chunk 32 --query-batch 64` | Reduce transient GPU memory use |
| `--sample-id SAMPLE --exclude RELATED_SAMPLE` | Exclude the target and related references when they occur in the bank |

CUDA is the default device. CPU inference is available with `--device cpu --amp fp32`. Prediction matrices are written to disk in batches; output size scales with the number of cells and requested genes. Automatic segmentation produces nuclei; use supplied cell boundaries when available.

## Output

```python
import anndata as ad
prediction = ad.read_h5ad("results/prediction.h5ad")
```

| Field | Contents |
| --- | --- |
| `X` | Combined gene-expression scores |
| `layers['A']` | Direct predictions in `log1p(mu)` scale |
| `layers['R']` | Reference expression scores; NaN where reference coverage is unavailable |
| `var['beta']` | Applied fusion weight for each gene |
| `var['covered']` | Reference coverage for each gene |
| `obsm['spatial']` | Cell centroids as x,y coordinates in source-image pixels |

The combined score is `beta * A + (1 - beta) * R`. Genes without reference coverage or fitted weights use the direct prediction. Combined scores describe variation across cells within each gene; they are not molecule counts.

## Training

[Training instructions](docs/training.md) cover RNA embedding preparation, image–RNA alignment, expression training, reference-bank construction, and fitting fusion weights. Training requires paired H&E images and measured spatial expression.

[Input formats](docs/formats.md) describe image geometry, prepared-slide files, reference-bank inputs, and output coordinates.
