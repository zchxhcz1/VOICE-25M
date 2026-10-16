# Data formats

## H&E images and cell boundaries

Images must be RGB. Supply physical source resolution in microns per pixel with `--mpp` when it is absent from image metadata. Each cell crop covers 55 µm and is resized to 224×224 pixels. Cell polygons are rasterized in that crop and pooled to a normalized 16×16 image-token mask.

Optional `cells.npz` fields:

| Field | Shape and meaning |
| --- | --- |
| `y_pixel`, `x_pixel` | Cell centroids, each [N], in source-image pixels |
| `cell_id` | Unique cell identifiers [N] |
| `indptr` | Polygon offsets [N+1] |
| `vertex_x`, `vertex_y` | Concatenated polygon vertices in source-image pixels |
| `mpp` | Optional scalar source resolution in microns per pixel |

## Prepared slides

```text
slide/
  dataset.json
  manifest.csv.gz
  patch_cell_boundaries.npz
  patches/...
  expression.npz
  genes.tsv
```

`expression.npz` and `genes.tsv` are required for training and reference preparation. Image prediction uses the image crops, geometry, and manifest.

| File | Required contents |
| --- | --- |
| `dataset.json` | `um_per_pixel_source`, `um_per_pixel_output` (=55/224), `patch_size` (=224), `patch_size_um` (=55) |
| `manifest.csv.gz` | CSV columns `expr_row`, `patch_path`; crop paths are relative to the slide directory |
| `patch_cell_boundaries.npz` | `expr_rows` [N], `indptr` [N+1], `vertex_x_patch`, `vertex_y_patch`, `y_pixel`, `x_pixel`, scalar `output_size`; optional `crop_px` |
| `expression.npz` | SciPy CSR raw-count matrix [expression rows, measured genes] |
| `genes.tsv` | Tab-separated table with a header and RNA gene symbols in its first column |

Boundary vertices ending in `_patch` use crop-local pixels; centroids use source-image pixels. `expr_rows` indexes the expression matrix, and all cell arrays follow boundary-row order. Every boundary row must have a crop. Filter protein, control, and non-gene channels before preparing training data.

## Reference-bank inputs

`train/prepare_bank.py` generates a manifest with one entry per reference slide. Each entry supplies:

- `E.npy`: raw image-alignment features [N,1536].
- `Y.npy`: standardized reference expression [N,P].
- `panel.npy`: ordered global gene indices [P].
- Feature metadata identifying the image checkpoint.

Reference expression is `log1p(counts)` standardized within each slide using population standard deviation with a floor of 1e-3, then stored as float16. Bank construction clusters normalized image features within each slide. Cluster keys are normalized feature means; cluster expression values are float32 means of the standardized expression. Each gene uses only references that measured it.

The training settings and retrieval parameters are described in [training.md](training.md).

## Prediction output

Output is an AnnData `.h5ad` file. Predictions and reference scores are float32; optional stored embeddings are float16. `obsm['spatial']` uses x,y source-image pixels. `obs` also contains pixel and micron coordinates. Each cell is predicted once.

The final file is written after all predictions complete. An interrupted run can leave a `.partial` file; choose a new output or use `--overwrite` to restart.
