# Input/output contracts

## H&E images

Use an RGB image in WSI, TIFF, PNG or JPEG format. Supply the physical source resolution in µm/pixel with `--mpp` when it is absent from WSI metadata. Crops cover 55 µm and are resampled to 224×224. Polygon masks are rasterized in the same rounded crop frame and pooled to a normalized 16×16 token mask. Degenerate polygons use the central token, matching the model's fallback.

An optional `cells.npz` contains `y_pixel`, `x_pixel` [N], `cell_id` [N], `indptr` [N+1], and concatenated `vertex_x`, `vertex_y` [V]. Centroids and polygon vertices use source-image pixels. Optional scalar `mpp` is used if the image metadata and command line do not specify resolution. Vertices should describe cell boundaries when available; automatic segmentation produces nuclei.

## Prepared slides

```text
slide/
  dataset.json
  manifest.csv.gz
  patch_cell_boundaries.npz
  patches/...
  expression.npz             # training/reference preparation only
  genes.tsv                  # training/reference preparation only
```

`dataset.json`: `um_per_pixel_source`, `um_per_pixel_output` (=55/224), `patch_size` (=224), `patch_size_um` (=55).

`manifest.csv.gz`: CSV columns `expr_row`, `patch_path`, with paths relative to the slide directory. Every boundary row must have a crop. `patch_cell_boundaries.npz`: `expr_rows` [N] indexes the expression matrix; `indptr` [N+1]; `vertex_x_patch`, `vertex_y_patch` in crop-local pixels; `y_pixel`, `x_pixel` in source-image pixels; scalar `output_size` (=224); optional `crop_px`. Cells follow boundary row order, and all arrays must agree on that order.

`expression.npz` is a SciPy CSR raw-count matrix [all expression rows, measured genes]. `genes.tsv` is a tab-separated file with a header and gene symbols in its first column. Supply RNA features only: filter protein, negative-control and non-gene channels before training/reference preparation. No target expression file is opened during prediction.

## Cluster bank

The bank builder consumes a JSON manifest with explicit paths. `train/prepare_bank.py` generates it from a direct release and a training-reference list. Each object supplies row-aligned `E.npy` [N,1536] raw Stage-1 features, `Y.npy` [N,P] fp16 expression values, `panel.npy` [P] global gene IDs and feature provenance.

The expression protocol is log1p counts → per-object population z-score (standard deviation floor 1e-3) → fp16. K-means uses PCA64 of normalized Stage-1 features, at most 100,000 PCA fitting rows per object and 10 Lloyd iterations. It creates floor(N/100), at least one, initial clusters, drops empty clusters and renumbers them. Each key is the normalized mean of the original fp16 unit 1536-dimensional features; each value is the float32 mean of the existing z-expression. Values are not standardized again and cluster counts do not weight retrieval.

Retrieval scores original 1536-dimensional fp16 cosine similarities. Each object contributes at most its top 50 clusters. For each gene coverage group, the eligible candidates are merged and top 50 selected, with exponential weights at temperature 0.03. Threshold ties share the remaining mass within the retained candidate union; clusters beyond an object's top 50 are not recovered to extend ties. Unmeasured genes remain NaN in R.

## Output

Output is a standard, dense-on-disk AnnData `.h5ad`. A, R and X are float32; requested stored embeddings are float16. Coordinates in `obs` include pixels and microns; `obsm['spatial']` is x,y. Each cell belongs to exactly one spatial tile; all cell rows are retained. The final output is published by atomic rename after completion. A failed run can leave `.partial`; it is not a finished output. This entry point does not resume partial predictions: use a new output or explicitly `--overwrite`.
