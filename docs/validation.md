# Validation performed for this source release

Tested with Python 3.11, PyTorch 2.9.1+cu128, timm 1.0.16 and an NVIDIA H200. Additional tested dependency versions are recorded in [tested_versions.json](../configs/tested_versions.json).

- Ten regression tests passed: V3 fallback, uncovered/unfitted-gene behavior, spatial partitions including coincident and collinear cells, image resolution, crop masks, bank gene coverage, self-exclusion and corrupted-artifact rejection.
- A real H&E image crop was processed from the image alone, with physical resolution supplied. Automatic Cellpose segmentation produced eight nuclei; full-model prediction wrote an 8×18,580 AnnData file with A, R and V3 fused X. Standard `anndata.read_h5ad` and the fusion identity were checked.
- On the same image tensors and masks, the released direct predictions, Stage-1 features, Stage-2 features and spatial decoder features were exactly equal to the original 25M model implementation. On the same GPU and panel, comm100/K50 R and V3 fusion were also exactly equal to the original implementations.
- Stage-1 and Stage-2 each completed a small synthetic-data training run with the real UNI2-h architecture. scGPT cell/gene preparation was checked separately with synthetic scGPT weights. These are execution checks, not model-quality measurements.
- Reference preparation, the full-cell k-means builder and leave-self-out V3 fitting completed on a small synthetic corpus. The complete existing 82-object, 254,622-cluster release passed artifact and identity validation.
- All twelve command-line entry points passed help checks, Python sources compiled, and a wheel built successfully. Public source/configuration/docs were scanned for private machine paths; local Markdown links were checked.

This validation does not include a new full 25M training run, a whole-slide scale/performance benchmark, or segmentation-quality benchmarking. CPU and GPU fp16 matrix multiplication can select different threshold ties; numerical parity claims above apply to the tested GPU execution.

Run the portable regression suite with:

```bash
pip install -r requirements-dev.txt
python -m pytest -q
```
