# Training

Training consists of preparing RNA embeddings, aligning image and RNA features, fitting the expression model, building a reference bank, and fitting gene-wise fusion weights.

```bash
pip install -r requirements-train.txt
```

Use CUDA for model training. Commands below support one GPU with `python` or multiple GPUs with `torchrun`.

## Prepare the corpus

Prepare each slide as described in [formats.md](formats.md):

```text
corpus/
  MANIFEST.csv
  training_set/{xenium,cosmx}/TISSUE/SAMPLE/...
  cross_slide_test_set/{xenium,cosmx}/TISSUE/SAMPLE/...
  splits/
    training_set.txt
    inslide_test_set.txt
    cross_slide_test_set.txt
    stage2_global_training.txt
    retrieval_references.txt
    gate_fit_references.txt
  metadata/shared_origin_groups.csv
```

Each split file contains one globally unique sample name per line. Image–RNA alignment uses `training_set.txt`. The expression-training set equals the training set minus the in-slide test samples. Cross-slide test samples are disjoint from the training set. `retrieval_references.txt` and `gate_fit_references.txt` contain the expression-training samples. Empty test splits are allowed. `MANIFEST.csv` records the corpus used for training.

The optional `shared_origin_groups.csv` identifies related donor or tissue-block samples. It contains `field,samples`, with `field=donor_or_block_id` and semicolon-separated sample names. Related samples are excluded when fitting fusion weights for a reference slide.

## Prepare RNA embeddings

Obtain the scGPT whole-human files `best_model.pt`, `vocab.json`, and `args.json` from [scGPT](https://github.com/bowang-lab/scGPT). Generate cell embeddings in expression-matrix row order:

```bash
python train/embed_scgpt.py --dataset /data/corpus \
  --scgpt-dir /models/scgpt_human --out /runs/scgpt_cells
```

The extractor uses nonzero genes, a classification token, 51 expression bins, at most 1,200 tokens, and normalized 512-dimensional cell features. `--only sample_a,sample_b` and sharding options support processing subsets.

Create the ordered gene-embedding table and model gene space:

```bash
python train/build_gene_table.py --genes weights/genes.tsv \
  --scgpt-dir /models/scgpt_human --out /runs/gene_table.npz
python train/prepare.py --dataset /data/corpus --work-dir /runs/voice \
  --gene-table /runs/gene_table.npz
```

Gene embeddings use scGPT's embedding layer followed by encoder layer normalization. Symbols are matched exactly, then without case sensitivity. Genes outside the vocabulary start with zero embeddings and can be learned by the expression head. A different ordered gene list requires a corresponding model head, bank, and fitted fusion weights.

## Align image and RNA features

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase1.py \
  --dataset /data/corpus --work-dir /runs/voice \
  --uni2h /models/uni2h --cell-embeddings /runs/scgpt_cells
```

| Setting | Value |
| --- | --- |
| Training duration | 3 epochs, all usable cells, retained tail batches |
| Objective | Symmetric contrastive loss, temperature 0.07 |
| Optimizer | AdamW, weight decay 0.01 |
| Learning rates | Image adapters 1.5e-4; projection towers 1e-3 |
| Image adapters | Last 12 encoder blocks, rank 16, alpha 32, dropout 0.05 |
| Augmentation | Random rotations and reflections |
| Batch | 2,048 cells per rank, encoder chunks of 256 |

Output: `/runs/voice/checkpoints/alignment/stage1_final.pt`. Adjust `--chunk` and `--eff_batch` for your hardware. `--resume` restores weights, optimizer state, and data position; use the same world size and corpus when resuming.

## Train expression prediction

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase2.py \
  --dataset /data/corpus --work-dir /runs/voice --uni2h /models/uni2h \
  --stage1 /runs/voice/checkpoints/alignment/stage1_final.pt
```

| Setting | Value |
| --- | --- |
| Training duration | 3 epochs with complete target-cell coverage |
| Spatial decoder | 6 layers, 256 features, maximum distance 137 µm |
| Neighborhood | Up to 128 supervised cells and 64 additional context cells |
| Learning rates | Image adapters 3e-5; decoder and expression head 1e-4 |
| Expression head | Learnable gene embeddings initialized from scGPT |
| Objective | Negative-binomial loss plus log1p mean squared error |
| Encoder updates | First 3 adapted blocks frozen; remaining adapters trainable |
| Augmentation | Disabled during expression training |

Output: `/runs/voice/checkpoints/expression/stage2_last.pt`. Context cells affect image context but do not contribute expression loss. Each slide supervises only its measured genes. Gradient checkpointing is enabled; `--resume` restores training state.

## Build the reference bank

Package the direct model, encode reference slides, and construct the bank:

```bash
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/voice/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/voice/checkpoints/expression/stage2_last.pt \
  --space /runs/voice/direct/gene_space_all.npz \
  --xenium-space /runs/voice/direct/gene_space_xen.npz \
  --out /models/voice25m_direct
python train/prepare_bank.py --release /models/voice25m_direct \
  --dataset /data/corpus --references /data/corpus/splits/retrieval_references.txt \
  --out /runs/reference_inputs
python predict/build_bank.py --manifest /runs/reference_inputs/manifest.json \
  --output /runs/bank --device cuda --backend torch
```

Bank construction uses PCA with 64 components, fitted on up to 100,000 cells per slide, followed by 10 k-means iterations. The initial number of clusters is `max(1, floor(N/100))`; empty clusters are removed. Each reference requires at least 64 cells. The backend can be `torch` or GPU-enabled `faiss`.

Retrieval uses cosine similarity in the 1,536-dimensional image feature space, 50 neighboring clusters per measured-gene coverage group, and exponential weights with temperature 0.03. Each reference contributes at most its 50 closest clusters. Ties at the selection threshold share the remaining weight within this candidate set. Cluster counts do not weight expression retrieval.

## Fit fusion weights

```bash
python train/fit_beta.py --release /models/voice25m_direct --bank /runs/bank \
  --dataset /data/corpus --references /data/corpus/splits/gate_fit_references.txt \
  --reference-inputs /runs/reference_inputs --out /runs/beta.json
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/voice/checkpoints/alignment/stage1_final.pt \
  --stage2 /runs/voice/checkpoints/expression/stage2_last.pt \
  --space /runs/voice/direct/gene_space_all.npz \
  --xenium-space /runs/voice/direct/gene_space_xen.npz \
  --bank /runs/bank --beta /runs/beta.json --out /models/voice25m
```

Fusion weights are fitted on reference slides, excluding each reference and related donor/block samples from its retrieval bank. Calibration uses up to 20,000 deterministically sampled cells per reference by default (`--max-cells`, `--seed`). Each gene's weight is selected from 41 values between 0 and 1 by prediction correlation with measured log1p expression, then averaged over eligible references with equal reference weight.

Weights are grouped by tissue and panel width, using a 1,000-gene threshold and at least two references per group. Prediction first uses matching tissue and width, then another eligible width of the same tissue, then global weights. Tissue names are compared without case sensitivity. Target-slide expression is not used during prediction.
