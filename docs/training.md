# Training, bank construction, and V3 calibration

The two retained training engines are the ones used for the 25M scGPT model, with explicit user paths and shared model definitions. They support CUDA, torchrun, checkpoint resume, complete-cell coverage, and fixed epoch budgets. scGPT model/collator/binning code needed for preparation is included under its own license; it is not imported during H&E inference.

```bash
pip install -r requirements-train.txt
```

## Prepared corpus

Organize paired images, boundaries, raw RNA counts and gene symbols as specified in [formats.md](formats.md):

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
  metadata/shared_origin_groups.csv       # optional donor/block relations
```

Each split text file contains one sample name per line. Sample names must be globally unique. The in-slide set is a subset of Stage-1 training; Stage-2 training equals Stage-1 training minus in-slide samples. The cross-slide set is disjoint from Stage-1 training. `retrieval_references.txt` and `gate_fit_references.txt` equal the Stage-2 training set. Empty test splits are allowed for a new training-only corpus. `MANIFEST.csv` records the corpus and its checksum binds training checkpoints to that version.

The original 25M experiment used 82 Stage-1 objects (25,466,122 cells), 79 Stage-2 objects (25,074,898 cells), an 18,580-gene head, and three epochs in each stage. These counts are descriptive; user datasets are not required to contain those numbers of samples or cells.

For donor/block exclusions, `shared_origin_groups.csv` uses columns `field,samples`, with `field=donor_or_block_id` and semicolon-separated sample names. Related groups are expanded transitively when fitting beta.

## 1. scGPT cell and gene targets

Obtain the scGPT whole-human model files (`best_model.pt`, `vocab.json`, `args.json`) following the [upstream scGPT project](https://github.com/bowang-lab/scGPT). Prepare row-aligned cell embeddings only for training objects:

```bash
python train/embed_scgpt.py --dataset /data/corpus \
  --scgpt-dir /models/scgpt_human --out /runs/scgpt_cells
```

The cell embedding extractor follows the original whole-human protocol: nonzero RNA genes, CLS token, 51-bin quantization, up to 1,200 tokens, fp16 encoding and normalized 512-dimensional CLS features. Output `<sample>.npy` uses expression-matrix row order; zero-feature cells are excluded from Stage-1 as in the original run. `--only sample_a,sample_b` and sharding options are available.

Choose an ordered RNA gene list, one symbol per line (the released list is `weights/genes.tsv`), and construct the Stage-2 initialization table:

```bash
python train/build_gene_table.py --genes /data/head_genes.txt \
  --scgpt-dir /models/scgpt_human --out /runs/gene_table.npz
python train/prepare.py --dataset /data/corpus --work-dir /runs/voice \
  --gene-table /runs/gene_table.npz
```

Gene-table rows use scGPT's embedding followed by its encoder layer normalization, with exact then case-insensitive symbol matching. Out-of-vocabulary rows start at zero and can be learned by the free gene head. To reproduce the published head, use the original release's exact ordered gene symbols; a newly chosen gene head requires a newly built bank and refitted beta.

## 2. Stage-1 contrastive alignment

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase1.py \
  --dataset /data/corpus --work-dir /runs/voice \
  --uni2h /models/uni2h --cell-embeddings /runs/scgpt_cells
```

Defaults match the model recipe: AdamW, fixed-temperature symmetric InfoNCE (0.07), three epochs, all usable cells, retained tail batches, random D4 augmentation, LoRA on the final 12 blocks (rank 16, alpha 32, dropout 0.05), effective per-rank batch 2,048 and encoder chunk 256. Learning rates are 1.5e-4 for LoRA and 1e-3 for the projection towers. Set `--chunk`/`--eff_batch` for your hardware and record changes to the recipe. Single-GPU runs can use `python` in place of `torchrun`.

Output: `/runs/voice/clip_ab/ckpts/stage1_final.pt`. `--resume /runs/.../stage1_step500.pt` restores optimizer and coverage state; resume uses the same world size and input identities. No held-out RNA labels are used for validation/model selection by these defaults.

## 3. Stage-2 expression training

```bash
torchrun --standalone --nproc_per_node=3 train/train_phase2.py \
  --dataset /data/corpus --work-dir /runs/voice --uni2h /models/uni2h \
  --stage1 /runs/voice/clip_ab/ckpts/stage1_final.pt
```

Defaults match the actual Stage-2 checkpoint: 256-dimensional spatial features, six SE(2) layers, maximum distance 137 µm, 128 supervised target cells plus up to 64 context cells per block, three epochs with complete target-cell coverage, live LoRA updates and gradient checkpointing, **D4 disabled**, a learnable scGPT-initialized gene head, and negative-binomial plus log1p-MSE loss. The first ceil(0.2×12) adapted encoder blocks are frozen. Learning rates: LoRA 3e-5, decoder/head 1e-4.

Output: `/runs/voice/direct/ckpts_se2live/stage2_last.pt`. Resume with `--resume`. Context cells contribute image context but not expression loss. Gene supervision uses each sample's measured RNA panel.

## 4. Encode references and build the bank

First export a direct-only release. It can predict A and Stage-1 embeddings without an existing bank or beta:

```bash
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/voice/clip_ab/ckpts/stage1_final.pt \
  --stage2 /runs/voice/direct/ckpts_se2live/stage2_last.pt \
  --space /runs/voice/direct/gene_space_all.npz \
  --xenium-space /runs/voice/direct/gene_space_xen.npz \
  --out /models/voice25m_direct
python train/prepare_bank.py --release /models/voice25m_direct \
  --dataset /data/corpus --references /data/corpus/splits/retrieval_references.txt \
  --out /runs/reference_inputs
python predict/build_bank.py --manifest /runs/reference_inputs/manifest.json \
  --output /runs/bank --device cuda --backend torch
```

The reference preparation uses all cells, aligns measured expression via expression-row IDs and writes the documented per-object z-score values. The original released bank has 82 objects; a new conservative split can use the Stage-2-only reference list above. Changing reference objects changes the bank fingerprint and requires refitting beta. The number of objects is inferred from the manifest. PCA64 requires at least 64 cells per object. [formats.md](formats.md) documents k-means, original-feature keys, gene coverage and tie rules.

## 5. Fit V3 on references and export the fused release

```bash
python train/fit_gate.py --release /models/voice25m_direct --bank /runs/bank \
  --dataset /data/corpus --references /data/corpus/splits/gate_fit_references.txt \
  --reference-inputs /runs/reference_inputs --out /runs/gate_v3.json
python train/export_release.py --uni2h /models/uni2h \
  --stage1 /runs/voice/clip_ab/ckpts/stage1_final.pt \
  --stage2 /runs/voice/direct/ckpts_se2live/stage2_last.pt \
  --space /runs/voice/direct/gene_space_all.npz \
  --xenium-space /runs/voice/direct/gene_space_xen.npz \
  --bank /runs/bank --gate /runs/gate_v3.json --out /models/voice25m
```

By default, calibration uses up to 20,000 deterministically sampled cells per reference (`--max-cells`, `--seed`); all reference cells remain in the bank. Beta is fit over 41 grid points from 0 to 1 on reference log1p counts, then averaged per gene over eligible references with equal reference weight. Self and donor/block-related objects are excluded from each reference's retrieval. V3 uses case-folded tissue keys, narrow/wide cutoff 1,000 and minimum two references per stratum; it falls back from same tissue/width to same tissue/other width to global. Target labels are never required.

This generic workflow refits weights for your corpus. It does not promise identical numeric beta to the supplied released gate, which was fitted using the original stored reference rows and protocol. Use the matching shipped model/bank/gate to reproduce that model's inference.
