# Version provenance

Repository layout reference: [Luosanj/VOICE_HE_ST_Pred](https://github.com/Luosanj/VOICE_HE_ST_Pred), commit `e0e8d408f90b03b7ad3327575a3bc961f4c5dfbd`.

The retained implementation comes from the VOICE-25M scGPT run and its portable inference sources. Stage-1 is `v25mfull_scgpt_adamw_d4_final.pt`; Stage-2 is `live_scgpt_last.pt`. The comm100 bank and refitted V3 gate come from the `community100_k50_82ref` and `comm100_K50_refit_20261011` release line.

| Component | Configuration |
| --- | --- |
| Stage-1 alignment | scGPT cell targets, AdamW, InfoNCE, random D4, 3 epochs |
| Stage-1 corpus | 82 objects, 25,466,122 cells |
| Stage-2 corpus | 79 objects, 25,074,898 cells |
| Stage-2 | Live LoRA, six spatial layers, 256 features, 137 µm maximum distance, free scGPT-initialized head, D4 off |
| Output head | 18,580 ordered genes |
| Reference bank | 82 objects, 254,622 k-means clusters |
| Clustering | Per-object PCA64, 10 k-means iterations, approximately 100 cells per cluster |
| Retrieval | Original 1536-dimensional fp16 cosine, K50, tau 0.03, measured-gene coverage groups |
| Calibration | V3, tissue case folding, minimum 2 references, 1,000-gene panel cutoff, refitted for this bank |
| Branch scales | A: log1p(mu); R: per-object standardized log1p counts; fusion: beta*A+(1-beta)*R |

The calibrated bank fingerprint is `a4cde9b15cd5c8163795040936d3ea8bd8ad9b11ca6c19afb2f776147038da0c`. Another locally built bank has a different fingerprint; it is not interchangeable with this gate. Complete artifact SHA identities are in [weights/README.md](../weights/README.md).

Changes for public use: explicit portable release paths; inference separated from training imports; automatic segmentation integrated into a single command; comm100/K50 retrieval replaces the legacy K200 inference call; model/bank/gene/gate identities checked; expression matrices written in batches; source-resolution validation; source-pixel crop/polygon alignment; and no downstream/clinical analysis code.

The supplied beta values and numerical model/retrieval kernels are preserved. Generic training and calibration commands can operate on a new corpus and produce new weights and identities. They do not claim to reproduce the original run without its original data, exact row choices, hardware and random state.
