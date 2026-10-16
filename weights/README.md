# Model artifacts

The released VOICE checkpoints, V3 beta, gene space and matching comm100/K50 bank are available at **[zchx/VOICE-25M](https://huggingface.co/zchx/VOICE-25M)**. Download the complete snapshot, then obtain UNI2-h separately through the official access process. The source repository contains code; its linked Hugging Face repository contains the model artifacts.

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

A complete release directory contains:

```text
voice25m/
  release.json
  uni2h/pytorch_model.bin
  stage1.pt
  stage2.pt
  gene_space.npz
  xenium_genes.tsv             # optional; enables --genes xenium
  gate_v3.json
  gate_v3.meta.json
  bank/
    model.json
    bank_complete.json
    summaries/object_00_comm100/...
    ...
```

`release.json` uses paths relative to itself. It must bind Stage-1, Stage-2, the ordered gene space, the bank and the beta sidecar. See [release.example.json](../configs/release.example.json). `genes.tsv` and `xenium_genes.tsv` record the released ordered RNA gene lists, one symbol per line without a header. The supplied `gate_v3.json` is the actual refitted comm100/K50 V3 gate; its numerical beta values are unchanged. Original machine paths have been removed from its provenance, and its sidecar records the updated file checksum.

For the existing model, the gate expects:

| Artifact | Identity |
| --- | --- |
| Stage-1 | `5d00bb93a1b34c5028e6e42711d75965aeaf8cf5a34cdd710e4a8e748643c48f` |
| Stage-2 | `ee981add0036126728e78619d4718c426e8d3f2e65aae95eb4ecaac56f76a0fc` |
| Gene-space file | `87ac6eb3e95da658e2993ea54b8043f4bb60604fa48100aeea8697c4b84bd938` |
| Bank fingerprint | `a4cde9b15cd5c8163795040936d3ea8bd8ad9b11ca6c19afb2f776147038da0c` |

Use an artifact host for model/bank binaries. Obtain UNI2-h separately through its official access process and follow the relevant weight/data distribution terms. scGPT is needed for training preparation; its model is not loaded during H&E inference.

Validate an assembled release before inference:

```bash
python predict/validate_release.py --release /models/voice25m
```

Export an independent release from explicit paths:

```bash
python train/export_release.py \
  --uni2h /models/uni2h \
  --stage1 /runs/clip_ab/ckpts/stage1_final.pt \
  --stage2 /runs/direct/ckpts_se2live/stage2_last.pt \
  --space /runs/direct/gene_space_all.npz \
  --xenium-space /runs/direct/gene_space_xen.npz \
  --bank /runs/bank --gate /runs/gate_v3.json \
  --out /models/voice25m
```

For a direct-only release used to encode references before fitting a new gate, omit both `--bank` and `--gate`. Export checks identities and copies files into a relocatable directory; the destination must be empty.
