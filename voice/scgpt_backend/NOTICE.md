# scGPT preparation backend

Source: [bowang-lab/scGPT](https://github.com/bowang-lab/scGPT), the scGPT v0.2.4 implementation retained with the original VOICE-25M training code. License: [MIT](LICENSE).

`model.py`, `dsbn.py`, `grad_reverse.py` and `data_collator.py` preserve that implementation. `preprocess.py` retains only `_digitize` and `binning` and uses a local logger. The package import is scoped to these components so RNA target preparation does not depend on upstream's unrelated task, tokenizer, scanpy or torchtext imports. Weights are not included here. This module is used only for preparing training targets.
