"""The training and spatial prediction settings used by VOICE-25M."""
from argparse import Namespace

EPOCHS = 3
TARGET_CELLS = 128
CONTEXT_CELLS = 64
RADIUS_UM = 137.0
SEED = 1


def stage1_settings(**runtime):
    values = dict(optimizer='adamw', target='scgpt', loss='infonce', epochs=EPOCHS,
                  max_cells=0, retain_tail=True, no_validation=True, d4_aug=True,
                  eff_batch=2048, lr_lora=1.5e-4, lr_tower=1e-3, wd=0.01,
                  warmup=150, nblocks=12, r=16, alpha=32, dropout=0.05, seed=SEED)
    if values.keys() & runtime.keys():
        raise TypeError('Training recipe settings are fixed; only runtime arguments may be supplied')
    values.update(runtime)
    return Namespace(**values)


def stage2_settings(**runtime):
    values = dict(epochs=EPOCHS, sampling='full_coverage', target_cells=TARGET_CELLS,
                  context_cells=CONTEXT_CELLS, max_cells=0, no_validation=True,
                  max_dist=RADIUS_UM, d_model=256, n_layers=6, span=0.0, loop_r=1,
                  nblocks=12, lora_r=16, lora_alpha=32, lora_dropout=0.05,
                  freeze_frac=0.2, lr_lora=3e-5, lr_se2=1e-4, grad_ckpt=True,
                  d4=0, lam_aux=1.0, lam_mse=1.0, seed=SEED,
                  gene_emb='scgpt', free_head=True, free_init='emb')
    if values.keys() & runtime.keys():
        raise TypeError('Training recipe settings are fixed; only runtime arguments may be supplied')
    values.update(runtime)
    return Namespace(**values)


def check_stage2_settings(args):
    for key, value in vars(stage2_settings()).items():
        if args.get(key) != value:
            raise ValueError(f'Checkpoint does not use the VOICE-25M setting: {key}={value}')
