"""Common explicit path arguments for the retained training engines."""
from pathlib import Path
import argparse
import os
import sys


def arguments(stage):
    p = argparse.ArgumentParser(description=f'Train VOICE-25M stage {stage}', add_help=False)
    p.add_argument('--dataset', required=True, help='Prepared corpus root (see docs/training.md)')
    p.add_argument('--work-dir', required=True, help='Output root; contains direct/ and clip_ab/ckpts/')
    p.add_argument('--uni2h', required=True, help='Local UNI2-h weights directory')
    p.add_argument('--cell-embeddings', required=(stage == 1), help='Row-aligned <sample>.npy scGPT embeddings')
    p.add_argument('--stage1', required=(stage == 2), help='Stage-1 checkpoint to initialize Stage-2')
    # Help is handled before loading weights or requiring user paths.
    if '--help' in sys.argv or '-h' in sys.argv:
        p.print_help()
        print('\nAdditional engine options: --epochs --tag --workers --resume; see docs/training.md.')
        raise SystemExit(0)
    a, extra = p.parse_known_args()
    root = Path(a.work_dir).resolve()
    root.mkdir(parents=True,exist_ok=True)
    values = {'VOICE_ROOT':root, 'VOICE_DATASET':Path(a.dataset).resolve(),
              'VOICE_CKDIR':root/'checkpoints/alignment', 'VOICE_UNI2H':Path(a.uni2h).resolve(),
              'VOICE_FEAT':root/'features', 'VOICE_TRAIN_CACHE':root/'cache'}
    if a.cell_embeddings:
        values['VOICE_SCGPT_EMB'] = Path(a.cell_embeddings).resolve()
    for key,value in values.items():
        os.environ[key] = str(value)
    sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'voice/training'))
    return a, extra


def defaults(extra, values):
    result = []
    for key,value in values.items():
        if key not in extra:
            result += [key] + ([] if value is None else [str(value)])
    return result + extra
