#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from train._entry import arguments, defaults


def main():
    a, extra = arguments(2)
    from voice.training_contract import read_ids
    ids = read_ids(Path(a.dataset)/'splits/stage2_global_training.txt')
    sys.argv = [sys.argv[0]] + defaults(extra, {'--tag':'stage2','--epochs':3,
                '--max_cells':0,'--no_validation':None,'--sampling':'full_coverage',
                '--target_cells':128,'--context_cells':64,'--workers':8,'--lora_ckpt':a.stage1,
                '--train_objects':','.join(ids),'--gene_emb':'scgpt','--free_head':None,
                '--free_init':'emb','--d4':0,'--grad_ckpt':None})
    from train_se2_live import main as run
    run()


if __name__ == '__main__':
    main()
