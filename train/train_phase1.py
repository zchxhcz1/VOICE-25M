#!/usr/bin/env python
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from train._entry import arguments, defaults


def main():
    a, extra = arguments(1)
    sys.argv = [sys.argv[0]] + defaults(extra, {'--optimizer':'adamw','--target':'scgpt',
                '--tag':'stage1','--save_every':500,'--epochs':3,'--max_cells':0,'--no_validation':None,
                '--retain_tail':None,'--d4_aug':None,'--uni2h':a.uni2h})
    from clip_muon_ab import main as run
    run()


if __name__ == '__main__':
    main()
