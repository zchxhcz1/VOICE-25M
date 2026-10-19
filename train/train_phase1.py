#!/usr/bin/env python
"""Train VOICE-25M stage 1 with the fixed three-epoch recipe."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from train._entry import arguments


def main():
    paths = arguments(1)
    from voice.training.stage1 import main as run
    run(paths)


if __name__ == '__main__':
    main()
