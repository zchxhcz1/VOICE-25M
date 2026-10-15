#!/usr/bin/env python
"""Build the comm100 bank from a portable full-cell feature/value manifest."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from voice.bank_builder import main
if __name__=='__main__':main()
