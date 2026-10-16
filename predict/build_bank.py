#!/usr/bin/env python
"""Build a clustered reference bank from cell features and expression values."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from voice.bank_builder import main
if __name__=='__main__':main()
