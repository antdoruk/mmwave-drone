# lets `python3 scripts/xxx.py` find the mmwd package without installing it
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
