"""Recording IDs from file names.

The training files are named so that sorting them by name gives the order in which they
were recorded. This matters: build_dataset() sorts the paths, and that order decides the
grouped cross-validation folds and the random forest's bootstrap samples, so a different
naming would give slightly different numbers than the report.
"""
import re
from pathlib import Path


def rid(path) -> str:
    """'R07_human_2_.npz' -> 'R07'."""
    stem = Path(str(path)).stem
    m = re.match(r"\d{8}_\d{6}", stem)
    return m.group(0) if m else stem.split("_")[0]
