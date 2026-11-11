"""Check that the frozen classification files and model are unchanged.

Used by scripts/eval_test.py and report/compute_test.py before they evaluate anything.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# files whose content decides a label; session.py only reads and writes recordings
MUST_MATCH = ("mmwd/classify.py", "mmwd/features.py", "mmwd/config.py", "configs/md_1tx.cfg")


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def verify(frozen_dir: Path | None = None, verbose: bool = True) -> Path:
    """Compare the frozen files with the manifest. Returns the model path; raises SystemExit
    if anything differs."""
    fz = Path(frozen_dir) if frozen_dir else ROOT / "models" / "frozen"
    manifest = next(p for p in sorted(fz.glob("FROZEN*.txt")) if "original" not in p.name)
    model = next(fz.glob("*.joblib"))
    want = {}
    for line in manifest.read_text().splitlines():
        m = re.match(r"^([0-9a-f]{32})\s+(\S+)$", line.strip())
        if m:
            want[m.group(2)] = m.group(1)
    checks = [(rel, ROOT / rel) for rel in MUST_MATCH]
    checks.append((next(k for k in want if k.endswith(".joblib")), model))
    ok = True
    for rel, path in checks:
        same = _md5(path) == want.get(rel)
        ok &= same
        if verbose:
            print(f"{'same   ' if same else 'CHANGED'} {rel}")
    if not ok:
        raise SystemExit("frozen code or model differs from the manifest: stopping")
    return model
