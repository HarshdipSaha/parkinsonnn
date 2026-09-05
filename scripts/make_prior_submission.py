"""Build a prior-only submission.zip (main.py + datsc + tests_synth.py + calib.json with lam=1)."""
from __future__ import annotations

import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from datsc.config import DEFAULT_PRIOR

if __name__ == "__main__":
    out = ROOT / "submission_prior.zip"
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main.py", "main.py")
        z.write(ROOT / "tests_synth.py", "tests_synth.py")
        for p in sorted((ROOT / "datsc").glob("*.py")):
            z.write(p, f"datsc/{p.name}")
        z.writestr("artifacts/calib.json", json.dumps({"w": 0.0, "T": 1.0, "lam": 1.0, "pi": DEFAULT_PRIOR}))
    print("wrote", out)
