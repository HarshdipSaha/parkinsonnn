"""Build the run-3 submission zip: main_v3.py -> main.py, tests_synth.py, datsc/*.py, artifacts/manifest.json + every
checkpoint the manifest references. python scripts/pack_v3.py [--artifacts artifacts_v3] [--out submission_v3.zip]"""
from __future__ import annotations

import argparse
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", default="artifacts_v3")
    ap.add_argument("--out", default="submission_v3.zip")
    a = ap.parse_args()
    art = Path(a.artifacts)
    man = json.loads((art / "manifest.json").read_text())
    cks = [ck for arm in man["arms"] for ck in arm["checkpoints"]]
    with zipfile.ZipFile(ROOT / a.out, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(ROOT / "main_v3.py", "main.py")
        z.write(ROOT / "tests_synth.py", "tests_synth.py")
        for p in sorted((ROOT / "datsc").glob("*.py")):
            z.write(p, f"datsc/{p.name}")
        z.write(art / "manifest.json", "artifacts/manifest.json")
        for extra in ("template.npz", "lr_v3.json"):
            if (art / extra).exists():
                z.write(art / extra, f"artifacts/{extra}")
        for ck in cks:
            src = art / Path(ck).name
            assert src.exists(), src
            z.write(src, ck)
    size = (ROOT / a.out).stat().st_size / 1e6
    print(f"wrote {a.out}: {len(cks)} checkpoints across {len(man['arms'])} arms, {size:.0f} MB, "
          f"OOF logloss {man.get('oof_logloss', float('nan')):.4f}")
