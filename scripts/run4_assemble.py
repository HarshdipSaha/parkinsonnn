"""One-shot assembly after a Kaggle run: nested stack over every arm with OOF files, drop zero-weight arms, pack the zip
with every referenced checkpoint, run the container-style local check on the smoke set.

python scripts/run4_assemble.py --src data/kaggle_output_v4 --extra-dir artifacts/featarm_v4 --smoke <smoke_dir> --zip data/submission_v4.zip
  --src        the downloaded version output (scripts/watch_kaggle_output.py); needs <src>/artifacts_v3 (.pt) and/or <src>/out (json, oof)
  --extra-dir  extra artefact dirs copied in first (e.g. the locally fitted feature arm)
"""
from __future__ import annotations

import argparse
import glob
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def sh(cmd: str) -> str:
    print("$", cmd, flush=True)
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd=ROOT)
    print(r.stdout[-6000:]); print(r.stderr[-1500:])
    assert r.returncode == 0, f"failed: {cmd}"
    return r.stdout


def stack_cmd(art: Path, arms, site, extras, grid: int) -> str:
    cmd = f'python scripts/fit_stack_v3.py --out "{art}" --lam 0.02 --t-floor 1.0 --nested --grid {grid} --cache cache_v4'
    cmd += "".join(f' --arm {n} "{p}"' for n, p in arms)
    cmd += "".join(f' --site-arm {n} "{q}"' for n, q in site)
    cmd += "".join(f' --extra {n} "{g}"' for n, g in extras)
    return cmd


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--extra-dir", action="append", default=[])
    ap.add_argument("--smoke", default=None)
    ap.add_argument("--zip", default="data/submission_v4.zip")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--min-weight", type=float, default=0.05, help="arms below this weight are dropped and the stack re-fitted")
    a = ap.parse_args()
    src = Path(a.src); art = src / "assembled"; art.mkdir(exist_ok=True)
    # 1) gather artefacts: .pt/.npz/.json/.csv from artifacts_v3 and out (+ extra dirs) into one folder
    for d in [Path(x) for x in a.extra_dir] + [src / "artifacts_v3", src / "out"]:
        if d.exists():
            for f in d.iterdir():
                if f.is_file() and f.suffix in (".pt", ".npz", ".json", ".csv") and not (art / f.name).exists():
                    shutil.copy(f, art / f.name)
    tpl = ROOT / "artifacts" / "template.npz"
    if tpl.exists() and not (art / "template.npz").exists():
        shutil.copy(tpl, art / "template.npz")
    print("artefacts:", len(list(art.glob("*.pt"))), "pt |", len(list(art.glob("oof_*.csv"))), "oof |", sorted(p.name for p in art.glob("*.npz")))
    # 2) arms from the generator's ARMS/EXTRAS, filtered to what exists
    ns = {"__file__": str(ROOT / "kaggle" / "gen_run3_inject.py"), "__name__": "gen"}
    exec(compile((ROOT / "kaggle" / "gen_run3_inject.py").read_text(encoding="utf-8"), "gen", "exec"), ns)
    arms = [(n, p) for n, p, _ in ns["ARMS"] if glob.glob(str(art / f"oof_{p}.csv"))]
    site = [(n, q) for n, _, q in ns["ARMS"] if q and glob.glob(str(art / f"oof_{q}.csv"))]
    extras = [(n, g) for n, g in ns["EXTRAS"] if glob.glob(str(art / f"{g}.pt")) and n in dict(arms)]
    print("arms:", [n for n, _ in arms], "| site arms:", [n for n, _ in site], "| extras:", extras)
    out = sh(stack_cmd(art, arms, site, extras, a.grid))
    man = json.loads((art / "manifest.json").read_text())
    w = man["weights"]; keep = [n for n, _ in arms if w.get(n, 0) >= a.min_weight]
    print("weights:", w, "| nested:", [l for l in out.splitlines() if "nested combiner estimate" in l])
    if len(keep) < len(arms):
        print(f"re-fitting without zero-weight arms: dropped {[n for n, _ in arms if n not in keep]}")
        arms2 = [(n, p) for n, p in arms if n in keep]; site2 = [(n, q) for n, q in site if n in keep]; extras2 = [(n, g) for n, g in extras if n in keep]
        out2 = sh(stack_cmd(art, arms2, site2, extras2, a.grid))
        man = json.loads((art / "manifest.json").read_text())
    print("final weights:", man["weights"], "T", round(man["T"], 3), "OOF", round(man["oof_logloss"], 4), "clip", man.get("clip"), "hedge", man.get("hedge"))
    # 3) pack + verify
    sh(f'python scripts/pack_v3.py --artifacts "{art}" --out "{a.zip}"')
    if a.smoke:
        sh(f'python scripts/local_e2e_v3.py "{a.zip}" "{a.smoke}"')
    print("DONE:", a.zip)
