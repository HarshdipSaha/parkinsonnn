"""Master v2: the 5 pillars with EQUAL weights (nested 0.2354 vs 0.2370 fitted), temperature from the leave-site-out pool
(T = 0.9, between the pool optimum 0.83 and the old floor 1.0), all hemisphere/SSL seeds and two V2-S seeds (the seed-0-only
pruning of master_final cost 0.011 on the V2-S arm). python scripts/build_master_v2.py [--T 0.9] [--lam 0.02] [--out data/submission_master_v2.zip]"""
import argparse, glob, json, zipfile
from pathlib import Path

ap = argparse.ArgumentParser(); ap.add_argument("--T", type=float, default=0.9); ap.add_argument("--lam", type=float, default=0.02)
ap.add_argument("--out", default="data/submission_master_v2.zip"); ap.add_argument("--v2s-seeds", default="01")
a = ap.parse_args()
ROOT = Path("."); master_art = Path("data/master_pool/artifacts_v3")
man = json.loads((master_art / "manifest.json").read_text())
keep = {"a3d_hemi": None, "a3d_ssl": None, "featc": None, "a2d_cnx_cor": None, "a2d_v2s": [f"_s{s}." for s in a.v2s_seeds]}
arms = []
for arm in man["arms"]:
    if arm["name"] not in keep:
        continue
    if keep[arm["name"]]:
        arm["checkpoints"] = [c for c in arm["checkpoints"] if any(k in c for k in keep[arm["name"]]) and "_fall_" not in c]
    else:
        arm["checkpoints"] = [c for c in arm["checkpoints"] if "_site" not in c]
    arms.append(arm)
man["arms"] = arms; man["weights"] = {x["name"]: 1.0 / len(arms) for x in arms}
man["T"], man["lam"], man["clip"], man["hedge"], man["typ"] = float(a.T), float(a.lam), None, None, None
man["note"] = "master v2: equal weights over 5 pillars; T from the leave-site-out pool; all hemi/SSL seeds; V2-S seeds " + a.v2s_seeds
sources = [Path("data/run6c/artifacts_v3"), Path("data/kaggle_output_v5/artifacts_v3"), Path("data/run6b/artifacts_v3"), Path("data/ckpt_ds_v4/artifacts_v3")]
zips = [zipfile.ZipFile(p) for p in ("data/submission_run7.zip", "data/submission_run5.zip") if Path(p).exists()]
def find(name):
    for d in sources:
        if (d / name).exists():
            return (d / name).read_bytes()
    for z in zips:
        m = [i for i in z.infolist() if i.filename.endswith("/" + name) or i.filename == name]
        if m:
            return z.read(m[0])
    raise FileNotFoundError(name)
out = Path(a.out); n = 0
with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
    zo.write(ROOT / "main_v3.py", "main.py"); zo.write(ROOT / "tests_synth.py", "tests_synth.py")
    for p in sorted((ROOT / "datsc").glob("*.py")):
        zo.write(p, f"datsc/{p.name}")
    zo.writestr("artifacts/template.npz", find("template.npz"))
    for arm in arms:
        for ck in arm["checkpoints"]:
            zo.writestr(ck, find(Path(ck).name)); n += 1
    zo.writestr("artifacts/manifest.json", json.dumps(man, indent=1))
print(f"wrote {out}: {n} checkpoint files across {len(arms)} arms; {out.stat().st_size/1e6:.0f} MB; weights {man['weights']}; T {man['T']} lam {man['lam']}")
for arm in arms:
    print(f"  {arm['name']:<12} {len(arm['checkpoints'])} ckpts")
