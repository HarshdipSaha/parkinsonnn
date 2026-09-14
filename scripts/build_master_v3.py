"""Master v3 = Master v2's five pillars + the run-8 arms that pass the nested gate, all seeds, equal weights, T 0.9.

python scripts/build_master_v3.py [--src data/run8/artifacts_v3 ...] [--smoke <smoke_dir>] [--zip data/submission_master_v3.zip]
                                  [--gate 0.003] [--T 0.9] [--lam 0.02] [--max-mb 850] [--no-e2e] [--mode equal|shrunk] [--allow-trim]

Prints the gate table (nested estimate, weights + T refit per fold): Master v2 as shipped, pillars with every local seed, pillars + each
run-8 arm, pillars + all run-8 arms, greedy best subset; builds the zip for the best set; runs scripts/local_e2e_v3.py on the smoke set.
The smoke set is never fitted on (ADR 0005)."""
import argparse, glob, json, subprocess, sys, zipfile
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))
src = open(ROOT / "scripts" / "fit_stack_v3.py", encoding="utf-8").read().split('if __name__ == "__main__":')[0]
ns = {"__name__": "fs", "__file__": str(ROOT / "scripts" / "fit_stack_v3.py")}; exec(src, ns)
logloss, sigmoid, logit, fit_temperature, compositions = ns["logloss"], ns["sigmoid"], ns["logit"], ns["fit_temperature"], ns["compositions"]

PILLARS = {"a3d_hemi": "r3d_hemi48_f?_s?", "a3d_ssl": "r3d_ssl_f?_s?", "featc": "featcarm", "a2d_cnx_cor": "r2d_convnext_tiny_cor_f?_s?", "a2d_v2s": "r2d_efficientnet_v2_s_f?_s[01]"}
V2_SHIPPED = {**PILLARS, "a3d_hemi": "r3d_hemi48_f?_s[012]"}
NEW = {"a3d_hemisoft": "r3d_hemisoft_f?_s?", "a3d_hemimw": "r3d_hemimw_f?_s?", "a3d_crophistsoft": "r3d_crophistsoft_f?_s?", "a2d_swin": "r2d_swin_t_f?_s?", "a2d_swinsoft": "r2d_swin_t_soft_f?_s?"}
TEMPLATE_ARM = {"a3d_hemisoft": "a3d_hemi", "a3d_hemimw": "a3d_hemi", "a3d_crophistsoft": "a3d_crophist", "a2d_swin": "a2d_v2s", "a2d_swinsoft": "a2d_v2s"}   # manifest entry to copy
DROP_ORDER = [("a2d_v2s", "_s1."), ("a3d_hemimw", ""), ("a3d_crophistsoft", ""), ("a3d_hemi", "_s4."), ("a3d_hemi", "_s3.")]   # size trimming


def load_arm(srcs, pat):
    fs = sorted(set(f for d in srcs for f in glob.glob(str(Path(d) / f"oof_{pat}.csv")) if "_site" not in Path(f).name and "_fall" not in Path(f).name))
    if not fs:
        return None, []
    df = pd.concat([pd.read_csv(f, dtype={"uid": str}) for f in fs])
    return df.groupby("uid").agg(y=("y", "first"), p=("p", "mean")), fs


class Pool:
    def __init__(self, srcs, cache, T_floor, lam, pi=0.548, grid=8):
        self.srcs, self.T_floor, self.lam, self.pi, self.grid = srcs, T_floor, lam, pi, grid
        self.oofs, self.files = {}, {}
        for n, p in {**V2_SHIPPED, **{k + "@all": v for k, v in PILLARS.items()}, **NEW}.items():
            o, fs = load_arm(srcs, p)
            if o is not None and len(o) >= 1300:
                self.oofs[n], self.files[n] = o, fs
        self.uids = sorted(set.intersection(*[set(o.index) for o in self.oofs.values()]))
        self.y = next(iter(self.oofs.values())).loc[self.uids, "y"].values.astype(float)
        self.folds = pd.read_csv(Path(cache) / "folds.csv", dtype={"uid": str}).set_index("uid").loc[self.uids, "fold"].values

    def Z(self, names):
        return np.stack([logit(self.oofs[n].loc[self.uids, "p"].values) for n in names])

    def fit_w(self, Zs, ys):
        best = (9, None)
        for comb in compositions(self.grid, Zs.shape[0]):
            w = np.array(comb, float) / self.grid; ll = logloss(ys, sigmoid((w[:, None] * Zs).sum(0)))
            if ll < best[0]:
                best = (ll, w)
        return best[1]

    def nested(self, names, mode="equal"):
        Zs, y, folds = self.Z(names), self.y, self.folds; pn = np.zeros(len(y))
        for k in np.unique(folds):
            tr, va = folds != k, folds == k
            we = np.ones(len(names)) / len(names)
            w = we if mode == "equal" else (lambda wf: wf if mode == "fitted" else 0.5 * (wf + we))(self.fit_w(Zs[:, tr], y[tr]))
            T = max(fit_temperature((w[:, None] * Zs[:, tr]).sum(0), y[tr]), self.T_floor)
            pn[va] = (1 - self.lam) * sigmoid((w[:, None] * Zs[:, va]).sum(0) / T) + self.lam * self.pi
        return logloss(y, pn)


def gate_table(pool, gate, mode):
    have_new = [n for n in NEW if n in pool.oofs]; pil = [n + "@all" for n in PILLARS if n + "@all" in pool.oofs]
    rows = []
    def add(label, names):
        rows.append((label, names, pool.nested(names, "equal"), pool.nested(names, "shrunk"), pool.nested(names, "fitted")))
    add("Master v2 as shipped (hemi s0-2, v2s s0-1)", [n for n in V2_SHIPPED if n in pool.oofs])
    add("pillars, every local seed", pil)
    for n in have_new:
        add(f"pillars + {n}", pil + [n])
    if len(have_new) > 1:
        add("pillars + all run-8 arms", pil + have_new)
    # greedy forward selection from the pillars by equal-weight nested
    cur, best = list(pil), pool.nested(pil, "equal"); left = list(have_new)
    while left:
        cand = [(pool.nested(cur + [n], "equal"), n) for n in left]; ll, n = min(cand)
        if ll >= best - 1e-4:
            break
        cur.append(n); left.remove(n); best = ll
    add("greedy best subset", cur)
    print("\nGATE TABLE (nested estimate; equal / shrunk / fitted weights; T floor %.2f; lam %.2f)" % (pool.T_floor, pool.lam))
    for label, names, e, s, f in rows:
        print(f"  {label:<48} equal {e:.4f}  shrunk {s:.4f}  fitted {f:.4f}   [{', '.join(n.replace('@all', '') for n in names)}]")
    v2 = rows[0][2]; col = {"equal": 2, "shrunk": 3, "fitted": 4}[mode]
    best_row = min(rows[1:], key=lambda r: r[col]); gain = v2 - best_row[col]
    print(f"\nbest set by {mode} nested: {best_row[0]} = {best_row[col]:.4f}; Master v2 shipped = {v2:.4f}; gain {gain:+.4f} -> GATE: {'PASS' if gain >= gate else 'FAIL (keep Master v2 for slot 1)'}")
    return best_row, gain >= gate


def build_zip(pool, names, mode, T, lam, out, max_mb, note, allow_trim=False):
    master = json.loads((ROOT / "data" / "master_pool" / "artifacts_v3" / "manifest.json").read_text())
    by_name = {a["name"]: a for a in master["arms"]}
    if "a3d_crophist" not in by_name and Path("data/submission_run5.zip").exists():   # template for crophistsoft lives in the run-5 manifest
        z = zipfile.ZipFile("data/submission_run5.zip"); m5 = json.loads(z.read([n for n in z.namelist() if n.endswith("manifest.json")][0]))
        by_name.update({a["name"]: a for a in m5["arms"] if a["name"] == "a3d_crophist"})
    arms = []
    for n in names:
        base = n.replace("@all", ""); tpl = by_name[TEMPLATE_ARM.get(base, base)]
        arm = json.loads(json.dumps(tpl)); arm["name"] = base
        if "swin" in base:
            arm["model_kw"]["backbone"] = "swin_t"
        arm["checkpoints"] = sorted(set("artifacts/" + Path(f).name.replace("oof_", "").replace(".csv", ".pt") for f in pool.files[n])) if base != "featc" else tpl["checkpoints"]
        arms.append(arm)
    weights = {a["name"]: 1.0 / len(arms) for a in arms}
    if mode != "equal":
        wf = pool.fit_w(pool.Z(names), pool.y); we = np.ones(len(names)) / len(names); w = wf if mode == "fitted" else 0.5 * (wf + we)
        weights = {a["name"]: float(x) for a, x in zip(arms, w)}
    man = {**master, "arms": arms, "weights": weights, "T": float(T), "lam": float(lam), "clip": None, "hedge": None, "typ": None, "note": note}
    sources = [Path(d) for d in pool.srcs] + [Path(p) for p in ("data/run6c/artifacts_v3", "data/kaggle_output_v5/artifacts_v3", "data/run6b/artifacts_v3", "data/ckpt_ds_v4/artifacts_v3")]
    zips = [zipfile.ZipFile(p) for p in ("data/submission_run7.zip", "data/submission_run5.zip", "data/submission_master_v2.zip") if Path(p).exists()]
    def find(name):
        for d in sources:
            if (d / name).exists():
                return (d / name).read_bytes()
        for z in zips:
            m = [i for i in z.infolist() if i.filename.endswith("/" + name) or i.filename == name]
            if m:
                return z.read(m[0])
        raise FileNotFoundError(name)
    def write(man):
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zo:
            zo.write(ROOT / "main_v3.py", "main.py"); zo.write(ROOT / "tests_synth.py", "tests_synth.py")
            for p in sorted((ROOT / "datsc").glob("*.py")):
                zo.write(p, f"datsc/{p.name}")
            zo.writestr("artifacts/template.npz", find("template.npz"))
            for arm in man["arms"]:
                for ck in arm["checkpoints"]:
                    zo.writestr(ck, find(Path(ck).name)); n += 1
            zo.writestr("artifacts/manifest.json", json.dumps(man, indent=1))
        return n, Path(out).stat().st_size / 1e6
    n, mb = write(man)
    trimmed = []
    for arm_name, key in DROP_ORDER:
        if mb <= max_mb:
            break
        for arm in man["arms"]:
            if arm["name"] == arm_name:
                before = len(arm["checkpoints"]); arm["checkpoints"] = [c for c in arm["checkpoints"] if key and key not in c]
                if key == "":
                    man["arms"] = [a for a in man["arms"] if a["name"] != arm_name]; man["weights"].pop(arm_name, None)
                    s = sum(man["weights"].values()); man["weights"] = {k: v / s for k, v in man["weights"].items()}
                print(f"zip {mb:.0f} MB > {max_mb}: dropped {arm_name} {key or '(whole arm)'} ({before} -> {len(arm['checkpoints'])} ckpts)")
                trimmed.append(f"{arm_name} {key or '(whole arm)'}")
        n, mb = write(man)
    if mb > max_mb:
        raise SystemExit(f"FATAL: {out} is {mb:.0f} MB > --max-mb {max_mb} and DROP_ORDER is exhausted. "
                         f"Raise --max-mb or extend DROP_ORDER; do NOT submit this zip.")
    if trimmed and not allow_trim:
        raise SystemExit(f"FATAL: the zip fitted only after trimming {trimmed}. The gate table above scored the "
                         f"UNTRIMMED set, and dropping an arm also renormalises the weights, so what would ship is "
                         f"not what was measured. Raise --max-mb and rebuild, or pass --allow-trim to accept it "
                         f"and re-read the gate table for the trimmed set.")
    print(f"wrote {out}: {n} checkpoint files across {len(man['arms'])} arms, {mb:.0f} MB; weights {json.dumps({k: round(v, 3) for k, v in man['weights'].items()})}; T {man['T']} lam {man['lam']}")
    for arm in man["arms"]:
        print(f"  {arm['name']:<16} {len(arm['checkpoints'])} ckpts")
    return man


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--src", action="append", default=None); ap.add_argument("--cache", default="cache_v4")
    ap.add_argument("--smoke", default=None); ap.add_argument("--zip", default="data/submission_master_v3.zip"); ap.add_argument("--gate", type=float, default=0.003)
    ap.add_argument("--T", type=float, default=0.9); ap.add_argument("--lam", type=float, default=0.02); ap.add_argument("--max-mb", type=float, default=850)
    ap.add_argument("--mode", default="equal", choices=["equal", "shrunk", "fitted"]); ap.add_argument("--no-e2e", action="store_true"); ap.add_argument("--table-only", action="store_true")
    ap.add_argument("--allow-trim", action="store_true", help="accept a zip that only fitted after DROP_ORDER trimming; by default that is fatal, because the gate table scored the untrimmed set")
    a = ap.parse_args()
    srcs = a.src or ["data/run8/artifacts_v3", "data/master_pool/artifacts_v3"]
    pool = Pool(srcs, a.cache, a.T, a.lam)
    print(f"pool: {len(pool.uids)} scans; arms with complete OOF: {sorted(pool.oofs)}")
    best_row, passed = gate_table(pool, a.gate, a.mode)
    if a.table_only:
        sys.exit(0)
    man = build_zip(pool, best_row[1], a.mode, a.T, a.lam, a.zip, a.max_mb, f"master v3: {best_row[0]}; nested {a.mode} {best_row[2 if a.mode == 'equal' else 3 if a.mode == 'shrunk' else 4]:.4f}; gate {'PASS' if passed else 'FAIL'}", a.allow_trim)
    if a.smoke and not a.no_e2e:
        print("container-style check on the smoke set ...", flush=True)
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "local_e2e_v3.py"), a.zip, a.smoke], capture_output=True, text=True)
        print(r.stdout[-1500:]); print(r.stderr[-500:]); sys.exit(r.returncode)
