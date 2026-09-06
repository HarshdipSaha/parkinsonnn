"""Poll the Kaggle commit and pull its output to this machine the moment it finishes.

Idempotent; meant to be fired every 5 min by Windows Task Scheduler (see kaggle/register_watcher.ps1), so it keeps
working when no Claude session is open. Writes data/kaggle_output/watch.log and, on success, DONE.json.
    python scripts/watch_kaggle_output.py [--kernel harshdipsaha/dat-parkinsons-train] [--out data/kaggle_output] [--once]
"""
from __future__ import annotations

import argparse, glob, hashlib, json, os, shutil, subprocess, sys, time, zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TASK = "DaT-KaggleOutputWatch"


def kaggle_exe() -> str:
    for c in (shutil.which("kaggle"), str(Path(sys.executable).parent / "Scripts" / "kaggle.exe"),
              str(Path(sys.executable).parent / "kaggle.exe")):
        if c and Path(c).exists():
            return c
    return "kaggle"


def log(out: Path, msg: str):
    line = f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC  {msg}"
    with open(out / "watch.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def status(kernel: str) -> str:
    r = run([kaggle_exe(), "kernels", "status", kernel])
    txt = (r.stdout + r.stderr).strip()
    for s in ("COMPLETE", "ERROR", "CANCEL", "RUNNING", "QUEUED"):
        if s in txt.upper():
            return s
    return "UNKNOWN: " + txt[-200:]


def verify(out: Path):
    zips = [p for p in out.rglob("submission_v3.zip") if p.is_file()]
    parts = sorted(p for p in out.rglob("submission_v3.zip.part*") if p.is_file())
    info = {"zip": None, "parts": len(parts), "md5": None, "testzip": None, "n_files": None, "size": None}
    if not zips and parts:
        tgt = out / "submission_v3.zip"
        with open(tgt, "wb") as w:
            for p in parts:
                w.write(p.read_bytes())
        zips = [tgt]
        info["joined"] = True
    if zips:
        z = max(zips, key=lambda p: p.stat().st_size)
        info["zip"] = str(z); info["size"] = z.stat().st_size
        info["md5"] = hashlib.md5(z.read_bytes()).hexdigest()
        try:
            zf = zipfile.ZipFile(z); info["testzip"] = zf.testzip(); info["n_files"] = len(zf.namelist())
            info["has_main"] = "main.py" in zf.namelist(); info["has_manifest"] = any(n.endswith("manifest.json") for n in zf.namelist())
        except zipfile.BadZipFile as e:
            info["testzip"] = f"BAD ZIP: {e}"
    md5s = list(out.rglob("*.md5")) + list(out.rglob("md5.txt"))
    if md5s:
        info["md5_expected_file"] = str(md5s[0]); info["md5_expected"] = md5s[0].read_text().split()[0] if md5s[0].read_text().split() else None
    return info


def fetch(kernel: str, out: Path) -> bool:
    """Download every output file of the latest version (the API pages 200 at a time; loop until no token)."""
    for attempt in range(3):
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi(); api.authenticate()
            token, n = None, 0
            while True:
                files, token = api.kernels_output(kernel, str(out), force=True, quiet=True, page_token=token, page_size=200)
                n += len(files)
                log(out, f"page fetched: {len(files)} files (total {n}); next token: {bool(token)}")
                if not token:
                    break
            return True
        except Exception as e:                     # noqa: BLE001
            log(out, f"api fetch attempt {attempt + 1} failed: {type(e).__name__}: {str(e)[-500:]}")
            r = run([kaggle_exe(), "kernels", "output", kernel, "-p", str(out), "--force", "--page-size", "200"])
            log(out, f"cli fallback rc={r.returncode} {(r.stdout + r.stderr).strip()[-400:]}")
            if r.returncode == 0:
                return True
            time.sleep(60)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="harshdipsaha/dat-parkinsons-train")
    ap.add_argument("--out", default=str(ROOT / "data" / "kaggle_output"))
    ap.add_argument("--once", action="store_true", help="just report status and exit (no download)")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    done, lock = out / "DONE.json", out / ".lock"
    if done.exists():
        log(out, "already fetched (DONE.json present); nothing to do")
        return 0
    if lock.exists() and time.time() - lock.stat().st_mtime < 3 * 3600:
        log(out, "another fetch in progress (lock < 3 h old); skipping")
        return 0
    st = status(a.kernel)
    log(out, f"status={st}")
    if a.once or st in ("RUNNING", "QUEUED") or st.startswith("UNKNOWN"):
        return 0
    lock.write_text(str(os.getpid()))
    try:
        ok = fetch(a.kernel, out)
        files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file() and p.name not in ("watch.log", ".lock"))
        log(out, f"downloaded {len(files)} files: {files[:40]}")
        info = verify(out) if ok else {}
        log(out, f"verify: {json.dumps(info)}")
        good = ok and info.get("zip") and info.get("testzip") is None and info.get("has_main")
        rec = {"kernel": a.kernel, "status": st, "fetched_utc": f"{datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S}", "ok": bool(good), **info, "files": files}
        if ok:
            done.write_text(json.dumps(rec, indent=2))
            log(out, "DONE " + ("zip verified" if good else "downloaded but zip missing/invalid -- check watch.log"))
            if os.name == "nt":
                run(["schtasks", "/Delete", "/TN", TASK, "/F"])
        return 0 if good else 1
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
