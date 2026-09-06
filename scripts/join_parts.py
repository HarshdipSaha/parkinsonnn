"""Reassemble submission_v3.zip from downloaded parts and verify the md5 printed by the split cell.
python scripts/join_parts.py data/parts_v3 <md5> [--out submission_v3.zip]"""
import glob, hashlib, sys, zipfile
from pathlib import Path
d, md5 = Path(sys.argv[1]), sys.argv[2]
out = Path(sys.argv[4] if len(sys.argv) > 4 else "submission_v3.zip")
parts = sorted(glob.glob(str(d / "submission_v3.zip.part*")))
with open(out, "wb") as w:
    for p in parts:
        w.write(open(p, "rb").read())
got = hashlib.md5(open(out, "rb").read()).hexdigest()
print(len(parts), "parts ->", out, out.stat().st_size, "bytes; md5", got, "OK" if got == md5 else "MISMATCH")
if got == md5:
    z = zipfile.ZipFile(out); print("testzip:", z.testzip(), "|", len(z.namelist()), "files")
