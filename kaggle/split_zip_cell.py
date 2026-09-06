# === SPLIT submission_v3.zip into <95 MB parts for browser download (Kaggle's in-editor download caps ~100 MB) ===
import os, subprocess, hashlib
os.makedirs('parts', exist_ok=True); subprocess.run('rm -f parts/*', shell=True)
subprocess.run('split -b 95m -d -a 2 submission_v3.zip parts/submission_v3.zip.part', shell=True, check=True)
md5 = hashlib.md5(open('submission_v3.zip', 'rb').read()).hexdigest()
open('parts/MD5', 'w').write(md5 + '  submission_v3.zip\n')
print('md5', md5, 'size', os.path.getsize('submission_v3.zip'))
print(sorted(os.listdir('parts')))
