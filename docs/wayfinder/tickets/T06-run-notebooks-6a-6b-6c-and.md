---
id: T06
title: Run notebooks 6a, 6b, 6c and return their outputs
label: wayfinder:task
type: task
mode: HITL
status: open
assignee:
blocked_by: 
map: ../map.md
---

## Question

Launch `kaggle kernels push -p kaggle/run6a_crop3d`, `.../run6b_triplanar`, `.../run6c_features_ssl` on the teammate's account (guide: `docs/how-to-run-on-a-second-kaggle-account.md`; 6a must not overlap run 5 on the same account) and fetch the outputs with `scripts/watch_kaggle_output.py --kernel <owner>/<slug> --out data/run6x`. Resolved when the three output folders exist locally; record per-arm OOF from each APPROACH's e2e table.
