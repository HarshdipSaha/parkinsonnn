"""Validation gate for the dat-endgame deep-research pass.

    python docs/research/dat-endgame/validate.py

Fails if any item in outline.yaml is missing a result JSON, if a result is missing a field defined in
fields.yaml, or if a value marked [uncertain] is not declared in that item's "uncertain" array (and vice versa).
"""
import json
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
outline = yaml.safe_load((HERE / "outline.yaml").read_text(encoding="utf-8"))
fields = yaml.safe_load((HERE / "fields.yaml").read_text(encoding="utf-8"))
results = HERE / "results"

required = [k for k in fields if k != "uncertain"]
errors, seen = [], []

for item in outline["items"]:
    path = results / f"{item['id']}.json"
    if not path.exists():
        errors.append(f"{item['id']}: MISSING {path.name}")
        continue
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        errors.append(f"{item['id']}: invalid JSON ({e})")
        continue
    seen.append(item["id"])

    for f in required:
        if f not in d:
            errors.append(f"{item['id']}: missing field '{f}'")
    if "uncertain" not in d or not isinstance(d["uncertain"], list):
        errors.append(f"{item['id']}: missing or non-list 'uncertain'")
        continue

    flagged = {f for f in required if "[uncertain]" in json.dumps(d.get(f, ""), ensure_ascii=False)}
    declared = set(d["uncertain"])
    for f in flagged - declared:
        errors.append(f"{item['id']}: '{f}' contains [uncertain] but is not in the uncertain array")
    for f in declared - flagged:
        errors.append(f"{item['id']}: '{f}' declared uncertain but carries no [uncertain] marker")
    if d.get("confidence") not in fields["confidence"]["enum"]:
        errors.append(f"{item['id']}: confidence '{d.get('confidence')}' not in {fields['confidence']['enum']}")

print(f"items in outline: {len(outline['items'])}; results found: {len(seen)}")
if errors:
    print("\nVALIDATION FAILED")
    for e in errors:
        print(f"  - {e}")
    sys.exit(1)
n_unc = sum(len(json.loads((results / f'{i}.json').read_text(encoding='utf-8'))["uncertain"]) for i in seen)
print(f"VALIDATION OK — every item complete; {n_unc} field(s) flagged uncertain across {len(seen)} items")
