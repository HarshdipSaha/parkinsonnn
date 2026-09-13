"""Build report.md from the dat-endgame deep-research JSONs.

    python docs/research/dat-endgame/generate_report.py

Reads outline.yaml for item order, fields.yaml for the field schema, and results/*.json for the findings.
Per the Deep-Research-skills report step, values marked [uncertain] are kept out of the body; the field names
are surfaced under "Needs checking" instead of being dropped silently, so nothing is hidden.
"""
import json
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
outline = yaml.safe_load((HERE / "outline.yaml").read_text(encoding="utf-8"))
fields = yaml.safe_load((HERE / "fields.yaml").read_text(encoding="utf-8"))
results = HERE / "results"

ORDER = ["verdict", "confidence", "numbers", "evidence", "risk_if_wrong", "action", "rejected_alternatives", "open_questions"]
TITLES = {
    "verdict": "Verdict", "confidence": "Confidence", "numbers": "Numbers", "evidence": "Evidence",
    "risk_if_wrong": "Risk if wrong", "action": "Action", "rejected_alternatives": "Rejected alternatives",
    "open_questions": "Open questions",
}


def anchor(name):
    return "".join(c for c in name.lower().replace(" ", "-") if c.isalnum() or c == "-")


def render(value, indent=""):
    """Flat strings, lists of strings, and dicts of scalars all appear in these results."""
    if isinstance(value, dict):
        if not value:
            return f"{indent}_(none)_\n"
        rows = "".join(f"{indent}| `{k}` | {str(v).replace('|', '\\|')} |\n" for k, v in value.items())
        return f"{indent}| metric | value |\n{indent}|---|---|\n{rows}"
    if isinstance(value, list):
        if not value:
            return f"{indent}_(none)_\n"
        out = ""
        for v in value:
            if isinstance(v, dict):
                out += f"{indent}- " + "; ".join(f"**{k}**: {vv}" for k, vv in v.items()) + "\n"
            else:
                out += f"{indent}- {v}\n"
        return out
    return f"{indent}{value}\n"


def main():
    items, missing = [], []
    for it in outline["items"]:
        p = results / f"{it['id']}.json"
        if p.exists():
            items.append((it, json.loads(p.read_text(encoding="utf-8"))))
        else:
            missing.append(it["id"])

    ctx = outline["context"]
    L = [
        f"# {outline['topic']}", "",
        f"Deep-research pass generated {outline['generated']} using the Deep-Research-skills workflow ",
        "(`research` -> `research-deep` -> `research-report`), with one agent per item and a validation gate.", "",
        "| | |", "|---|---|",
        f"| Now | {ctx['now_utc']} UTC |",
        f"| Deadline | {ctx['deadline_utc']} UTC |",
        f"| Last slot opens | {ctx['slot_opens_utc']} UTC (it opens; it does not expire) |",
        f"| Real submissions left | {ctx['real_submissions_remaining']} |",
        f"| Banked | public {ctx['banked_public']}, nested {ctx['banked_nested']}, rank {ctx['banked_rank']} |",
        f"| Leader / top 10 | {ctx['leader_public']} / {ctx['top10_public']} |",
        f"| GPU-hours left | {ctx['gpu_hours_remaining']}+ |",
        f"| Zip | {ctx['zip_size_mb']} MB of {ctx['zip_ceiling_mb']} MB, {ctx['checkpoints']} checkpoints |",
        "", "## Contents", "",
        "| # | Item | Confidence | Verdict |", "|---|---|---|---|",
    ]
    for i, (it, d) in enumerate(items, 1):
        v = str(d.get("verdict", "")).replace("|", "\\|")
        L.append(f"| {i} | [{it['name']}](#{anchor(it['name'])}) | {d.get('confidence', '?')} | {v} |")
    if missing:
        L += ["", f"> Items with no result file: {', '.join(missing)}"]

    for it, d in items:
        unc = set(d.get("uncertain", []))
        L += ["", "---", "", f"## {it['name']}", "", f"**Question.** {it['question']}", ""]
        for f in ORDER:
            if f not in d or f in unc:
                continue
            L += [f"### {TITLES[f]}", "", render(d[f]).rstrip(), ""]
        extra = [k for k in d if k not in ORDER and k != "uncertain" and not k.startswith("_")]
        if extra:
            L += ["### Other info", ""] + [render({k: d[k] for k in extra}).rstrip(), ""]
        if unc:
            L += [f"> **Needs checking** — could not be established from the repo, the rules or arithmetic: {', '.join(sorted(unc))}.", ""]

    out = HERE / "report.md"
    out.write_text("\n".join(L).rstrip() + "\n", encoding="utf-8")
    print(f"wrote {out} — {len(items)} item(s), {len(missing)} missing, "
          f"{sum(len(d.get('uncertain', [])) for _, d in items)} field(s) flagged uncertain")


if __name__ == "__main__":
    main()
