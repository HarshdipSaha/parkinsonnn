"""List the wayfinder frontier: open, unblocked, unclaimed tickets. python docs/wayfinder/frontier.py"""
import re
from pathlib import Path
T = Path(__file__).resolve().parent / "tickets"
tickets = {}
for p in sorted(T.glob("T*.md")):
    s = p.read_text(encoding="utf-8"); fm = s.split("---")[1] if s.startswith("---") else ""
    get = lambda k: (re.search(rf"^{k}:[ \t]*(.*)$", fm, re.M) or [None, ""])[1].strip()
    tickets[p.stem.split("-")[0]] = dict(file=p.name, title=get("title"), status=get("status"), assignee=get("assignee"), blocked_by=[b for b in re.split(r"[,\s]+", get("blocked_by")) if b], type=get("type"), mode=get("mode"))
for tid, t in tickets.items():
    blocked = [b for b in t["blocked_by"] if tickets.get(b, {}).get("status") != "closed"]
    state = t["status"] if t["status"] != "open" else ("blocked by " + ", ".join(blocked) if blocked else ("claimed by " + t["assignee"] if t["assignee"] else "FRONTIER"))
    print(f"{tid:<4} {t['type']:<10} {t['mode']:<4} {state:<28} {t['title']}")
