# Wayfinder tracker (local markdown)

No issue tracker is configured for this repo (run `/setup-matt-pocock-skills` to attach one), so the wayfinder map lives here.

- **Map**: `map.md` (label `wayfinder:map`) — destination, notes, decisions so far (index of closed tickets), fog (not yet specified), out of scope.
- **Tickets**: `tickets/T<nn>-<slug>.md` (label `wayfinder:<research|prototype|grilling|task>`, mode HITL/AFK). Front matter holds `status` (open / claimed / closed), `assignee` (the claim), `blocked_by` (native dependency substitute: ticket ids).
- **Frontier** = open, unblocked (all `blocked_by` closed), unassigned tickets. List it with `python docs/wayfinder/frontier.py`.
- **Resolution**: append a `## Resolution` section to the ticket, set `status: closed`, add one line to the map's *Decisions so far*, then graduate fog into new tickets.
- Names, not ids, in prose: refer to tickets by their title.
