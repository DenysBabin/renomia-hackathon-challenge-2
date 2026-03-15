---
name: feedback_focus_blacklist
description: User wants to focus on blacklist/token reduction, not endlessly tweak extraction rules
type: feedback
---

Stop endlessly iterating on extraction_rules.py prompt tweaks. User prefers deterministic approaches (blacklist filtering, postprocessing) over prompt engineering.

**Why:** Diminishing returns from prompt changes — Gemini is non-deterministic and small prompt tweaks cause regressions elsewhere.

**How to apply:** When score improvement is needed, prioritize token reduction (blacklist), deterministic postprocessing, and regex fallbacks over extraction_rule text changes.
