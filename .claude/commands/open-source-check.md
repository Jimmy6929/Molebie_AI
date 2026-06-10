---
description: Audit the repo (or current diff) for open-source / job-hunt readiness — report-only
---

Audit this repo — or the current uncommitted diff if there is one — for open-source
and job-hunt / portfolio readiness. The question you are answering is: **can an outside
engineer or hiring manager understand, run, trust, and find the depth in this project
within their first ~60 seconds?**

Reason like a senior engineer reviewing a candidate's portfolio repo: rigorous, calm,
allergic to busywork. This repo is already mature — do NOT recommend recreating things
that exist (README, CONTRIBUTING, SECURITY, LICENSE, docs/, CI workflows). Find the
*small* gaps that actually move the needle.

This command is **report-only**. Do not edit files, do not commit. Recommend the
**smallest useful fix** for each gap and let the user decide.

Check, in roughly this order:

1. **Claims match reality — verify, don't trust.** Every file path, command, and link
   referenced in `README.md` and `docs/` must actually resolve. Run them down (the path
   exists, the `make` target exists, the URL isn't dead). A README that points a reviewer
   at a file that doesn't exist is worse than no pointer at all.
2. **Reviewer path.** Is there a "where to look" / Code Tour section pointing at the
   strongest files? Are those genuinely the strongest, and do they all exist?
3. **One-command local verification.** Is there a single obvious entry point (`make verify`
   or equivalent) and does it actually mirror what CI gates on? Flag drift between the
   local command and `.github/workflows/`.
4. **Verification visibility.** Can a reviewer *see* the CI / CodeQL / secret-scan /
   installer-test / Dependabot story from the README, without digging into `.github/`?
5. **Staleness & rot.** Stale docs, dead commands, oversized files that should be split,
   duplicated helpers, TODO/FIXME left in shipped code.
6. **Touched-code quality** (diff mode): broad `except Exception` without a boundary
   reason, swallowed errors, untested new behavior, scope creep beyond the change.

Output format — group findings, and for each one give exactly:

```
Gap: <what's missing or wrong>
Why it matters: <what a reviewer concludes / the cost>
Smallest useful fix: <the one cheapest correct step>
```

End with a one-line readiness verdict. Do not auto-create a pile of files; surface the
gaps and stop.

$ARGUMENTS
