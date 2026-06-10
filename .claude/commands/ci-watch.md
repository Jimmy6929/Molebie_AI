---
description: Watch CI across ALL branches + main. Sweeps every workflow run, diagnoses each new failure via the ci-diagnoser subagent. Auto-apply only mechanical fixes; propose-only for ambiguous; hard-stop for security. Never runs git add/commit/push — the user owns those.
---

# /ci-watch — CI Watcher Coordinator (repo-wide)

You are the coordinator for the CI watcher loop. This is the post-push / always-sweeping
counterpart to the pre-push `/review` loop.

**Hard rule before you start anything**: you NEVER run `git add`, `git commit`,
`git push`, `gh pr merge`, `gh run rerun`, or any other action that mutates the
remote or commits work. After applying a fix, you print the exact commands for the
user to run; they decide.

## What changed vs. the old per-branch watcher

This command now monitors **every workflow run across every branch + `main`**, not
just the current branch's last push. It works in two auto-selected modes:

- **watch mode** — there are runs still `in_progress`/`queued` (typically right
  after *your* push). Poll them to completion, then diagnose.
- **sweep mode** — everything is already `completed` (typical at session start, or
  manual `/ci-watch`). Just diagnose any failure that is newly red and not already
  handled.

The unit of interest is **"the latest run for each (workflow, branch) pair."** A
branch is "currently red" for a workflow if that pair's most recent run failed. We
ignore superseded older failures (a newer run exists) — they're stale, not actionable.

---

## Step 1 — Enumerate all runs (repo-wide)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
gh -R "$(gh repo view --json nameWithOwner -q .nameWithOwner)" run list --limit 60 \
  --json databaseId,name,workflowName,headBranch,headSha,status,conclusion,event,createdAt,url
```

`gh run list` returns runs across **all** branches by default. If it returns nothing,
print *"No CI runs found for this repo. Is Actions enabled / are you authed (`gh auth status`)?"* and stop.

Skip the `Dependabot Updates` and `Copilot code review` workflows from *diagnosis*
(they are bot-managed and not code-fixable here) — but still report their status.

---

## Step 2 — Reduce to "latest run per (workflow, branch)"

Group the runs by `(workflowName, headBranch)`. For each group keep only the run with
the newest `createdAt`. This collapses churn to the current state of each pair.

```bash
# Given the JSON from Step 1 on stdin, emit one latest run per (workflow, branch).
python3 -c "
import json,sys
runs = json.load(sys.stdin)
latest = {}
for r in runs:
    key = (r['workflowName'], r['headBranch'])
    cur = latest.get(key)
    if cur is None or r['createdAt'] > cur['createdAt']:
        latest[key] = r
print(json.dumps(list(latest.values()), indent=2))
"
```

---

## Step 3 — Load the state ledger & compute what's new

Read `.claude/state/ci-watch.json` (run-keyed ledger). Schema:

```json
{
  "last_sweep_at": "<ISO ts>",
  "runs": {
    "<databaseId>": {
      "workflow": "CI",
      "branch": "feat/x",
      "sha": "abc1234",
      "event": "push",
      "status": "completed",
      "conclusion": "failure",
      "handled_at": "<ISO ts>",
      "incident_file": "tasks/ci-incidents/<file>.md"
    }
  }
}
```

A run from Step 2 needs attention if **either**:
- its `databaseId` is absent from `runs`, **OR**
- its stored record exists but stored `status != "completed"` (it was mid-flight last time).

**Cold-start guard**: if the ledger file does not exist yet, do NOT diagnose the
entire history. Only diagnose pairs whose latest run is currently `failure`/`timed_out`
**and** `createdAt` is within the last 7 days. Record everything else as handled
(baseline) without diagnosing. This prevents a noisy first run while still catching a
branch/main that is red *right now*.

---

## Step 4 — Wait for in-progress runs (watch mode, bounded)

If any attention-needing run has `status != "completed"` (`queued`/`in_progress`),
poll every **60s** until they complete, capped at **20 minutes** total:

```bash
NWO="$(gh repo view --json nameWithOwner -q .nameWithOwner)"
deadline=$(( $(date +%s) + 1200 ))
while true; do
  RUNS=$(gh -R "$NWO" run list --limit 60 --json databaseId,workflowName,headBranch,status,conclusion,createdAt)
  PENDING=$(echo "$RUNS" | python3 -c "import json,sys; d=json.load(sys.stdin); print(sum(1 for c in d if c.get('status')!='completed'))")
  if [ "$PENDING" = "0" ] || [ "$(date +%s)" -ge "$deadline" ]; then break; fi
  sleep 60
done
```

Print incremental progress in chat each cycle (which pairs finished and how). Update
`last_sweep_at` / the ledger as you go. If the 20-min cap is hit with runs still
pending, note which are still running and continue with whatever completed.

---

## Step 5 — Aggregate outcomes

For each attention-needing pair whose latest run is now `completed`:

- `success` → record in ledger as handled, no action.
- `failure` / `timed_out` → queue for diagnosis (Step 6).
- `cancelled` → record as handled, note in chat (usually intentional; don't diagnose).
- `neutral` / `skipped` → record as handled, non-blocking (flag only if a check that
  *should* have run was skipped).

If nothing is queued for diagnosis → go to Step 7 (success summary) and stop.

---

## Step 6 — Diagnose failures (priority order)

Handle failures one at a time in this priority order:

`Secret Scan` > `CodeQL` > `CI` > `Test Installers` > `Release` > anything else.

### 6a. Fetch the failed log

```bash
gh run view "<databaseId>" --log-failed 2>&1 | tail -200
```

Cap to the last ~200 lines — most failures are in the tail. (Run id comes straight
from the ledger/Step 2; no URL parsing needed.)

### 6b. Invoke the `ci-diagnoser` subagent

Call `ci-diagnoser` with:
- The workflow/check name (e.g. `CI`, `Analyze (python)`, `Scan for leaked secrets`, `test-macos`).
- The failed log (last ~200 lines).
- The branch + commit SHA + whether this is `main` (a red `main` is higher urgency).
- The trigger `event` (`push`, `pull_request`, `schedule`, `release`, `dynamic` for Dependabot).

It returns:
```
CHECK: <name>
CATEGORY: auto-fix-safe | propose-only | hard-stop
ROOT CAUSE: <one paragraph>
FIX: <diff or commands>
CONFIDENCE: high | medium | low
RATIONALE: <evidence>
```

### 6c. Branch by category

#### `auto-fix-safe`
1. Apply the fix with Edit/Write.
2. Quick local sanity check matching the failure type:
   - ruff → `cd gateway && ../.venv/bin/ruff check app/ tests/`
   - eslint → `cd webapp && npm run lint`
   - pip-audit → `cd gateway && ../.venv/bin/pip-audit -r requirements.txt`
3. Sanity passes → write incident report (§6d) with status `APPLIED`.
4. Sanity fails → revert the edit, downgrade to `propose-only`, re-run §6c.

**Only ever auto-apply on a branch you can edit locally.** If the failure is on a
branch other than the current checked-out branch, do NOT switch branches and do NOT
edit — downgrade to `propose-only` and tell the user which branch to check out.

#### `propose-only`
Do NOT apply. Write the incident report with the proposed patch marked `PROPOSED`.
Surface with a clear "your call" framing.

#### `hard-stop`
Do NOT apply, do NOT propose code changes. Write the incident report with status
`HALT`. Surface with urgency + the diagnoser's user-actionable steps (rotation, etc.).

### 6d. Write the incident report

Path: `tasks/ci-incidents/<UTC-ts>-<branch-slug>-<check-slug>.md`.

```markdown
# CI Incident: <check-slug> on <branch>

**When**: <ISO UTC>
**Branch / commit**: <branch> @ <short-sha>
**Workflow / run**: <workflow> (run <databaseId>) · <url>
**Trigger**: <event>
**Category**: auto-fix-safe | propose-only | hard-stop
**Status**: APPLIED | PROPOSED | HALT

## Root cause
<from diagnoser>

## What I did
<diff applied, or "Proposed only — not applied.", or "Halted — no file changes.">

## What you need to do
<APPLIED: exact commands — git diff to review, git add ..., git commit -m "...", git push>
<PROPOSED: review the proposed patch above, decide, apply manually, push>
<HALT: rotation/revocation/escalation steps from the diagnoser>

## Diagnoser verdict (full)
<verbatim diagnoser output>
```

Append a one-line entry to `tasks/lessons.md`:
```
- <YYYY-MM-DD> · ci-<branch-slug>-<check-slug> · <category> · see tasks/ci-incidents/<filename>
```

### 6e. Record the run in the ledger

Add/update the run's entry in `.claude/state/ci-watch.json` with `status: completed`,
its `conclusion`, `handled_at`, and `incident_file`. This is what prevents re-diagnosing
the same red run on the next sweep.

---

## Step 7 — Summaries to chat

**Success summary** (when nothing needed diagnosis):
```
✓ CI swept — all green across <N> branches.

Latest per workflow:
  main                        ✓ CI  ✓ CodeQL  ✓ Secret Scan
  feat/tiered-storage-service ✓ CI  ✓ CodeQL  ✓ Secret Scan

Nothing to do. (You own merges — I don't touch them.)
```

**Failure summary** (per handled failure):
```
✗ <workflow> failed on <branch> @ <short-sha> — <CATEGORY>

<one-line root cause>

<APPLIED:>
Applied locally. Review:  git diff
Then:  git add <files> && git commit -m "<msg>" && git push

<PROPOSED:>
Drafted a patch, did NOT apply. Review: tasks/ci-incidents/<file>

<HALT:>
⚠ HARD STOP — security-sensitive. Do NOT push until: <steps>
Report: tasks/ci-incidents/<file>
```

---

## Step 8 — Persist the ledger

Write `.claude/state/ci-watch.json` with `last_sweep_at` updated and the `runs` map
capped to the most recent ~100 entries (sort by `handled_at`, drop the oldest). Never
edit this file by hand outside this command.

---

## Notes

- **Idempotence**: re-running `/ci-watch` picks up the ledger and only acts on runs
  that are newly red or were mid-flight last time. Never restarts from scratch.
- **Never run** `git add/commit/push`, `gh pr merge`, `gh pr review --approve`,
  `gh run rerun`, or anything that mutates the remote.
- **Never edit** `CLAUDE.md`, `.gitignore`, `README.md`, or the `.github/workflows/*`
  YAMLs as part of a CI fix. Scope the fix to whatever caused the failure.
- **Cross-branch fixes are propose-only.** You only auto-apply on the branch currently
  checked out; for any other branch, propose and tell the user to check it out.
- **Flaky tests**: if the diagnoser flags an intermittent failure (no clear root cause,
  prior runs of the same test passed), category is `propose-only` with FIX suggesting
  the user run `gh run rerun <id>` (you never rerun it yourself).
- **Red `main` is top priority** — surface it first and loudly.
