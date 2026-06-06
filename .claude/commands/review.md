---
description: Run the 3-reviewer + hard-signals review loop on the current uncommitted diff. Aggregates verdicts and manages the retry counter at .claude/state/review-attempts.json. On 2nd consecutive FAIL, writes an escalation report and surfaces it to the user.
---

# /review — Self-Review Loop Coordinator

You are running the **review coordinator** for Molebie AI's self-review loop. This is the manual entry point (the Stop hook will also invoke this same flow when Claude tries to finish a turn).

The flow you must execute, in order:

---

## Step 1 — Detect the diff

Run:
```bash
git status --porcelain && git diff HEAD --stat
```

If there are **no code changes** (only docs, only `.gitignore`, only files outside `gateway/`, `webapp/`, `cli/`, or no diff at all):
- Output a single line: `✓ /review: nothing to review (no code changes).`
- Stop. Do NOT invoke reviewers or hard signals.

Otherwise, continue.

---

## Step 2 — Fan out to all 3 reviewers + hard signals IN PARALLEL

In a **single message with multiple tool calls**, do the following five things concurrently:

1. `Agent` with `subagent_type: "reviewer-project-goal"` — prompt: *"Review the current uncommitted diff against your role. The user's most recent request is summarized as: <one-sentence summary of what the user asked for>. Read CLAUDE.md, README.md, and the diff (git diff HEAD). Output your verdict in the required format."*

2. `Agent` with `subagent_type: "reviewer-task-goal"` — prompt: *"Review the current uncommitted diff against your role. The user's most recent request is: <one-sentence summary of what the user asked for>. Judge whether the diff does that — no more, no less. Read the diff (git diff HEAD) and recent commits. Output your verdict in the required format."*

3. `Agent` with `subagent_type: "reviewer-senior-engineer"` — prompt: *"Review the current uncommitted diff against your role. Read CLAUDE.md and the diff (git diff HEAD). Apply the senior-staff-engineer bar. Output your verdict in the required format."*

4. `Bash` — scoped hard signals for Python:
   ```bash
   cd "$(git rev-parse --show-toplevel)/gateway" && \
     ruff check app/ tests/ 2>&1 | tail -30 && echo "---RUFF EXIT $?---" && \
     CHANGED_TESTS=$(git diff --name-only HEAD | grep -E '^gateway/tests/test_.*\.py$' | tr '\n' ' ') && \
     CHANGED_SRC=$(git diff --name-only HEAD | grep -E '^gateway/app/.*\.py$' | sed 's|gateway/app/|gateway/tests/test_|; s|\.py$|.py|' | tr '\n' ' ') && \
     SCOPED="$CHANGED_TESTS $CHANGED_SRC" && \
     if [ -n "$(echo $SCOPED | tr -d ' ')" ]; then \
       cd "$(git rev-parse --show-toplevel)/gateway" && \
       PYTHONDONTWRITEBYTECODE=1 pytest $(echo $SCOPED | xargs -n1 ls 2>/dev/null | sort -u) -x -q 2>&1 | tail -40 && echo "---PYTEST EXIT $?---"; \
     else \
       echo "PYTEST SKIPPED (no touched test files)"; \
     fi
   ```

5. `Bash` — scoped hard signals for webapp (only if webapp files touched):
   ```bash
   cd "$(git rev-parse --show-toplevel)" && \
     if git diff --name-only HEAD | grep -qE '^webapp/'; then \
       cd webapp && npx tsc --noEmit 2>&1 | tail -30 && echo "---TSC EXIT $?---"; \
     else \
       echo "TSC SKIPPED (no webapp changes)"; \
     fi
   ```

**All 5 calls go in a single message** so they execute in parallel — this is critical for keeping review latency under 60s.

---

## Step 3 — Aggregate the verdict

- **Overall PASS** = all 3 reviewers returned `VERDICT: PASS` AND ruff exit was 0 AND pytest exit was 0 (or skipped) AND tsc exit was 0 (or skipped).
- **Overall FAIL** = any reviewer FAIL OR any hard signal failure.

---

## Step 4 — Update the attempt counter

Read `.claude/state/review-attempts.json`. The file structure:
```json
{"task_id": "<sha256 of user's most recent request, first 8 chars>", "attempts": N, "updated_at": "<ISO8601>"}
```

If the file doesn't exist, treat `attempts` as 0 and the `task_id` as unmatched.

If `task_id` in the file matches the current task → use its `attempts` count.
If it doesn't match → this is a new task, reset to `attempts: 0`.

On PASS: delete the file (counter cleared).
On FAIL: increment `attempts` and write back. If `attempts >= 2`, jump to Step 6 (escalation).

---

## Step 5 — Write the result file

Write `.claude/state/last-review.json`:
```json
{
  "verdict": "PASS" | "FAIL",
  "attempt": N,
  "timestamp": "<ISO8601 UTC>",
  "diff_sha": "<git hash of the working tree diff>",
  "task_id": "<same as counter>",
  "reviewers": {
    "project_goal": "PASS" | "FAIL",
    "task_goal": "PASS" | "FAIL",
    "senior_engineer": "PASS" | "FAIL"
  },
  "hard_signals": {
    "ruff": "pass" | "fail",
    "pytest": "pass" | "fail" | "skipped",
    "tsc": "pass" | "fail" | "skipped"
  }
}
```

The Stop hook reads this file to decide whether to allow the next stop attempt.

---

## Step 6 — Output

### If PASS:
Print to chat:
```
✓ /review: PASS (attempt N)

1. Project Goal: PASS — <one-line reasoning>
2. Task Goal: PASS — <one-line reasoning>
3. Senior Engineer: PASS — <one-line reasoning>
Hard signals: ruff ✓ · pytest ✓ · tsc ✓
```

### If FAIL and attempts < 2:
Print to chat the full structured report:
```
✗ /review: FAIL (attempt N of 2)

### 1. Project Goal: PASS | FAIL
<full reasoning from reviewer>

### 2. Task Goal: PASS | FAIL
<full reasoning from reviewer>

### 3. Senior Engineer: PASS | FAIL
<full reasoning from reviewer>

### Hard signals
- ruff: pass | fail (<head of failure output if any>)
- pytest: pass | fail | skipped (<head of failure output if any>)
- tsc: pass | fail | skipped (<head of failure output if any>)

### Required changes
- [ ] <every required-change line from every reviewer, deduplicated>

I will now address these and re-run review.
```

Then **act on the required changes** — fix each one, then automatically re-invoke `/review` to retry. Do NOT report "done" to the user until either PASS or escalation.

### If FAIL and attempts >= 2:
Jump to Step 7 (escalation).

---

## Step 7 — Escalation (only on 2nd consecutive FAIL)

1. Compute timestamp slug: `date -u +%Y-%m-%dT%H-%M-%SZ`.
2. Compute task slug from the user's most recent request (kebab-case, max 40 chars).
3. Write `tasks/escalations/<timestamp>-<slug>.md`:

```markdown
# Escalation: <slug>

**Timestamp**: <ISO8601 UTC>
**Task ID**: <task_id>
**Branch**: <current branch>

## Original user request
<verbatim or one-paragraph summary>

## Diff under review
```diff
<output of git diff HEAD; truncate to 200 lines if longer>
```

## Attempt 1 verdict
<full structured report from attempt 1, all 3 reviewers + hard signals>

## Attempt 2 verdict
<full structured report from attempt 2>

## What changed between attempts
<diff summary of what Claude tried to fix between attempt 1 and 2; what new code was added/removed>

## Human, please clarify
<one short paragraph: what is the specific ambiguity, missing context, or design decision that the reviewers cannot resolve without your input?>
```

4. Append one line to `tasks/lessons.md`:
```
- <YYYY-MM-DD> · <slug> · 2x review fail · see tasks/escalations/<filename>
```

5. Print to chat:
```
⚠ /review: ESCALATED after 2 failed attempts.

I tried twice and could not satisfy the reviewers. Wrote a full report to:
  tasks/escalations/<timestamp>-<slug>.md

Summary of the blocker:
<3-sentence summary of what couldn't be resolved>

I am stopping here. Please review and clarify.
```

6. **Stop**. Do not retry a third time. Do not silently continue.

---

## Notes

- All three reviewer subagents are at `.claude/agents/reviewer-*.md`. Their system prompts are editable — tune one without touching the others.
- The reviewers have **no Write/Edit access**, so they cannot "fix" the diff — only judge it. You (the coder) act on their feedback.
- The hard signals are **non-overridable**: any hard FAIL (ruff/pytest/tsc) forces overall FAIL even if all 3 reviewers PASS. This guards against same-model sycophancy.
- The full pytest suite still runs in CI on push — the scoped pytest here is fast-path correctness signal, not full coverage.
