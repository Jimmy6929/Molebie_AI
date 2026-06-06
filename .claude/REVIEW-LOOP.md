# Self-Review Loop — Local Tooling

**Status**: local-only. Nothing in this loop is committed to git. The project's `CLAUDE.md`, `.gitignore`, and CI are untouched.

## What it does

After Claude makes code changes, a Stop hook intercepts the "I'm done" signal and forces a review through three specialized judges + automated hard signals (ruff, scoped pytest, tsc). If the review FAILs, Claude must address the feedback and re-run. After 2 consecutive FAILs, the loop escalates to you with a written report.

```
Coder ──► 3 reviewers (parallel) + hard signals ──► PASS or FAIL
   ▲                                                    │
   └───── re-code with feedback (1 retry) ◄─────────────┘
                                                        │
                                       2nd FAIL ──► tasks/escalations/*.md
```

## File map

| File | Purpose | Edit when… |
|------|---------|------------|
| `.claude/agents/reviewer-project-goal.md` | Reviewer #1 — judges project fit (privacy / local-first / zero-Docker) | You want to change what "fits Molebie" means |
| `.claude/agents/reviewer-task-goal.md` | Reviewer #2 — judges scope creep & under-delivery | You want to tune what "the user asked for X" means |
| `.claude/agents/reviewer-senior-engineer.md` | Reviewer #3 — judges code quality (root cause, simplicity, security) | You want to raise/lower the engineering bar |
| `.claude/commands/review.md` | The `/review` coordinator — fans out, aggregates, manages state | You want to add/remove a hard signal or change output format |
| `.claude/hooks/review-stop-hook.sh` | Stop-hook script — decides allow vs. block | You want to change when the loop fires (e.g., per-language) |
| `.claude/settings.local.json` | Wires the Stop hook in (local-only) | Only to disable: comment out the `hooks` block |
| `.claude/state/review-attempts.json` | Per-task retry counter (auto-managed) | Never edit manually |
| `.claude/state/last-review.json` | Verdict + diff SHA from most recent /review (auto-managed) | Never edit manually |
| `tasks/escalations/*.md` | Escalation reports written on 2nd consecutive FAIL | Read when escalated |
| `tasks/lessons.md` | One-line summary of every escalation (newest first) | Auto-appended; you can curate |

## Triggers

- **Manual**: type `/review` at any time. Useful for ad-hoc checks or pre-PR review.
- **Automatic**: Stop hook fires when Claude tries to end a turn with uncommitted code changes.

## Bypassing the loop

Include the sentinel `[skip-review]` anywhere in your message to bypass the Stop hook for that turn. Use sparingly — typo fixes, doc tweaks, config-only changes.

## Disabling temporarily

Edit `.claude/settings.local.json` and comment out (or remove) the `hooks` block. The reviewer subagents and `/review` slash command will still work manually.

## How "PASS" is computed

```
overall PASS = reviewer-project-goal.verdict == PASS
            AND reviewer-task-goal.verdict   == PASS
            AND reviewer-senior-engineer.verdict == PASS
            AND ruff exit == 0
            AND pytest exit == 0  (or skipped — no touched test files)
            AND tsc exit == 0     (or skipped — no webapp changes)
```

Hard signals are **non-overridable**: any hard FAIL forces overall FAIL even if all 3 reviewers PASS. This guards against same-model sycophancy (Opus reviewing Opus tends to drift toward politeness).

## Retry logic

- First FAIL: hook blocks stop, injects the full review report as a directive. Claude (the coder) addresses required changes and re-runs `/review`.
- Second consecutive FAIL on the same task: `/review` writes `tasks/escalations/<timestamp>-<slug>.md` and surfaces it. No third retry.

Task identity is computed from the user's most recent request (sha256, first 8 chars). A new user request resets the counter.

## Plan & rationale

This document is the design rationale. The implementation lives in
`.claude/commands/review.md` (coordinator), `.claude/agents/reviewer-*.md`
(the three reviewer rubrics), and `.claude/hooks/review-stop-hook.sh`.

## Caveats

- **Same-model reviewer**: all reviewers run on `claude-opus-4-7`, same as the coder. Research warns this is the riskiest setup for sycophancy. Mitigated by (a) adversarial role prompts ("default to FAIL when in doubt"), (b) hard signals as ground truth, (c) reviewers have no Write/Edit access. **If you see the loop pass diffs you wouldn't have approved**, switch one reviewer to Sonnet 4.6 — change the `model:` line in the frontmatter.
- **Scoped pytest only**: the loop runs only test files touching changed source paths. Full pytest still runs in CI on push. Don't rely on the loop alone for coverage.
- **State files in `.claude/state/`** are gitignored via `.git/info/exclude`. If you clone this repo fresh, you'll need to re-add those entries (see plan §7).
