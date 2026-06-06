# CI Watcher — Local Tooling

**Status**: local-only (Layer 1). Nothing in this loop is committed to git. The
project's `CLAUDE.md`, `.gitignore`, and CI workflows are untouched. Layer 2 (an
optional cloud Routine) lives in your Claude account, not in this repo.

## What it does

The watcher monitors **every workflow run across every branch + `main`** — not just
your last push. It runs in two layers:

- **Layer 1 (local, while you work)** — Claude sweeps all CI via `/ci-watch`. On any
  branch/main that is currently red, a specialized diagnoser subagent classifies the
  failure and either auto-applies a safe fix, proposes a patch, or hard-stops on
  security findings. You always own `git add` / `commit` / `push` / merge.
- **Layer 2 (cloud, while you're away)** — an optional Claude Code Routine, triggered
  by GitHub `workflow run` events, auto-diagnoses failures that land when no session
  is open and opens a draft fix PR. See "Layer 2" below.

```
                    ┌─ git push (PostToolUse hook) ─┐
session start ──────┤                               ├──► /ci-watch
(SessionStart hook) └─ manual / /loop ──────────────┘   (sweep all branches)
                                                              │
                                  gh run list → latest run per (workflow, branch)
                                                              │
                                       ┌──────────────────────┴──────────────────────┐
                                       ▼                                              ▼
                                  all green                                     any red pair
                                       │                                              │
                                       ▼                                              ▼
                                 "✓ swept, green"                          ci-diagnoser subagent
                                                                                      │
                                  ┌───────────────────────────┬───────────────────────────┐
                                  ▼                           ▼                           ▼
                             auto-fix-safe              propose-only                  hard-stop
                             (lint, fmt, unreachable    (real test fail, semantic     (gitleaks,
                              CVE ignore)                err, install/release,         CodeQL critical,
                              — only on checked-out      Dependabot bump,              credentials)
                              branch                     cross-branch fixes)
                                  │                           │                           │
                             apply locally,            write patch to incident,    surface only,
                             write incident            do NOT apply                rotation/revoke
                                  └───────────────► incident under ◄────────────────────┘
                                               tasks/ci-incidents/<ts>.md
                                                          │
                                                          ▼
                                                   you review, decide, push
```

## File map

| File | Purpose | Edit when… |
|------|---------|------------|
| `.claude/hooks/session-start-sweep.sh` | **SessionStart** hook. On a fresh session (startup/resume/clear, not compact), nudges Claude to run a repo-wide `/ci-watch` sweep. Throttled to once per 30 min. | You want to change the throttle, the session sources, or disable the auto-sweep. |
| `.claude/hooks/post-push-hook.sh` | **PostToolUse** hook on Bash. Detects a successful `git push`, nudges `/ci-watch` (which then waits for your just-triggered runs). | You want to change when the post-push watch fires (e.g., skip certain branches). |
| `.claude/commands/ci-watch.md` | The `/ci-watch` coordinator — sweeps all runs, dedups, watches in-flight runs, classifies, applies, reports. | You want to change cadence, scope, or branching logic. |
| `.claude/agents/ci-diagnoser.md` | Specialized diagnoser subagent — categorizes failures, proposes fixes; includes trigger-specific rules (Dependabot/schedule/release/main). | You want to tune the auto-fix-safe whitelist, expand hard-stop, or change the verdict format. |
| `.claude/settings.local.json` | Wires the SessionStart + PostToolUse hooks (and the Stop hook from the review loop). | Only to disable: remove the relevant hook block. |
| `.claude/state/ci-watch.json` | Run-keyed ledger (auto-managed). One entry per workflow run, so failures are diagnosed exactly once. | Never edit manually. |
| `tasks/ci-incidents/*.md` | One file per CI incident — root cause, what was done, what you need to do. | Read when surfaced. |
| `tasks/lessons.md` | Newest-first one-line entries summarizing each incident (auto-appended). | Curate as you like. |

## Triggers (Layer 1)

- **Session start** (`session-start-sweep.sh`): every fresh session nudges one
  repo-wide sweep, throttled to once per 30 min. This is what catches failures that
  landed while you were away — merges to `main`, weekly scheduled CodeQL/Secret Scan,
  Dependabot PRs, release tags, pushes from another machine.
- **After `git push`** (`post-push-hook.sh`): fires `/ci-watch`, which waits for the
  runs your push just triggered, then diagnoses. Skipped for `--dry-run`, failed
  pushes, and `git pull`.
- **Manual**: type `/ci-watch` any time.
- **Continuous**: `/loop 15m /ci-watch` re-sweeps every 15 min during a long session.

## Scope

Sweeps **all** workflows on **all** branches + `main`, every trigger type:
CI, CodeQL, Secret Scan, Test Installers, Release. `Dependabot Updates` and
`Copilot code review` are reported but not diagnosed (bot-managed, not code-fixable
here). Dependabot version-bump *failures* in CI/Test Installers are diagnosed as
`propose-only`.

## How "currently red" is computed

The unit of interest is **the latest run for each (workflow, branch) pair**. A pair is
red if its most recent run's `conclusion` is `failure` or `timed_out`. Superseded older
failures (a newer run exists for the same pair) are stale and ignored. `cancelled` is
recorded but not diagnosed; `neutral`/`skipped` are non-blocking (flagged only if a
check that should have run was skipped). Each red run is diagnosed once — the
run-keyed ledger (`ci-watch.json`) prevents re-diagnosis on the next sweep.

**Cold start**: with no ledger yet, the first sweep does NOT diagnose all history — it
only diagnoses pairs that are red *right now* and created within the last 7 days, and
baselines the rest. This avoids a noisy first run.

## Fix policy (the most important section)

Three categories. The diagnoser picks exactly one per failure.

### auto-fix-safe — applied automatically (only on the checked-out branch)

| Category | Required evidence |
|---|---|
| ruff lint with `--fix` | log shows ruff output + `[*]` fix marker |
| eslint with `--fix` | log shows eslint output + `(fixable)` |
| isort / black / prettier | formatter check failed |
| Dep CVE ignore for unreachable vuln | diagnoser must grep + cite zero use sites of the affected symbol |
| Trivial type-annotation fix | `-> None`, `Optional[X]` → `X | None`, etc. |

Auto-fix is applied **only** when the red branch is the one currently checked out.
A failure on any other branch is downgraded to `propose-only` (the watcher never
switches branches).

### propose-only — written to incident report, NOT applied

Real test failures, semantic type errors, install/release script bugs, medium/low
CodeQL findings, dep version conflicts, Dependabot bumps, and any cross-branch fix.
The fix is suggested as a unified diff; you decide.

### hard-stop — NEVER auto-fix, surface only

- gitleaks finds a real-looking secret → rotation/revocation guidance, never `sed` to
  remove (doesn't fix history)
- CodeQL critical-severity finding → design-level review required
- Anything touching credentials / tokens / API keys / certs

## Layer 2 — cloud Routine for when you're away (the "bridge")

Claude Code Routines **cannot** trigger natively on CI failures — GitHub event
triggers only support `pull_request` and `release`, not `workflow_run`. So Layer 2
bridges GitHub Actions to a Routine's **API trigger**:

```
any workflow fails → .github/workflows/ci-failure-notify.yml (on: workflow_run)
                       → POST routine /fire endpoint (failing run + log link in body)
                          → cloud Routine: diagnose → open DRAFT fix PR / hard-stop issue
```

Fires within seconds of any failure on any branch, laptop closed. The watchdog
workflow (`.github/workflows/ci-failure-notify.yml`) is the one **committed** part of
this system — `workflow_run` workflows run from the default branch, so it must be on
`main` to work. It no-ops until the two secrets exist.

### One-time setup runbook

1. **Install the Claude GitHub App** on `Jimmy6929/Molebie_AI`:
   https://claude.ai/code/onboarding?magic=github-app-setup
   (Gives the Routine clone + PR access. `/web-setup` alone is *not* enough for PRs.)
2. **Create the Routine** at https://claude.ai/code/routines → **New routine** → **Remote**:
   - Name: `Molebie CI failure responder`
   - Repository: `Jimmy6929/Molebie_AI`
   - Environment: `Default` (Trusted network — GitHub/`gh` reachable)
   - Model: `claude-sonnet-4-6` (bump to Opus for harder diagnoses)
   - Prompt: paste the **Routine prompt** below.
   - Trigger: **Add trigger → API → Save** → **Generate token** → copy the **URL** and **token**.
3. **Add repo Actions secrets** (Settings → Secrets and variables → Actions), or run locally:
   ```bash
   gh secret set CLAUDE_ROUTINE_URL   --body '<the /fire URL>'
   gh secret set CLAUDE_ROUTINE_TOKEN --body '<the token>'
   ```
4. **Merge `ci-failure-notify.yml` to `main`** (own branch + PR, per GitHub Flow). It
   only activates from the default branch.
5. **Test**: routine page → **Run now** (sanity), then push a deliberately failing
   commit to a throwaway branch and confirm a draft PR / issue appears.

### Routine prompt (paste into the form)

```
You are the Molebie AI CI failure responder, a cloud routine triggered via API when a
GitHub Actions workflow fails. The triggering message contains the workflow name,
branch, commit SHA, failed run id, and run URL.

Do this:
1. Parse the run id and branch from the message.
2. Fetch the failed log: `gh run view <run_id> --log-failed` (last ~200 lines).
3. Read `.claude/agents/ci-diagnoser.md` in the cloned repo and apply its rubric to
   classify the failure: auto-fix-safe | propose-only | hard-stop.
4. Act by category:
   - auto-fix-safe / propose-only: create a `claude/ci-fix-<short-sha>` branch off the
     failing branch, apply the fix, and open a DRAFT pull request. Body: root cause
     (cite file:line + error codes), the fix, and the category. Do NOT mark ready, do
     NOT merge.
   - hard-stop (leaked secret, CodeQL critical, anything credential-adjacent): do NOT
     push code. Open an issue titled "CI hard-stop: <workflow> on <branch>" with the
     redacted finding and rotation/revocation guidance.
5. One PR/issue per failure: before creating, check for an existing open one for the
   same branch+workflow and comment on it instead of duplicating.
6. Never merge, never force-push. Keep changes scoped to the failure. If the log is
   unavailable or unparseable, open an issue summarizing what you saw and stop.
```

### Trade-offs
Runs in Anthropic cloud (clones the repo), uses Claude usage, research-preview feature,
subject to per-account hourly webhook/run caps. Overlaps Layer 1 by design — Layer 1's
run-keyed ledger dedups locally; the Routine covers the "no session open" case. If
Layer 1 proves enough, you can leave the secrets unset and the watchdog stays dormant.

## Bypassing or disabling

- **Disable the auto-sweep on session start**: remove the `SessionStart` block in
  `.claude/settings.local.json` (or delete `session-start-sweep.sh`). `/ci-watch`
  still works manually.
- **Disable the post-push watch**: remove the `PostToolUse` block.
- The watcher only watches; it never gates a push.

## Relationship to the pre-push review loop

| Layer | Catches | When |
|---|---|---|
| `/review` (pre-push, Stop hook) | local lint, scoped tests, 3-reviewer rubric | before you ship |
| `/ci-watch` (Layer 1, SessionStart + post-push) | CodeQL, gitleaks, dep CVE DB updates, cross-platform installer tests, real CI flake — across all branches | after you push / while you work |
| Routine (Layer 2, cloud) | the same, for failures that land while no session is open | always-on |

## Plan & rationale

This document is the design rationale. The implementation lives in
`.claude/commands/ci-watch.md` (coordinator), `.claude/agents/ci-diagnoser.md`
(failure classifier), and the `.claude/hooks/` SessionStart + post-push nudges.

## Caveats

- **Layer 1 only runs while Claude Code is open.** The SessionStart sweep + post-push
  hook + `/loop` cover "while you work." For laptop-closed coverage, use Layer 2.
- **`gh` CLI must be authed.** Already is on this machine.
- **Polling at 60s**, 20-min cap per watch — well under GitHub rate limits.
- **Auto-fix never commits or pushes.** Even on success, the watcher only edits files;
  you review with `git diff` and decide.
- **Cross-branch failures are propose-only** — the watcher never checks out another
  branch to apply a fix.
