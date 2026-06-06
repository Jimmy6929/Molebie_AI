# Branch Manager — Local Tooling

**Status**: local-only (Layer 1). Nothing in this loop is committed to git or mutates
the repo. The project's `CLAUDE.md`, `.gitignore`, and CI workflows are untouched. Layer 2
(an optional cloud Routine) lives in your Claude account, not in this repo.

This is the **branch-hygiene sibling of the CI watcher** (`.claude/CI-WATCHER.md`). Same
architecture — coordinator command + adversarial subagent + run-keyed ledger + nudge
hooks — applied to branches instead of CI runs.

## What it does

Audits **every branch in the repo** against the GitHub Flow rules in `CLAUDE.md` (atomic
short-lived branches; `feat|fix|chore|docs|refactor/` names; squash-merge to `main`;
auto-delete-on-merge; no long-lived branches). It runs in two layers:

- **Layer 1 (local, while you work)** — Claude sweeps all branches via `/branch-manager`.
  Each branch is classified by the `branch-auditor` subagent, then any actionable
  recommendation is grounded with a `/best-way` research pass. It **tells you what to
  add / delete / do** and prints the exact commands. **You always own `git push --delete`
  / `gh pr merge` / rebase — it never touches a branch.**
- **Layer 2 (cloud, while you're away)** — an optional Claude Code Routine on a **cron**
  trigger audits branches on a schedule and opens/updates a single tracking issue. See
  "Layer 2" below.

```
                    ┌─ gh pr merge / push to main (PostToolUse hook) ─┐
session start ──────┤                                                 ├──► /branch-manager
(SessionStart hook) └─ manual / /loop ────────────────────────────────┘    (sweep all branches)
                                                                                  │
                                  git for-each-ref + gh pr list → facts per branch
                                                                                  │
                                                      branch-auditor subagent (one verdict/branch)
                                                                                  │
                                  ┌───────────────────────────┬───────────────────────────┐
                                  ▼                           ▼                           ▼
                            delete-recommended           attention                    healthy
                            (merged PR, branch        (stale / long-lived /         (recent,
                             still on remote)          diverged / bad name /         compliant,
                                  │                     dependabot pileup)            PR moving)
                                  └─────────────┬─────────────┘                         │
                                                ▼                                    no action
                                     /best-way research pass
                              (confirm / refine each recommendation)
                                                │
                                                ▼
                                  report under tasks/branch-reports/<ts>-audit.md
                                                │
                                                ▼
                                  you review, run the printed commands
```

## File map

| File | Purpose | Edit when… |
|------|---------|------------|
| `.claude/hooks/branch-manager-sweep.sh` | **SessionStart** hook. On a fresh session (startup/resume/clear, not compact), nudges `/branch-manager`. Throttled to once per **6 h**. | You want to change the throttle or the session sources. |
| `.claude/hooks/post-merge-branch-hook.sh` | **PostToolUse** hook on Bash. Nudges `/branch-manager` only after a successful `gh pr merge` or a push to `main` — the moments a branch becomes deletable. Ordinary feature pushes are left to `/ci-watch`. | You want to change which commands trigger the audit. |
| `.claude/commands/branch-manager.md` | The `/branch-manager` coordinator — enumerates branches, gathers facts, classifies, runs `/best-way`, reports. | You want to change thresholds, scope, or report format. |
| `.claude/agents/branch-auditor.md` | The classifier subagent — categorizes one branch, proposes an action; adversarial (defaults to least-destructive). | You want to tune thresholds, the merge-detection rule, or the verdict format. |
| `.claude/settings.local.json` | Wires the SessionStart + PostToolUse hooks (alongside the CI + review hooks). | Only to disable: remove the relevant hook entry. |
| `.claude/state/branch-manager.json` | Branch-keyed ledger (auto-managed). One entry per branch, so unchanged branches aren't re-nagged. | Never edit manually. |
| `tasks/branch-reports/*.md` | One snapshot report per actionable sweep — categories, reasons, and the commands for you to run. | Read when surfaced. |

## Triggers (Layer 1)

- **Session start** (`branch-manager-sweep.sh`): every fresh session nudges one sweep,
  throttled to once per 6 h. Catches branches that aged or merged while you were away.
- **After a merge** (`post-merge-branch-hook.sh`): fires after a successful `gh pr merge`
  or a push to `main` — exactly when a branch becomes deletable. Deliberately **not**
  fired on ordinary `feat/...` pushes (that's the CI watcher's lane; firing on every push
  would double-nudge).
- **Manual**: type `/branch-manager` any time.
- **Continuous**: `/loop 6h /branch-manager` during a long-running session.

## Scope

Audits **all** remote branches except `main`/`HEAD` (which are sanity-checked only).
`dependabot/*` branches are included but exempt from the naming rule (bot-named by
design); their pileup vs. the `.github/dependabot.yml` caps (5 pip / 5 npm / 3 actions)
is reported as `attention`.

## The three categories (fix policy)

The `branch-auditor` picks exactly one per branch. **Everything is report-only** — even
`delete-recommended` only ever prints a command for you.

### delete-recommended — work already in `main`, branch still present
Proof required: a PR with this `headRefName` is `MERGED` **or** git ancestry confirms it,
**and** the remote branch still exists, **and** no open PR / release tag references it.
Proposed command: `git push origin --delete <branch>` (you run it).

### attention — needs your judgment
Stale (≥ 14 days no commits), long-lived (> 7 days, GitHub Flow limit), diverged/behind
`main` (≥ 30 commits), bad name (not `feat|fix|chore|docs|refactor/`), stale open PR
(≥ 7 days untouched/draft), or Dependabot pileup past cap.

### healthy — recent, compliant, PR moving
No action.

## Merge detection — the one non-obvious correctness point

This repo **squash-merges** PRs. A squash merge puts a *new* commit on `main` that is not
an ancestor of the feature branch, so `git branch --merged main` and
`git merge-base --is-ancestor` **miss squash-merged branches** and wrongly call them
unmerged. The reliable signal is the **PR state** (`MERGED`), not git ancestry. The
coordinator and auditor both prefer PR state; ancestry is only a fallback for branches
that never had a PR.

## Research-before-recommend (`/best-way`)

When a sweep produces anything actionable, the coordinator runs `/best-way` **once per
sweep** (not per branch) to confirm/refine the recommendations against current best
practice — e.g. demoting a delete to "leave (still referenced)" or upgrading to
"rebase-then-delete". The report records the reasoning. All-healthy sweeps skip it.

## Layer 2 — cloud Routine for when you're away

Unlike the CI watcher (which needs a `workflow_run` → API-trigger bridge because CI
failures are event-driven), branch hygiene is **schedule-driven**, so Layer 2 uses a
Routine's **native cron trigger** — no `.github/workflows` change, no repo secrets.

```
cron (e.g. weekly Mon) → cloud Routine: audit all branches → open/update ONE tracking issue
```

### One-time setup runbook

1. **Install the Claude GitHub App** on `Jimmy6929/Molebie_AI` (if not already done for
   the CI watcher): https://claude.ai/code/onboarding?magic=github-app-setup
   (Gives the Routine clone + issue access.)
2. **Create the Routine** at https://claude.ai/code/routines → **New routine** → **Remote**:
   - Name: `Molebie branch auditor`
   - Repository: `Jimmy6929/Molebie_AI`
   - Environment: `Default` (Trusted network — GitHub/`gh` reachable)
   - Model: `claude-sonnet-4-6`
   - **Trigger: cron** — e.g. weekly Mon 07:00 UTC (after Dependabot's 06:00 run), or daily.
   - Prompt: paste the **Routine prompt** below.
3. **Test**: routine page → **Run now**, then confirm a single "Branch audit" issue
   appears with recommendations and that **no branch was deleted**.

### Routine prompt (paste into the form)

```
You are the Molebie AI branch auditor, a cloud routine that runs on a schedule to keep
the repo's branches healthy under GitHub Flow (atomic short-lived branches; names
feat|fix|chore|docs|refactor/; squash-merge to main; auto-delete-on-merge; Dependabot
weekly). The repo SQUASH-MERGES, so detect "merged" via PR state == MERGED, not git
ancestry.

Do this:
1. Read .claude/agents/branch-auditor.md in the cloned repo and apply its rubric.
2. Enumerate branches: `git fetch --prune`, `git for-each-ref refs/remotes/origin`, and
   `gh pr list --state all --json number,headRefName,state,isDraft,createdAt,updatedAt,mergedAt`.
3. Classify each branch (except main/HEAD): delete-recommended | attention | healthy.
4. REPORT-ONLY. Do NOT run git branch -d/-D, git push --delete, gh pr merge, gh pr close,
   or git rebase. Never delete or merge anything.
5. Open or UPDATE a single tracking issue titled "Branch audit: <date>" listing the
   delete-recommended and attention branches with the exact commands the user should run
   (e.g. `git push origin --delete feat/x`). If an open "Branch audit" issue already
   exists, comment on / edit it instead of opening a duplicate.
6. If gh/git data is unavailable, open an issue summarizing what you saw and stop.
```

### Trade-offs
Runs in Anthropic cloud (clones the repo), uses Claude usage, research-preview feature,
subject to per-account caps. Overlaps Layer 1 by design — Layer 1's ledger dedups
locally; the Routine covers the "no session open" case. If Layer 1 is enough, just don't
create the Routine.

## Bypassing or disabling

- **Disable the session-start sweep**: remove the `branch-manager-sweep.sh` entry from the
  `SessionStart` block in `.claude/settings.local.json` (or delete the script).
  `/branch-manager` still works manually.
- **Disable the post-merge nudge**: remove the `post-merge-branch-hook.sh` entry from the
  `PostToolUse` block.
- The manager only audits; it never gates a push or a merge.

## Relationship to the CI watcher and review loop

| Layer | Catches | When |
|---|---|---|
| `/review` (pre-push, Stop hook) | local lint, scoped tests, 3-reviewer rubric | before you ship |
| `/ci-watch` (SessionStart + post-push) | red CI across all branches | after you push / while you work |
| `/branch-manager` (SessionStart + post-merge) | merged/stale/diverged/misnamed branches, Dependabot pileup | after merges / while you work |
| Routines (cloud) | the same, for when no session is open | always-on |

## Caveats

- **Layer 1 only runs while Claude Code is open.** For laptop-closed coverage use Layer 2.
- **`gh` CLI must be authed** for reliable merge detection (PR state). Without it the
  manager falls back to git ancestry, which misses squash-merges — confidence drops.
- **Report-only, always.** Even on `delete-recommended`, it only prints
  `git push origin --delete ...`; you review and run it.
- **`main` is never a delete/attention target** — only sanity-checked.
