---
description: Audit branch health across the repo. Sweeps every branch, classifies each via the branch-auditor subagent (delete-recommended | attention | healthy), runs a /best-way research pass to ground any recommendation, then tells you what to add/delete/do with the exact commands. Report-only — never deletes, merges, or pushes.
---

# /branch-manager — Branch Hygiene Coordinator (repo-wide)

You are the coordinator for the branch-manager loop. This is the branch-hygiene
counterpart to the `/ci-watch` CI loop — same architecture, different domain.

**Hard rule before you start anything**: you NEVER run `git branch -d/-D`,
`git push origin --delete`, `git push`, `gh pr merge`, `gh pr close`, `git rebase`,
or any other command that mutates a branch, ref, or remote. You **report and propose**.
For every recommendation you print the exact command for the **user** to run; they
decide. The only git/gh commands you may run are read-only inspection (`fetch --prune`,
`for-each-ref`, `rev-list`, `merge-base --is-ancestor`, `log`, `gh pr list/view`).

The repo uses GitHub Flow (see `CLAUDE.md`): atomic short-lived branches, names
`feat|fix|chore|docs|refactor/`, squash-merge to `main`, auto-delete-on-merge enabled,
Dependabot weekly. Your thresholds encode that policy.

---

## Step 1 — Enumerate branches (repo-wide)

```bash
REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"
git fetch --prune --quiet                       # read-only on local refs; prunes stale remote-tracking only

# Remote branches: name, head short-sha, last-commit ISO date, committer date (epoch)
git for-each-ref --format='%(refname:short)%09%(objectname:short)%09%(committerdate:iso8601)%09%(committerdate:unix)' refs/remotes/origin

# PR cross-reference (open + merged + closed)
gh pr list --state all --limit 100 \
  --json number,headRefName,state,isDraft,createdAt,updatedAt,mergedAt,title,author,isCrossRepository
```

`git for-each-ref` lists every remote branch. If `gh pr list` errors, note *"PR data
unavailable — is `gh` authed (`gh auth status`)? Falling back to git-only merge
detection."* and continue with reduced confidence (git ancestry only).

Drop `origin/HEAD` and `origin/main` from the candidate set (main is never a
delete/attention target — sanity-check it only). Treat `dependabot/*` as exempt from the
naming rule.

---

## Step 2 — Gather facts per branch (no judgment yet)

For each candidate branch, compute the plain facts the auditor needs:

```bash
# Age in days from committer date; ahead/behind main:
git rev-list --left-right --count origin/main...origin/<branch>   # -> "<behind>\t<ahead>"
# Squash-merge-safe ancestry fallback (only meaningful for no-PR branches):
git merge-base --is-ancestor origin/<branch> origin/main && echo merged-by-ancestry
```

Use a small python block on the Step-1 JSON to assemble, per branch:
`{name, head_sha, age_days, behind, ahead, naming_ok, is_dependabot,
pr_number, pr_state, pr_is_draft, pr_age_days}`.

**Merge detection (the one non-obvious correctness point):** this repo **squash-merges**,
so `git branch --merged` / `merge-base --is-ancestor` MISS squash-merged branches. The
**primary** merged signal is a PR with `state == MERGED` and matching `headRefName`. Only
use ancestry for branches that never had a PR.

Also tally Dependabot **open**-PR counts per ecosystem (pip / npm / github-actions) from
the PR list, to detect pileup vs. the caps in `.github/dependabot.yml` (5/5/3).

---

## Step 3 — Load the ledger & compute what's new

Read `.claude/state/branch-manager.json` (branch-keyed). Schema:

```json
{
  "last_sweep_at": "<ISO ts>",
  "branches": {
    "feat/x": {
      "head_sha": "abc1234",
      "category": "delete-recommended",
      "reasons": ["merged via PR #65", "remote branch still present"],
      "pr": 65,
      "handled_at": "<ISO ts>",
      "report_file": "tasks/branch-reports/<file>.md"
    }
  }
}
```

A branch needs (re)auditing if **any** of:
- its name is absent from `branches`, **OR**
- its stored `head_sha` differs from the current head (new commits landed), **OR**
- enough time passed that an age-based category could have flipped (a `healthy` branch
  can become `attention` purely by aging past 7/14 days with no new commits) — so always
  re-evaluate age thresholds even when the sha is unchanged.

Branches whose sha AND category are unchanged since last sweep stay quiet — don't re-nag.

**Cold-start guard**: if the ledger doesn't exist yet, baseline every branch (record it)
but still surface anything that is `delete-recommended` or `attention` *right now*. Don't
invent history; just report the current state.

---

## Step 4 — Classify via the branch-auditor subagent

For each branch needing attention, invoke the `branch-auditor` subagent with the gathered
facts (Step 2). It returns one verdict:

```
BRANCH: <name>
CATEGORY: delete-recommended | attention | healthy
REASONS:
- <facts>
RECOMMENDED ACTION: <delete | rebase-then-delete | leave | rename | open-PR | close-PR | none>
PROPOSED COMMAND: <exact command, or "none">
CONFIDENCE: high | medium | low
```

`main` and the currently checked-out branch are never `delete-recommended`. If every
audited branch comes back `healthy`, skip to Step 6 (success summary) — no `/best-way`,
no report file.

---

## Step 4.5 — Research-before-recommend (`/best-way`)

If the sweep produced **any** actionable item (`delete-recommended` or `attention`), run
the `/best-way` slash command **once per sweep** (not per branch) before surfacing
anything. Scope it to the concrete decisions on the table, e.g.:

> "Under GitHub Flow with squash-merge and auto-delete-on-merge enabled, for these
> branches — `<merged-but-present>`, `<N-days stale>`, `<M behind main>`,
> `<bad-named>` — what is the current best practice: delete now / rebase then delete /
> leave behind a feature flag / rename? Cite sources."

Use its researched answer to **confirm, refine, or downgrade** each recommendation:
- demote a `delete-recommended` to "leave" if research/evidence shows it's still
  referenced (open PR, release tag, in-flight work),
- upgrade "delete" to "rebase-then-delete" where that's the safer current practice,
- adjust the `RECOMMENDED ACTION` / `PROPOSED COMMAND` accordingly.

Record the best-way reasoning so it lands in the report. One call per sweep keeps
recommendations grounded without burning research on every branch. (All-healthy sweeps
never reach this step.)

---

## Step 5 — Write the report (only if actionable)

If anything is `delete-recommended` or `attention`, write ONE snapshot report:

Path: `tasks/branch-reports/<UTC-ts>-audit.md`.

```markdown
# Branch Audit — <ISO UTC>

**Swept**: <N> branches · **delete-recommended**: <a> · **attention**: <b> · **healthy**: <c>
**Research**: /best-way pass — <one-line gist + any source cited>

## Delete-recommended (work already in main)
| Branch | Why | Recommended action | Command for you to run |
|--------|-----|--------------------|------------------------|
| feat/x | merged via PR #65 (squash) | delete | `git push origin --delete feat/x` |

## Attention (your judgment)
| Branch | Why | Recommended action | Command for you to run |
|--------|-----|--------------------|------------------------|
| fix/y  | 21d stale, 40 behind main | rebase-then-delete or close | `git push origin --delete fix/y` (if abandoned) |

## Healthy
<one line each, or a count if many>

## Notes
- Report-only. I ran zero mutating commands. You own every deletion/merge.
- Best-way reasoning behind each recommendation: <summary>
```

If everything is healthy, write **no file** — just the chat summary (avoids clutter).

---

## Step 6 — Summarize to chat

**Success summary** (all healthy):
```
✓ Branches swept — all healthy across <N> branches.
  main                  ✓ protected
  feat/tiered-storage   ✓ 2d old, compliant, PR #71 moving
Nothing to do. (You own deletions — I don't touch branches.)
```

**Actionable summary** (loud, copy-pasteable):
```
Branch audit — <N> branches: <a> delete-recommended, <b> attention, <c> healthy.
Researched with /best-way: <one-line gist>.

Delete-recommended (work already in main):
  feat/x   merged via PR #65   →  git push origin --delete feat/x

Attention (your call):
  fix/y    21d stale, 40 behind main   →  rebase or close; if abandoned: git push origin --delete fix/y

Report: tasks/branch-reports/<file>.md
You own deletions — I don't touch branches.
```

---

## Step 7 — Persist the ledger

Write `.claude/state/branch-manager.json` with `last_sweep_at` updated and one entry per
**currently existing** branch: `{head_sha, category, reasons, pr, handled_at,
report_file}`. Prune entries for branches that no longer exist on the remote. Never edit
this file by hand outside this command.

---

## Notes

- **Idempotence**: re-running `/branch-manager` reads the ledger and only re-audits
  branches that changed (new sha) or could have aged into a new category. Unchanged
  branches stay quiet.
- **Report-only, always.** Never run `git branch -d/-D`, `git push --delete`, `git push`,
  `gh pr merge`, `gh pr close`, `git rebase`, or `git checkout <other-branch>`. Print
  commands; the user runs them.
- **Squash-merge reality**: trust PR `state == MERGED` over git ancestry for merge
  detection. Ancestry is only a fallback for branches that never had a PR.
- **`main` is never a target** — only sanity-checked (e.g. flag loudly if `main` itself
  is somehow behind `origin/main`).
- **Dependabot branches** are exempt from the naming rule and are bot-managed — report
  pileup (> cap) as `attention`, but the action is "let Dependabot/CI churn them or merge
  the stack", never a manual delete.
- **`/best-way` only runs when there's something actionable** — never on an all-healthy
  sweep.
