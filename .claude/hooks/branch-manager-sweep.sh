#!/usr/bin/env bash
# SessionStart hook. Nudges Claude to run a repo-wide /branch-manager sweep when a new
# session opens, so branches that went stale or got merged while you were away (merged
# PRs whose branch wasn't auto-deleted, long-lived branches aging past the GitHub Flow
# limit, Dependabot pileup) get surfaced.
#
# This is the branch-hygiene sibling of session-start-sweep.sh (the CI watcher). Same
# hook-contract plumbing; different command, different state file, much longer throttle
# (branches change far slower than CI runs).
#
# Behaviour:
#   - Fires only on a genuinely fresh session: source in {startup, resume, clear}.
#     Skips source=compact (mid-session; we don't want to re-nudge).
#   - Throttled: if .claude/state/branch-manager.json shows a sweep in the last 6 hours,
#     stay quiet.
#   - Only nudges; never polls or mutates anything. Claude runs /branch-manager.
#
# Hook contract (Claude Code SessionStart):
#   stdin:  JSON with {session_id, source, ...}
#   stdout: JSON {"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"..."}}
#           OR empty to no-op.
#   Exit 0 = ok.

set -uo pipefail

# Resolve repo root portably (independent of where the hook is invoked from).
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_FILE="$REPO_ROOT/.claude/state/branch-manager.json"
THROTTLE_SECONDS=21600  # 6 hours

# Consume stdin so the caller doesn't see SIGPIPE.
HOOK_INPUT="$(cat 2>/dev/null || echo '{}')"

# Only nudge on a fresh session, not on compaction.
SOURCE="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  print(d.get("source", ""))
except Exception:
  print("")
' 2>/dev/null)"

case "$SOURCE" in
  startup|resume|clear) ;;       # proceed
  *) exit 0 ;;                    # compact or unknown -> stay quiet
esac

# Throttle: skip if we swept recently.
if [ -f "$STATE_FILE" ]; then
  RECENT="$(python3 -c '
import json, sys, datetime
try:
  d = json.load(open(sys.argv[1]))
  ts = d.get("last_sweep_at")
  if not ts:
    print("0"); sys.exit(0)
  ts = ts.replace("Z", "+00:00")
  last = datetime.datetime.fromisoformat(ts)
  if last.tzinfo is None:
    last = last.replace(tzinfo=datetime.timezone.utc)
  age = (datetime.datetime.now(datetime.timezone.utc) - last).total_seconds()
  print("1" if age < '"$THROTTLE_SECONDS"' else "0")
except Exception:
  print("0")
' "$STATE_FILE" 2>/dev/null)"
  if [ "${RECENT:-0}" = "1" ]; then
    exit 0
  fi
fi

DIRECTIVE="New Claude Code session started.

Branch manager: at a convenient point this session, run the /branch-manager slash command to audit branch health across the repo. It will:
  1. Enumerate every branch (git for-each-ref + gh pr list) and gather facts (age, merged-PR status, ahead/behind main, naming, open PRs).
  2. Classify each branch via the branch-auditor subagent: delete-recommended | attention | healthy.
  3. If anything is actionable, run /best-way once to ground the recommendations in current best practice.
  4. Tell you what to add/delete/do with the exact commands, and write a report under tasks/branch-reports/.

This catches branches that went stale or got merged while you were away. It is report-only — you (Claude) never run git branch -d, git push --delete, or gh pr merge.

If the user's first request is urgent and unrelated, do that first and interleave /branch-manager — but don't skip it for the session."

printf '%s' "$DIRECTIVE" | python3 -c '
import json, sys
ctx = sys.stdin.read()
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
'
exit 0
