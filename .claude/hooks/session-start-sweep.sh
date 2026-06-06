#!/usr/bin/env bash
# SessionStart hook. Nudges Claude to run a repo-wide /ci-watch sweep when a new
# session opens, so CI failures that landed while you were away (merges to main,
# scheduled CodeQL/Secret Scan, Dependabot, release tags, pushes from elsewhere)
# get picked up.
#
# This is the "while I'm working" half of the watcher: the PostToolUse hook
# (post-push-hook.sh) catches your own pushes; this catches everything else, once,
# at the start of a session.
#
# Behaviour:
#   - Fires only on a genuinely fresh session: source in {startup, resume, clear}.
#     Skips source=compact (that fires mid-session; we don't want to re-nudge).
#   - Throttled: if .claude/state/ci-watch.json shows a sweep in the last 30 min,
#     stay quiet (avoids re-nudging on quick restarts / resumes).
#   - Only nudges; never polls or mutates anything. Claude runs /ci-watch.
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
STATE_FILE="$REPO_ROOT/.claude/state/ci-watch.json"
THROTTLE_SECONDS=1800  # 30 min

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

CI watcher: early in this session, run the /ci-watch slash command to sweep CI across ALL branches + main. It will:
  1. Enumerate every workflow run repo-wide (gh run list) and reduce to the latest run per (workflow, branch).
  2. Diagnose any branch/main that is currently red and not already handled, via the ci-diagnoser subagent.
  3. Auto-apply only mechanical fixes (lint/format, unreachable-CVE ignore) on the checked-out branch; propose-only for ambiguous or cross-branch failures; hard-stop for security findings.
  4. Write incident reports under tasks/ci-incidents/ and summarize in chat.

This catches failures that landed while you were away (merges to main, scheduled CodeQL/Secret Scan, Dependabot, release tags). You (Claude) never run git add / commit / push.

If the user's first request is urgent and unrelated, you may do that first and interleave /ci-watch — but don't skip it for the session."

printf '%s' "$DIRECTIVE" | python3 -c '
import json, sys
ctx = sys.stdin.read()
print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart", "additionalContext": ctx}}))
'
exit 0
