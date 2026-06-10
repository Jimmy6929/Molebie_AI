#!/usr/bin/env bash
# PostToolUse hook on Bash. Nudges Claude to run /branch-manager at the moments a branch
# becomes deletable or stale — i.e. right after a PR merge or a push to main. It does NOT
# fire on ordinary feature-branch pushes (those are the CI watcher's job via
# post-push-hook.sh); firing on every push would double-nudge.
#
# Like the CI hooks, this only injects a directive — it does not poll or mutate anything.
# Claude (in the conversation thread) runs /branch-manager, which audits and reports.
#
# Hook contract (Claude Code PostToolUse):
#   stdin:  JSON event with {tool_name, tool_input, tool_response, ...}
#   stdout: optional JSON {"continue": true, "additionalContext": "..."} OR empty to no-op.
#   Exit 0 = ok; non-zero treated as error by Claude Code.

set -uo pipefail

# Resolve repo root portably (independent of where the hook is invoked from).
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Consume stdin so the caller doesn't see SIGPIPE.
HOOK_INPUT="$(cat 2>/dev/null || echo '{}')"

TOOL_NAME="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  print(d.get("tool_name", ""))
except Exception:
  print("")
' 2>/dev/null)"

if [ "$TOOL_NAME" != "Bash" ]; then
  exit 0
fi

COMMAND="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  print(d.get("tool_input", {}).get("command", ""))
except Exception:
  print("")
' 2>/dev/null)"

# Fire only on branch-becomes-deletable moments:
#   - `gh pr merge ...`           (a PR just merged -> its branch is now deletable)
#   - `git push ... main`         (a push landed on main -> branches may now be merged)
# Exclude --dry-run. Ordinary `git push origin feat/x` is intentionally NOT matched.
MATCHED=""
if echo "$COMMAND" | grep -Eq '(^|[^a-z])gh[^a-z]+pr[^a-z]+merge([^a-z]|$)'; then
  MATCHED="pr-merge"
elif echo "$COMMAND" | grep -Eq '(^|[^a-z])git[^a-z]+push([^a-z]|$)' \
   && echo "$COMMAND" | grep -Eq '(^|[^a-z])main([^a-z]|$)'; then
  MATCHED="push-main"
fi

if [ -z "$MATCHED" ]; then
  exit 0
fi

if echo "$COMMAND" | grep -Eq -- '--dry-run'; then
  exit 0
fi

# Confirm the command actually succeeded (exit 0) before nudging.
TOOL_EXIT="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  resp = d.get("tool_response", {}) or {}
  rc = resp.get("exit_code", resp.get("returncode", 0))
  print(int(rc))
except Exception:
  print(0)
' 2>/dev/null)"

if [ "${TOOL_EXIT:-0}" != "0" ]; then
  exit 0
fi

DIRECTIVE="A merge-class git action just succeeded (${MATCHED}).

Branch manager: please run the /branch-manager slash command now to audit branch health. A PR merge or a push to main is exactly when branches become deletable or fall out of sync. It will:
  1. Enumerate branches and gather facts (merged-PR status, age, ahead/behind main, naming).
  2. Classify each via the branch-auditor subagent (delete-recommended | attention | healthy).
  3. If anything is actionable, run /best-way once to ground the recommendations, then tell you what to delete/do with the exact commands and write a report under tasks/branch-reports/.

It is report-only — you (Claude) never run git branch -d, git push --delete, or gh pr merge. The user runs any deletion after reviewing.

If the user's current request is mid-flight, you can interleave /branch-manager — but don't skip it after a merge."

printf '%s' "$DIRECTIVE" | python3 -c '
import json, sys
ctx = sys.stdin.read()
print(json.dumps({"continue": True, "additionalContext": ctx}))
'
exit 0
