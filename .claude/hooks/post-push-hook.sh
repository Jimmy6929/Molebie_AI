#!/usr/bin/env bash
# PostToolUse hook on Bash. Detects `git push` commands and nudges Claude to
# start the /ci-watch coordinator in the next turn.
#
# This hook itself does not poll CI — that would tie up the hook script and
# can't apply file edits anyway. It just injects a directive into Claude's
# context. Claude (in the conversation thread) then runs /ci-watch which can
# poll, invoke subagents, and apply fixes.
#
# Hook contract (Claude Code PostToolUse):
#   stdin:  JSON event with {tool_name, tool_input, tool_response, ...}
#   stdout: optional JSON {"continue": true, "additionalContext": "..."} OR
#           empty to no-op.
#   Exit 0 = ok; non-zero treated as error by Claude Code.

set -uo pipefail

# Resolve repo root portably (independent of where the hook is invoked from).
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Consume stdin so the caller doesn't see SIGPIPE.
HOOK_INPUT="$(cat 2>/dev/null || echo '{}')"

# Parse the event. We care about: tool_name == "Bash" with a command containing
# `git push`. Anything else: no-op.
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

# Cheap substring check first — `git push` somewhere in the command.
# Exclude --dry-run (no remote mutation, no CI fires) and `git pull` (false friend).
if ! echo "$COMMAND" | grep -Eq '(^|[^a-z])git[^a-z]+push([^a-z]|$)'; then
  exit 0
fi

if echo "$COMMAND" | grep -Eq -- '--dry-run'; then
  exit 0
fi

# Confirm the push actually succeeded by checking the tool response exit code if present.
TOOL_EXIT="$(printf '%s' "$HOOK_INPUT" | python3 -c '
import json, sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  resp = d.get("tool_response", {}) or {}
  # tool_response may carry "stdout"/"stderr"/"exit_code" depending on event shape
  rc = resp.get("exit_code", resp.get("returncode", 0))
  print(int(rc))
except Exception:
  print(0)
' 2>/dev/null)"

if [ "${TOOL_EXIT:-0}" != "0" ]; then
  # Push failed — no CI to watch.
  exit 0
fi

# Extract the current branch for the directive.
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

# Build the directive Claude will see as additionalContext on its next turn.
DIRECTIVE="git push detected on branch '${BRANCH}'.

CI watcher: please run the /ci-watch slash command now to monitor the post-push CI checks for this branch. It will:
  1. Poll all CI checks until they complete.
  2. On any failure, invoke the ci-diagnoser subagent to classify and propose a fix.
  3. Auto-apply fixes only for mechanical/safe categories (lint, dep CVE ignore for unreachable vulns). Propose-only for ambiguous failures. Hard-stop for security findings.
  4. Surface results in chat and write an incident report under tasks/ci-incidents/.

You (Claude) never run git add / commit / push — the user owns those after reviewing any applied fix.

If the user's request already specifies waiting for CI or merging the PR, you can interleave /ci-watch with that work. Otherwise, run /ci-watch now."

# Emit JSON for Claude Code's hook contract.
printf '%s' "$DIRECTIVE" | python3 -c '
import json, sys
ctx = sys.stdin.read()
print(json.dumps({"continue": True, "additionalContext": ctx}))
'
exit 0
