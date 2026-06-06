#!/usr/bin/env bash
# Stop hook for Molebie AI's self-review loop.
# Reads stdin (JSON event from Claude Code), decides whether to allow Claude
# to finish the turn or to block-and-inject a directive to run /review.
#
# Decision rules:
#   1. If no code files changed this session -> allow stop.
#   2. If user's last message contained [skip-review] -> allow stop.
#   3. If .claude/state/last-review.json exists, its diff_sha matches the
#      CURRENT diff, and verdict is PASS -> allow stop.
#   4. If attempts >= 2 in .claude/state/review-attempts.json -> allow stop
#      (escalation already happened; do not loop forever).
#   5. Otherwise -> block stop, inject directive: "Run /review now."
#
# Hook contract (Claude Code Stop hook):
#   stdin:  JSON with at minimum {session_id, transcript_path, ...}
#   stdout: JSON {"decision": "block" | "approve", "reason": "..."} OR
#           empty/exit 0 to allow stop by default.
#   Non-zero exit = error; Claude Code treats as allow-stop with a warning.

set -uo pipefail

# Resolve repo root portably (independent of where the hook is invoked from).
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
[ -n "$REPO_ROOT" ] || REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATE_DIR="$REPO_ROOT/.claude/state"
LAST_REVIEW="$STATE_DIR/last-review.json"
ATTEMPTS_FILE="$STATE_DIR/review-attempts.json"

mkdir -p "$STATE_DIR"

# Consume stdin (the hook event) so the caller doesn't see a SIGPIPE.
HOOK_INPUT="$(cat 2>/dev/null || echo '{}')"

allow_stop() {
  # Empty output + exit 0 = allow stop (Claude Code default).
  exit 0
}

block_stop() {
  local reason="$1"
  # Claude Code Stop-hook JSON contract: {"decision":"block","reason":"..."}
  # The "reason" is injected back into Claude's context so it knows what to do.
  printf '{"decision":"block","reason":%s}\n' "$(printf '%s' "$reason" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')"
  exit 0
}

# ---- Rule 1: any code files changed in working tree? ----
cd "$REPO_ROOT" 2>/dev/null || allow_stop

CHANGED_CODE="$(git -C "$REPO_ROOT" status --porcelain 2>/dev/null \
  | awk '{print $2}' \
  | grep -E '^(gateway/|webapp/|cli/).*\.(py|ts|tsx|js|jsx)$' \
  | head -5)"

if [ -z "$CHANGED_CODE" ]; then
  # Nothing reviewable changed -> let Claude finish.
  allow_stop
fi

# ---- Rule 2: [skip-review] sentinel in the last user message ----
# Read the transcript path from the hook event and grep the last user turn.
TRANSCRIPT_PATH="$(printf '%s' "$HOOK_INPUT" | python3 -c 'import json,sys
try:
  d = json.loads(sys.stdin.read() or "{}")
  print(d.get("transcript_path",""))
except Exception:
  print("")' 2>/dev/null)"

if [ -n "$TRANSCRIPT_PATH" ] && [ -f "$TRANSCRIPT_PATH" ]; then
  LAST_USER="$(tail -200 "$TRANSCRIPT_PATH" 2>/dev/null \
    | grep -E '"role":"user"' \
    | tail -1)"
  if echo "$LAST_USER" | grep -qE '\[skip-review\]'; then
    allow_stop
  fi
fi

# ---- Compute current diff SHA so we can tell if a prior PASS is stale ----
CURRENT_DIFF_SHA="$(git -C "$REPO_ROOT" diff HEAD 2>/dev/null | shasum -a 256 | awk '{print $1}')"

# ---- Rule 3: prior PASS for THIS exact diff -> allow ----
if [ -f "$LAST_REVIEW" ]; then
  LAST_VERDICT="$(python3 -c 'import json,sys
try:
  d = json.load(open(sys.argv[1]))
  print(d.get("verdict",""))
except Exception:
  print("")' "$LAST_REVIEW" 2>/dev/null)"

  LAST_SHA="$(python3 -c 'import json,sys
try:
  d = json.load(open(sys.argv[1]))
  print(d.get("diff_sha",""))
except Exception:
  print("")' "$LAST_REVIEW" 2>/dev/null)"

  if [ "$LAST_VERDICT" = "PASS" ] && [ "$LAST_SHA" = "$CURRENT_DIFF_SHA" ]; then
    allow_stop
  fi
fi

# ---- Rule 4: already escalated (attempts >= 2) -> allow stop, do not loop ----
if [ -f "$ATTEMPTS_FILE" ]; then
  ATTEMPTS="$(python3 -c 'import json,sys
try:
  d = json.load(open(sys.argv[1]))
  print(int(d.get("attempts",0)))
except Exception:
  print(0)' "$ATTEMPTS_FILE" 2>/dev/null)"

  if [ "${ATTEMPTS:-0}" -ge 2 ]; then
    # Escalation already happened. Let Claude finish so the user sees the report.
    allow_stop
  fi
fi

# ---- Rule 5: code changed, no PASS on file, not escalated -> block & direct ----
REASON="REVIEW REQUIRED before finishing.

You have uncommitted code changes that have not passed the self-review loop. You must:

1. Run the /review slash command now. It will:
   - Invoke 3 specialized reviewers (Project Goal, Task Goal, Senior Engineer) in parallel
   - Run hard signals (ruff, scoped pytest, tsc if webapp touched)
   - Aggregate the verdict and write .claude/state/last-review.json

2. If /review returns PASS, you may finish the turn.

3. If /review returns FAIL, address every item under 'Required changes' and re-run /review. After 2 consecutive FAILs the loop will escalate to the user automatically.

4. To bypass this loop for trivial work (typo fixes, doc tweaks), include the sentinel [skip-review] in the user's request.

Do not attempt to finish the turn without doing this first."

block_stop "$REASON"
