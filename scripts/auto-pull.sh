#!/bin/bash
# auto-pull.sh — Polls GitHub and pulls new commits on main.
# Designed to run every 60s via launchd on the MacBook Pro 2016 home server.
# Errors are handled explicitly per-command (no set -euo pipefail).

REPO_DIR="/Users/jimmy/Documents/App-project/Local_AI_Project"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/auto-pull.log"
BRANCH="main"

# Ensure HOME is set (launchd may not provide it)
export HOME="${HOME:-/Users/jimmy}"

# Git HTTP timeouts — fail fast instead of hanging forever
export GIT_HTTP_LOW_SPEED_LIMIT=1000
export GIT_HTTP_LOW_SPEED_TIME=30

# If using SSH remote, try to find the SSH agent socket
REMOTE_URL=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || echo "")
if [[ "$REMOTE_URL" == git@* ]] || [[ "$REMOTE_URL" == ssh://* ]]; then
    if [ -z "$SSH_AUTH_SOCK" ]; then
        SOCK=$(find /tmp/com.apple.launchd.*/Listeners -type s 2>/dev/null | head -1)
        if [ -n "$SOCK" ]; then
            export SSH_AUTH_SOCK="$SOCK"
        fi
    fi
fi

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

# Log rotation: truncate to last 5000 lines when exceeding 10000
if [ -f "$LOG_FILE" ]; then
    LINE_COUNT=$(wc -l < "$LOG_FILE" | tr -d ' ')
    if [ "$LINE_COUNT" -gt 10000 ]; then
        tail -5000 "$LOG_FILE" > "$LOG_FILE.tmp" && mv "$LOG_FILE.tmp" "$LOG_FILE"
        log "Log rotated (was $LINE_COUNT lines, kept last 5000)"
    fi
fi

# Heartbeat — proves the service is alive
log "HEARTBEAT: auto-pull check starting"

cd "$REPO_DIR" || { log "ERROR: cannot cd to $REPO_DIR"; exit 1; }

# Skip if there are local uncommitted changes
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    log "WARN: uncommitted changes detected, skipping pull"
    exit 0
fi

# Fetch latest from origin (capture errors explicitly)
FETCH_OUTPUT=$(git fetch origin "$BRANCH" 2>&1) || {
    log "ERROR: git fetch failed: $FETCH_OUTPUT"
    exit 1
}

LOCAL=$(git rev-parse HEAD 2>/dev/null)
REMOTE=$(git rev-parse "origin/$BRANCH" 2>/dev/null)

if [ -z "$LOCAL" ] || [ -z "$REMOTE" ]; then
    log "ERROR: could not resolve HEAD or origin/$BRANCH"
    exit 1
fi

# Nothing new — exit quietly (heartbeat already logged)
if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0
fi

log "New commits found: $LOCAL -> $REMOTE"

PULL_OUTPUT=$(git pull --ff-only origin "$BRANCH" 2>&1) || {
    log "ERROR: git pull failed: $PULL_OUTPUT"
    exit 1
}
log "$PULL_OUTPUT"

# Determine what changed
CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE")
CHANGED_ONELINE=$(echo "$CHANGED" | tr '\n' ' ')
log "Changed files: $CHANGED_ONELINE"

# Install gateway dependencies if requirements changed (subshell to preserve cwd)
if echo "$CHANGED" | grep -q "^gateway/requirements.txt"; then
    log "Installing gateway dependencies..."
    (cd "$REPO_DIR/gateway" && pip3 install -r requirements.txt >> "$LOG_FILE" 2>&1) || {
        log "WARN: gateway pip install failed"
    }
fi

# Install webapp dependencies if package.json changed (subshell to preserve cwd)
if echo "$CHANGED" | grep -q "^webapp/package.json"; then
    log "Installing webapp dependencies..."
    (cd "$REPO_DIR/webapp" && npm install >> "$LOG_FILE" 2>&1) || {
        log "WARN: webapp npm install failed"
    }
fi

# Gateway: uvicorn --reload handles file changes automatically

# Webapp: rebuild only if running in production mode (next start)
if echo "$CHANGED" | grep -q "^webapp/"; then
    if pgrep -f "next start" > /dev/null 2>&1; then
        log "Rebuilding webapp (production mode detected)..."
        (cd "$REPO_DIR/webapp" && npm run build >> "$LOG_FILE" 2>&1) || {
            log "WARN: webapp build failed"
        }
        pkill -f "next start" || true
        log "Webapp process killed — launchd or startup script should restart it"
    else
        log "Webapp in dev mode — Next.js will hot-reload automatically"
    fi
fi

log "Pull complete"
