#!/bin/bash
# auto-pull.sh — Polls GitHub and pulls new commits on main.
# Designed to run every 60s via launchd on the MacBook Pro 2016 home server.
set -euo pipefail

REPO_DIR="/Users/jimmy/Documents/App-project/Local_AI_Project"
LOG_DIR="$REPO_DIR/logs"
LOG_FILE="$LOG_DIR/auto-pull.log"
BRANCH="main"

mkdir -p "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >> "$LOG_FILE"; }

cd "$REPO_DIR"

# Skip if there are local uncommitted changes
if [ -n "$(git status --porcelain)" ]; then
    log "WARN: uncommitted changes detected, skipping pull"
    exit 0
fi

# Fetch latest from origin
git fetch origin "$BRANCH" 2>>"$LOG_FILE"

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse "origin/$BRANCH")

# Nothing new — exit silently
if [ "$LOCAL" = "$REMOTE" ]; then
    exit 0
fi

log "New commits found: $LOCAL -> $REMOTE"
git pull --ff-only origin "$BRANCH" >> "$LOG_FILE" 2>&1

# Determine what changed
CHANGED=$(git diff --name-only "$LOCAL" "$REMOTE")
log "Changed files:\n$CHANGED"

# Install gateway dependencies if requirements changed
if echo "$CHANGED" | grep -q "^gateway/requirements.txt"; then
    log "Installing gateway dependencies..."
    cd "$REPO_DIR/gateway" && pip3 install -r requirements.txt >> "$LOG_FILE" 2>&1
fi

# Install webapp dependencies if package.json changed
if echo "$CHANGED" | grep -q "^webapp/package.json"; then
    log "Installing webapp dependencies..."
    cd "$REPO_DIR/webapp" && npm install >> "$LOG_FILE" 2>&1
fi

# Gateway: uvicorn --reload handles file changes automatically

# Webapp: rebuild only if running in production mode (next start)
if echo "$CHANGED" | grep -q "^webapp/"; then
    if pgrep -f "next start" > /dev/null 2>&1; then
        log "Rebuilding webapp (production mode detected)..."
        cd "$REPO_DIR/webapp" && npm run build >> "$LOG_FILE" 2>&1
        pkill -f "next start" || true
        log "Webapp process killed — launchd or startup script should restart it"
    else
        log "Webapp in dev mode — Next.js will hot-reload automatically"
    fi
fi

log "Pull complete"
