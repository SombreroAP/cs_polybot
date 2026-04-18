#!/usr/bin/env bash
# deploy.sh — pull latest code on the VPS + restart the bot cleanly.
#
# Two ways to use this:
#
#   1. From your Mac (ships edits in one command):
#        git push && ssh bot@<vps> "cd ~/esports && ./deploy.sh"
#
#   2. On the VPS directly (after pulling manually):
#        cd ~/esports && ./deploy.sh
#
# Safe to rerun. Rejects if there are local uncommitted changes on the VPS
# (you should never edit directly on the VPS — commit on Mac, push, deploy).

set -euo pipefail
cd "$(dirname "$0")"

echo "[deploy] $(date -u '+%Y-%m-%d %H:%M:%S UTC') starting..."

# 1. Safety check — no local edits on the VPS
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "[deploy] ERROR: uncommitted local changes on this host:"
    git status --short
    echo "[deploy] refusing to deploy. Either commit + push from Mac first, or revert:"
    echo "         git checkout -- ."
    exit 1
fi

# 2. Pull latest
echo "[deploy] fetching from origin..."
git fetch origin main

LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)

if [ "$LOCAL" = "$REMOTE" ]; then
    echo "[deploy] already up to date at $(git rev-parse --short HEAD)"
else
    echo "[deploy] updating $(git rev-parse --short HEAD) -> $(git rev-parse --short origin/main)"
    echo "[deploy] changelog:"
    git log --oneline "${LOCAL}..${REMOTE}" | sed 's/^/  /'
    git reset --hard origin/main
fi

# 3. Install any new Python deps
if [ -f requirements.txt ] && [ -d venv ]; then
    # Only install if requirements.txt changed in this pull (fast path)
    if git diff --name-only "${LOCAL}..${REMOTE}" 2>/dev/null | grep -q '^requirements.txt$'; then
        echo "[deploy] requirements.txt changed — installing..."
        venv/bin/pip install -q -r requirements.txt
    fi
fi

# 4. Restart the bot via systemd (only works as root; bot user has passwordless
#    sudo for this one unit — see first-run setup below)
echo "[deploy] restarting cs2bot.service..."
if command -v systemctl &>/dev/null; then
    if sudo -n systemctl restart cs2bot.service 2>/dev/null; then
        sleep 3
        systemctl is-active cs2bot.service && echo "[deploy] ✓ cs2bot is active"
        systemctl status cs2bot.service --no-pager --lines=0 | head -3
    else
        echo "[deploy] warn: could not restart cs2bot.service (need: sudo visudo as root,"
        echo "         add:  bot ALL=NOPASSWD: /bin/systemctl restart cs2bot.service, /bin/systemctl status cs2bot.service"
        echo "         Service NOT restarted — do it manually: sudo systemctl restart cs2bot.service"
    fi
fi

echo "[deploy] done."
