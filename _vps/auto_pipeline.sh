#!/usr/bin/env bash
# Runs every 6h on the VPS via systemd timer.
# Keeps the SFT corpus freshly built so daily reports + evals always see current state.

set -uo pipefail
cd /home/bot/esports

LOG=/home/bot/esports/_logs/auto_pipeline.log
mkdir -p /home/bot/esports/_logs

echo "[$(date -u +%FT%TZ)] auto_pipeline start" >> "$LOG"

# 1. Git pull (optional — only if we're behind)
git fetch origin main >> "$LOG" 2>&1
LOCAL=$(git rev-parse HEAD)
REMOTE=$(git rev-parse origin/main)
if [ "$LOCAL" != "$REMOTE" ]; then
    echo "[$(date -u +%FT%TZ)]   git pulling $LOCAL -> $REMOTE" >> "$LOG"
    git reset --hard origin/main >> "$LOG" 2>&1
fi

# 2. process-all → build-sft (no-op if nothing changed)
source venv/bin/activate
python data/processor.py process-all 2>&1 | tail -3 >> "$LOG"
python data/processor.py build-sft 2>&1 | tail -6 >> "$LOG"

# 3. Export counts for downstream jobs
SFT_COUNT=$(wc -l < data/training/sft.jsonl 2>/dev/null || echo 0)
FILES_COUNT=$(ls data/recordings/*.jsonl 2>/dev/null | wc -l)
echo "[$(date -u +%FT%TZ)]   done — $SFT_COUNT sft examples, $FILES_COUNT raw files" >> "$LOG"
