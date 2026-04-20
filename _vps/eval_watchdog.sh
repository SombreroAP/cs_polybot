#!/usr/bin/env bash
# Eval watchdog — every 4h, runs evaluator against production model on
# held-out test set. If accuracy drops >5pct from last baseline, Telegram alerts.

set -uo pipefail
cd /home/bot/esports

LOG=/home/bot/esports/_logs/eval_watchdog.log
BASELINE_FILE=/home/bot/esports/_logs/eval_baseline.json

echo "[$(date -u +%FT%TZ)] eval_watchdog start" >> "$LOG"

# Only run if we have enough SFT data
SFT_COUNT=$(wc -l < data/training/sft.jsonl 2>/dev/null || echo 0)
if [ "$SFT_COUNT" -lt 500 ]; then
    echo "[$(date -u +%FT%TZ)]   skip: only $SFT_COUNT examples, need >=500" >> "$LOG"
    exit 0
fi

# Run evaluator
source venv/bin/activate
CURRENT_MODEL=${EDGE_CLAUDE_MODEL:-claude-haiku-4-5-20251001}
RESULT=$(python evaluator.py --model "$CURRENT_MODEL" --samples 80 2>&1 | tail -25)
echo "$RESULT" >> "$LOG"

# Extract accuracy (pattern: "  accuracy                 0.911")
ACC=$(echo "$RESULT" | grep -E "^\s*accuracy" | awk '{print $2}' | head -1)
PNL=$(echo "$RESULT" | grep -E "sim_pnl_pct_on_buys" | awk '{print $2}' | head -1)
echo "[$(date -u +%FT%TZ)]   accuracy=$ACC  sim_pnl=$PNL" >> "$LOG"

# Compare to baseline
DRIFT_MSG=""
if [ -f "$BASELINE_FILE" ]; then
    BASELINE_ACC=$(python -c "import json; print(json.load(open('$BASELINE_FILE')).get('accuracy',0))")
    if python -c "import sys; sys.exit(0 if float('$ACC') < float('$BASELINE_ACC') - 0.05 else 1)"; then
        DRIFT_MSG="WARNING: accuracy dropped: $BASELINE_ACC -> $ACC"
        echo "[$(date -u +%FT%TZ)]   $DRIFT_MSG" >> "$LOG"

        # Send Telegram alert
        python - <<PY 2>>"$LOG"
import json, os, urllib.request, urllib.parse
from pathlib import Path
env = Path("/home/bot/esports/.env").read_text()
for line in env.splitlines():
    if "=" in line and not line.startswith("#"):
        k, v = line.split("=", 1)
        os.environ.setdefault(k, v)
text = """CS2 eval watchdog: ACCURACY DRIFT DETECTED

Baseline: $BASELINE_ACC
Current:  $ACC
Drift:    -$(echo "$BASELINE_ACC - $ACC" | bc)

$DRIFT_MSG

This may indicate:
  1. Dataset distribution shift (new data != old)
  2. Model regression (rare)
  3. Prompt drift
Check latest run: /home/bot/esports/_logs/eval_watchdog.log"""
data = urllib.parse.urlencode({
    "chat_id": os.environ["TELEGRAM_USER_ID"],
    "text": text,
}).encode()
url = f"https://api.telegram.org/bot{os.environ['TELEGRAM_BOT_TOKEN']}/sendMessage"
req = urllib.request.Request(url, data=data)
urllib.request.urlopen(req, timeout=10).read()
PY
    fi
fi

# Save/update baseline
python -c "
import json
json.dump({'accuracy': float('$ACC'), 'sim_pnl': float('$PNL' or 0), 'ts': $(date +%s)},
          open('$BASELINE_FILE', 'w'))
"

echo "[$(date -u +%FT%TZ)]   done" >> "$LOG"
