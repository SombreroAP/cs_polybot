#!/usr/bin/env bash
# nightly_shadow_replay.sh — daily check on whether a local 5090 model agrees
# with production Haiku, replayed against the same prompts.
#
# Cron: 30 2 * * * (02:30 local each day)

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=_logs/shadow_replay.log
mkdir -p _logs
exec >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === nightly shadow replay ==="

set -a; source .env; set +a

# Use python3.12 (system pip has anthropic + dotenv + requests installed).
PY="${PY:-python3.12}"

# Replay the last 24 h through both clean-JSON local candidates, in series.
# (Sequential is fine — ollama is single-GPU; parallel calls just queue.)
$PY shadow_replay.py --model ollama://ministral-3:14b --since-hours 24 --telegram
$PY shadow_replay.py --model ollama://mistral-small3.2:24b --since-hours 24 --telegram

echo "[+] done"
