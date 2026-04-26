#!/usr/bin/env bash
# weekly_audit.sh — runs on the Mac via launchd (every Monday).
#
# 1. Pulls latest training data from the VPS.
# 2. Runs `python data/processor.py audit`.
# 3. If BOTH gates green (TRAIN_READY_VOLUME ✅ + TEMPORAL ✅):
#       scp sft.jsonl to the 5090, launch train_5090.py + eval_5090.py,
#       Telegram the loss + buy-precision + buy-PnL.
#    Otherwise: Telegram a one-line status with the current gap.
#
# Manual run: ./scripts/weekly_audit.sh
# Logs to: _logs/weekly_audit.log

set -euo pipefail
cd "$(dirname "$0")/.."

LOG=_logs/weekly_audit.log
mkdir -p _logs
exec >> "$LOG" 2>&1
echo "[$(date -u '+%Y-%m-%d %H:%M:%S UTC')] === weekly audit start ==="

# Load env (TELEGRAM_BOT_TOKEN + TELEGRAM_USER_ID)
set -a; source .env; set +a

tg() {
  curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
    -d chat_id=${TELEGRAM_USER_ID} \
    --data-urlencode text="$1" > /dev/null
}

# 1. Pull latest VPS recordings. VPS stores them flat in data/recordings/ as
# uncompressed .jsonl (file path discovered 2026-04-26 — earlier rsync path
# had `raw/` and `.jsonl.gz` and silently no-op'd).
echo "[+] syncing VPS recordings..."
rsync -avz --include='*.jsonl' --exclude='*' \
  bot@85.137.174.57:~/esports/data/recordings/ \
  data/recordings/raw/ 2>&1 | tail -5 || true

# 2. Re-process + audit
echo "[+] processing + auditing..."
python3 data/processor.py ingest 2>&1 | tail -5 || true
python3 data/processor.py process-all 2>&1 | tail -5 || true
python3 data/processor.py build-sft 2>&1 | tail -5 || true

AUDIT_OUT=$(python3 data/processor.py audit 2>&1)
echo "$AUDIT_OUT"

VOL_OK=$(echo "$AUDIT_OUT" | grep -c "✅  TRAIN_READY_VOLUME" || true)
TEMPORAL_OK=$(echo "$AUDIT_OUT" | grep -c "✅  TEMPORAL" || true)
MATCHES=$(echo "$AUDIT_OUT" | awk '/unique_matches/ {print $2}')
DAYS=$(echo "$AUDIT_OUT" | awk '/temporal_range_days/ {print $2}')

# 3a. Not ready — just status ping
if [ "$VOL_OK" = "0" ] || [ "$TEMPORAL_OK" = "0" ]; then
  echo "[+] gates not green: VOL_OK=$VOL_OK TEMPORAL_OK=$TEMPORAL_OK"
  tg "Weekly audit: not ready yet.
matches=$MATCHES (need 150+)
days=$DAYS (need 30+)
Will recheck next Monday."
  exit 0
fi

# 3b. Both gates green — kick off train + eval on 5090
echo "[+] BOTH GATES GREEN — auto-retraining"
tg "Audit gates flipped GREEN — auto-launching Qwen-14B retrain on 5090.
matches=$MATCHES days=$DAYS
~52 min train + ~5 min eval. Will report results."

# Push fresh data
scp -q data/training/sft.jsonl Andre@192.168.76.196:/E:/training/sft_clean.jsonl
scp -q _remote/train_5090.py  Andre@192.168.76.196:/E:/training/train_5090.py
scp -q _remote/eval_5090.py   Andre@192.168.76.196:/E:/training/eval_5090.py

# Train (synchronous over keepalive SSH)
TS=$(date -u +%Y%m%d_%H%M)
ADAPTER="cs2_qwen14b_5090_${TS}"
ssh -o ServerAliveInterval=30 -o ServerAliveCountMax=10000 Andre@192.168.76.196 \
  "py -3.11 E:\\training\\train_5090.py --epochs 1 --no-merge --output E:\\training\\${ADAPTER}" \
  > "_logs/train_${TS}.log" 2>&1 || {
    tg "Auto-retrain FAILED on 5090. Check _logs/train_${TS}.log on the Mac."
    exit 1
  }

# Update eval script's adapter path then run
ssh Andre@192.168.76.196 \
  "py -3.11 E:\\training\\eval_5090.py --adapter E:\\training\\${ADAPTER}" \
  > "_logs/eval_${TS}.log" 2>&1 || true

# Pull metrics
LOSS=$(grep "TRAIN OK" "_logs/train_${TS}.log" | grep -oE "loss=[0-9.]+" | head -1)
PREC=$(grep "precision_buy" "_logs/eval_${TS}.log" | tail -1 | awk '{print $NF}')
PNL=$(grep "sim_pnl_pct_on_buys" "_logs/eval_${TS}.log" | tail -1 | awk '{print $NF}')

tg "Auto-retrain DONE.
adapter=${ADAPTER}
${LOSS}
precision_buy=${PREC}
sim_pnl_pct_on_buys=${PNL}

Decide manually whether to merge+deploy."
echo "[+] done"
