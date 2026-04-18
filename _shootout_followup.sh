#!/usr/bin/env bash
# Follow-up shootout run for models pulled AFTER the main _model_shootout.sh
# was kicked off. Waits for the main driver to finish, then runs the extras
# on the same config (same files, same prompt, same port) so results land in
# the same _logs/shootout/ directory and show up in the same index table.
#
# Usage: bash _shootout_followup.sh
set -u
cd "$(dirname "$0")"
PY=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
LOGS=_logs/shootout
FILES_LIST=_audit/good_files.txt
DURATION=0
CALLS_PER_MATCH=40
PORT=8084
BAL=1000
OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.76.196:11434}"

# Extra models to slot in after the main shootout.
EXTRAS=(
    "qwen3.6:35b-a3b|qwen36_35b_a3b"
    # Add more follow-ups here if the user pulls additional models
)

# Wait for the main shootout driver to exit and for the Ollama pulls to land.
echo "[followup] waiting for main _model_shootout.sh driver to exit..."
while pgrep -f _model_shootout.sh > /dev/null; do
    sleep 30
done
echo "[followup] main driver exited"

# Wait until the followup model is actually pullable on the server
for entry in "${EXTRAS[@]}"; do
    MODEL="${entry%%|*}"
    echo "[followup] checking $MODEL is available on $OLLAMA_HOST..."
    tries=0
    while [ $tries -lt 180 ]; do
        RESP=$(curl -s -m 30 "$OLLAMA_HOST/api/show" -d "{\"name\":\"$MODEL\"}" 2>/dev/null)
        if echo "$RESP" | grep -q "family"; then
            echo "[followup] $MODEL is installed"
            break
        fi
        tries=$((tries+1))
        sleep 30
    done
done

warmup_model() {
    local model="$1"
    echo "  [warmup] loading $model into VRAM..."
    curl -s -m 300 "$OLLAMA_HOST/api/generate" -d "{
        \"model\": \"$model\",
        \"prompt\": \"Return JSON: {\\\"ok\\\":true}\",
        \"stream\": false,
        \"format\": \"json\",
        \"keep_alive\": \"4h\",
        \"options\": {\"temperature\":0, \"num_predict\":20}
    }" -o /tmp/warmup.json 2>&1 > /dev/null
    [ -s /tmp/warmup.json ] && grep -q "response" /tmp/warmup.json && return 0
    return 1
}

for entry in "${EXTRAS[@]}"; do
    MODEL="${entry%%|*}"
    LABEL="${entry##*|}"
    echo ""
    echo "============================================================"
    echo "  [followup] $LABEL  ($MODEL)"
    echo "============================================================"

    while pgrep -f replay_backtest.py > /dev/null; do
        echo "  waiting for previous backtest to free port $PORT..."
        sleep 5
    done

    if ! warmup_model "$MODEL"; then
        echo "  SKIPPED $LABEL (warmup failed)"
        echo "WARMUP FAILED" > "$LOGS/${LABEL}.log"
        continue
    fi

    LOG="$LOGS/${LABEL}.log"
    $PY replay_backtest.py \
        --files-list "$FILES_LIST" \
        --duration $DURATION \
        --calls-per-match $CALLS_PER_MATCH \
        --port $PORT \
        --starting-balance $BAL \
        --tag "followup_${LABEL}" \
        --model "$MODEL" > "$LOG" 2>&1 &
    PID=$!
    echo "  pid=$PID  log=$LOG"
    while kill -0 $PID 2>/dev/null; do
        sleep 30
    done
    echo "  finished $LABEL"
done

echo ""
echo "[followup] all extras done — regenerating summary"
$PY _shootout_summary.py 2>/dev/null || true
