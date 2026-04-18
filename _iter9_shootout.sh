#!/usr/bin/env bash
# iter9 shootout — tests the MINIMAL prompt on three models.
# Waits for the main _model_shootout.sh + _shootout_followup.sh to finish first
# so we don't clobber their Ollama access.
#
# Architecture: iter9 shrinks the user prompt from ~600 tokens to ~100 tokens,
# and the system prompt from ~1200 tokens to ~100 tokens. Only includes:
#   (1) map W-L context   (2) single clearest current advantage
#   (3) ask + spread       (4) ONE latest trigger event
#
# Results land in _logs/shootout/iter9_<model>.log so they show up in the
# index table alongside the iter8 shootout for direct comparison.
#
# Usage: bash _iter9_shootout.sh

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

# Models we want to test on iter9-minimal.
# Added deepseek-r1:8b for the 0528 reasoning upgrade (April 2026 update).
MODELS=(
    "qwen3:30b-a3b|iter9_qwen3_30b"
    "deepseek-r1:14b|iter9_deepseek_r1_14b"
    "deepseek-r1:8b|iter9_deepseek_r1_8b_0528"
    "qwen3.6:35b-a3b|iter9_qwen36_35b"
)

echo "[iter9] waiting for main shootout + followup to finish..."
while pgrep -f "_model_shootout.sh|_shootout_followup.sh" > /dev/null; do
    sleep 30
done
echo "[iter9] previous driver runs exited — starting iter9 tests"

# Freeze iter8 and swap in iter9
if [ ! -f _logs/edge_analyst.iter8_frozen.py ]; then
    echo "[iter9] ERROR: iter8 snapshot missing — aborting to avoid losing iter8"
    exit 1
fi
cp edge_analyst.py _logs/edge_analyst.pre_iter9_backup.py
cp _logs/edge_analyst.iter9_minimal.py edge_analyst.py
echo "[iter9] swapped edge_analyst.py → iter9-minimal"

# Restore iter8 on exit no matter what (crash, ctrl-C, completion)
trap 'cp _logs/edge_analyst.pre_iter9_backup.py edge_analyst.py; echo "[iter9] restored edge_analyst.py to iter8"' EXIT

warmup_model() {
    local model="$1"
    echo "  [warmup] $model..."
    curl -s -m 300 "$OLLAMA_HOST/api/generate" -d "{
        \"model\": \"$model\",
        \"prompt\": \"Return JSON: {\\\"ok\\\":true}\",
        \"stream\": false,
        \"format\": \"json\",
        \"keep_alive\": \"4h\",
        \"options\": {\"temperature\":0, \"num_predict\":20}
    }" -o /tmp/warmup_iter9.json 2>&1 > /dev/null
    [ -s /tmp/warmup_iter9.json ] && grep -q '"response"' /tmp/warmup_iter9.json && return 0
    return 1
}

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|*}"
    LABEL="${entry##*|}"
    echo ""
    echo "============================================================"
    echo "  [iter9] $LABEL  ($MODEL)"
    echo "============================================================"

    # Wait for any leftover backtest
    while pgrep -f replay_backtest.py > /dev/null; do
        sleep 5
    done

    # Verify model is actually installed (qwen3.6 may still be pulling)
    RESP=$(curl -s -m 30 "$OLLAMA_HOST/api/show" -d "{\"name\":\"$MODEL\"}" 2>/dev/null)
    if ! echo "$RESP" | grep -q '"family"'; then
        echo "  [iter9] $MODEL not installed on server — SKIPPING"
        echo "NOT INSTALLED" > "$LOGS/${LABEL}.log"
        continue
    fi

    if ! warmup_model "$MODEL"; then
        echo "  [iter9] warmup failed — SKIPPING"
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
        --tag "iter9_${LABEL}" \
        --model "$MODEL" > "$LOG" 2>&1 &
    PID=$!
    echo "  pid=$PID  log=$LOG"
    while kill -0 $PID 2>/dev/null; do
        sleep 30
    done
    echo "  finished $LABEL"
done

echo ""
echo "[iter9] all iter9 tests complete"
