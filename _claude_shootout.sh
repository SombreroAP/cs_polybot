#!/usr/bin/env bash
# Claude API shootout — runs Haiku, Sonnet, and Opus against the same 32-file
# corpus as the local-model shootout, with the SAME iter8 prompt, so results
# land in _logs/shootout/ alongside local-model results for direct comparison.
#
# Works even when the gaming PC (Ollama) is off — Claude API is cloud-side.
#
# Usage: bash _claude_shootout.sh
#        bash _claude_shootout.sh iter9     # use iter9-minimal prompt instead

set -u
cd "$(dirname "$0")"
PY=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
LOGS=_logs/shootout
mkdir -p "$LOGS"

# Prompt variant — default iter8, override to iter9 by passing "iter9" as arg
VARIANT="${1:-iter8}"
if [ "$VARIANT" = "iter9" ]; then
    echo "[claude] swapping edge_analyst.py → iter9-minimal"
    cp edge_analyst.py _logs/edge_analyst.pre_claude_backup.py
    cp _logs/edge_analyst.iter9_minimal.py edge_analyst.py
    trap 'cp _logs/edge_analyst.pre_claude_backup.py edge_analyst.py; echo "[claude] restored edge_analyst.py"' EXIT
    TAG_PREFIX="claude_iter9"
else
    TAG_PREFIX="claude_iter8"
fi

# Anthropic models to test
# Note: we route via Claude API by UNsetting OLLAMA_URL so edge_analyst falls
# through to the Anthropic path. replay_backtest passes --model to override
# OLLAMA_MODEL which is ignored when Ollama is disabled — we need a different
# switch: EDGE_CLAUDE_MODEL env var.
MODELS=(
    "claude-haiku-4-5-20251001|${TAG_PREFIX}_haiku45"
    "claude-sonnet-4-5-20251001|${TAG_PREFIX}_sonnet45"
    "claude-opus-4-7-20260115|${TAG_PREFIX}_opus47"
)

FILES_LIST=_audit/good_files.txt
DURATION=0
CALLS_PER_MATCH=40
PORT=8084
BAL=1000

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|*}"
    LABEL="${entry##*|}"
    echo ""
    echo "============================================================"
    echo "  [claude] $LABEL  ($MODEL)"
    echo "============================================================"

    while pgrep -f replay_backtest.py > /dev/null; do
        echo "  waiting for previous backtest..."
        sleep 5
    done

    LOG="$LOGS/${LABEL}.log"

    # Route via Claude API by:
    #  - clearing OLLAMA_URL  (disables local path)
    #  - setting EDGE_CLAUDE_MODEL to the tier under test
    #  - `unset ANTHROPIC_API_KEY` first so the empty value inherited from
    #    Claude-for-Desktop's shell doesn't shadow the real key in .env
    (unset ANTHROPIC_API_KEY; \
     OLLAMA_URL="" \
     EDGE_CLAUDE_MODEL="$MODEL" \
     $PY replay_backtest.py \
        --files-list "$FILES_LIST" \
        --duration $DURATION \
        --calls-per-match $CALLS_PER_MATCH \
        --port $PORT \
        --starting-balance $BAL \
        --tag "${LABEL}" > "$LOG" 2>&1) &
    PID=$!
    echo "  pid=$PID  log=$LOG"

    while kill -0 $PID 2>/dev/null; do
        sleep 30
        if [ -f "$LOG" ]; then
            LAST=$(grep -E "BACKTEST COMPLETE|\[BUY\]|\[EXIT\]" "$LOG" | tail -1)
            [ -n "$LAST" ] && echo "  ... $LAST"
        fi
    done
    echo "  finished $LABEL"
done

echo ""
echo "[claude] all Claude-tier runs complete"
