#!/usr/bin/env bash
# Model shootout — runs the SAME backtest configuration across multiple
# Ollama models and produces a side-by-side comparison.
#
# All models run on the same big file sample, same starting balance,
# same iter8 prompt. Only OLLAMA_MODEL varies.
#
# Dashboard: http://localhost:8084  (auto-refreshes every 2s)
# Shootout index: _logs/shootout/index.html
#
# Usage: bash _model_shootout.sh
#
set -u
cd "$(dirname "$0")"
PY=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
LOGS=_logs/shootout
mkdir -p "$LOGS"

# Models verified to actually load on the Ollama server.
# Ordered fastest → slowest so we get signal from affordable models first.
# phi4:14b + mistral-small3.2:24b are placed LATE because they were still
# pulling when this was kicked off — the warmup check will skip them
# gracefully if they're still not ready by the time their slot arrives.
# llama3.3:70b-q4 is last because it needs ~40GB free VRAM.
MODELS=(
    "llama3.1:8b-instruct-q4_K_M|llama31_8b"
    "gemma4:latest|gemma4_8b"
    "qwen2.5:14b-instruct-q4_K_M|qwen25_14b"
    "deepseek-r1:14b|deepseek_r1_14b"
    "qwen3:30b-a3b|qwen3_30b_a3b_baseline"
    "esports-qwen3:latest|esports_qwen3_custom"
    "qwen2.5:32b-instruct-q4_K_M|qwen25_32b"
    "qwen2.5-coder:32b|qwen25_coder_32b"
    "phi4:14b|phi4_14b"
    "mistral-small3.2:24b|mistral_small_24b"
    "deepseek-r1:32b|deepseek_r1_32b"
    "qwen3:30b-a3b-think|qwen3_30b_think"
    "llama3.3:70b-instruct-q4_K_M|llama33_70b"
)

# Use the 32-file curated set — real recordings have no rich snapshots yet so
# they emit zero triggers. Once today's bo3.gg-payload recorder patch starts
# producing data, we can rerun on _audit/shootout_cs2.txt for the big sample.
FILES_LIST=_audit/good_files.txt
DURATION=0                           # 0 = no time limit
CALLS_PER_MATCH=40
PORT=8084                            # dashboard port (per-model backtest)
INDEX_PORT=8085                      # always-on index server
BAL=1000
OLLAMA_HOST="${OLLAMA_HOST:-http://192.168.76.196:11434}"

# Active model indicator file for the dashboard
ACTIVE_FILE="$LOGS/_active_model.txt"

# Kick off a tiny always-on HTTP server on INDEX_PORT so the link stays alive
# during model-swap gaps (when replay_backtest.py is NOT running).
INDEX_PID_FILE="$LOGS/_index_server.pid"
if [ -f "$INDEX_PID_FILE" ] && kill -0 "$(cat $INDEX_PID_FILE)" 2>/dev/null; then
    echo "[index] server already running pid=$(cat $INDEX_PID_FILE)"
else
    $PY _shootout_index_server.py $INDEX_PORT > "$LOGS/_index_server.log" 2>&1 &
    echo $! > "$INDEX_PID_FILE"
    echo "[index] started pid=$(cat $INDEX_PID_FILE)  →  http://localhost:$INDEX_PORT/"
    sleep 1
fi
trap 'kill $(cat "$INDEX_PID_FILE" 2>/dev/null) 2>/dev/null' EXIT

warmup_model() {
    local model="$1"
    echo "  [warmup] loading $model into VRAM..."
    # Single short request forces ollama to load the model. This can take
    # 30-90s for 24-70B models on first call.
    curl -s -m 180 "$OLLAMA_HOST/api/generate" -d "{
        \"model\": \"$model\",
        \"prompt\": \"Return JSON: {\\\"ok\\\":true}\",
        \"stream\": false,
        \"format\": \"json\",
        \"options\": {\"temperature\":0, \"num_predict\":20}
    }" -o /tmp/warmup.json 2>&1 > /dev/null
    if [ $? -eq 0 ] && [ -s /tmp/warmup.json ]; then
        echo "  [warmup] $model ready"
        return 0
    fi
    echo "  [warmup] $model FAILED to warmup — skipping"
    return 1
}

# Write the shootout index page
write_index() {
    local current="$1"
    python3 -c "
import os, json, re
logs_dir = '$LOGS'
current = '$current'
rows = []
for fn in sorted(os.listdir(logs_dir)):
    if not fn.endswith('.log'): continue
    label = fn[:-4]
    path = os.path.join(logs_dir, fn)
    with open(path) as f: text = f.read()
    m = re.search(r'BACKTEST COMPLETE.+?PnL \\\$([+-][\d.]+).+?(\d+) trades \((\d+)W/(\d+)L\).+?(\d+) qwen calls', text, re.S)
    if m:
        status = 'DONE'
        pnl, trades, w, l, calls = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
    elif 'WARMUP FAILED' in text:
        status = 'SKIPPED'
        pnl = trades = w = l = calls = '—'
    else:
        status = 'RUNNING' if label == current else 'PENDING'
        pnl = trades = w = l = calls = '—'
    rows.append((label, status, pnl, trades, w, l, calls))

html = '<!doctype html><html><head><title>Shootout</title><meta http-equiv=\"refresh\" content=\"5\"><style>body{font-family:-apple-system,Arial;margin:20px;background:#0d1117;color:#c9d1d9}h1{color:#58a6ff}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #30363d;text-align:left}th{color:#8b949e;text-transform:uppercase;font-size:11px}.pos{color:#7ee787}.neg{color:#ff7b72}.run{color:#f1e05a}.pending{color:#8b949e}a{color:#58a6ff}</style></head><body>'
html += '<h1>Model Shootout — iter8 prompt · 200 recordings · \$1000 start</h1>'
html += '<p>Live dashboard for the active run: <a href=\"http://localhost:8084\" target=\"_blank\">http://localhost:8084</a></p>'
html += '<table><tr><th>Model</th><th>Status</th><th>PnL</th><th>Trades</th><th>W</th><th>L</th><th>Qwen Calls</th></tr>'
for r in rows:
    label, status, pnl, trades, w, l, calls = r
    cls = 'run' if status=='RUNNING' else ('pos' if (isinstance(pnl,str) and pnl.startswith('+')) else 'neg' if (isinstance(pnl,str) and pnl.startswith('-')) else ('neg' if status=='SKIPPED' else 'pending'))
    html += f'<tr><td><b>{label}</b></td><td class=\"{cls}\">{status}</td><td class=\"{cls}\">{pnl}</td><td>{trades}</td><td>{w}</td><td>{l}</td><td>{calls}</td></tr>'
html += '</table>'
html += f'<p style=\"color:#8b949e;font-size:11px\">refreshes every 5s · {len(rows)} models tracked</p>'
html += '</body></html>'
with open(os.path.join(logs_dir, 'index.html'), 'w') as f: f.write(html)
"
}

for entry in "${MODELS[@]}"; do
    MODEL="${entry%%|*}"
    LABEL="${entry##*|}"
    echo ""
    echo "============================================================"
    echo "  [shootout] $LABEL  ($MODEL)"
    echo "============================================================"
    echo "$LABEL" > "$ACTIVE_FILE"
    write_index "$LABEL"

    # Wait until no backtest is running
    while pgrep -f replay_backtest.py > /dev/null; do
        echo "  waiting for previous backtest to free port $PORT..."
        sleep 5
    done

    # Warmup: force ollama to load the model before we start the backtest.
    # Skip this model if warmup fails.
    if ! warmup_model "$MODEL"; then
        echo "  SKIPPED $LABEL (warmup failed)"
        echo "WARMUP FAILED" > "$LOGS/${LABEL}.log"
        write_index ""
        continue
    fi

    LOG="$LOGS/${LABEL}.log"
    $PY replay_backtest.py \
        --files-list "$FILES_LIST" \
        --duration $DURATION \
        --calls-per-match $CALLS_PER_MATCH \
        --port $PORT \
        --starting-balance $BAL \
        --tag "shootout_${LABEL}" \
        --model "$MODEL" > "$LOG" 2>&1 &

    PID=$!
    echo "  pid=$PID  log=$LOG"

    # Poll until this run finishes (process exits).
    while kill -0 $PID 2>/dev/null; do
        sleep 30
        write_index "$LABEL"
        if [ -f "$LOG" ]; then
            LAST=$(grep -E "BACKTEST COMPLETE|snapshots=|\[BUY\]|\[EXIT\]" "$LOG" | tail -1)
            [ -n "$LAST" ] && echo "  ... $LAST"
        fi
    done
    echo "  finished $LABEL"
    write_index ""
done

echo ""
echo "============================================================"
echo "  all runs finished — generating summary"
echo "============================================================"
rm -f "$ACTIVE_FILE"
write_index ""
$PY _shootout_summary.py
