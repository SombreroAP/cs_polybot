#!/bin/bash
# Resilient auto-restart wrapper for edge bot — designed to NEVER stop
#
# Launch (detached from terminal, survives logout):
#   cd /Users/andrew/Documents/Claude/Projects/Trading/esports
#   setsid nohup ./run_overnight.sh < /dev/null >> overnight.log 2>&1 &
#
# Stop gracefully:
#   touch .stop_bot
#
# Check if wrapper alive:
#   cat .wrapper.pid && ps -p $(cat .wrapper.pid)

set -u
cd "$(dirname "$0")"

# ─── Config ──────────────────────────────────────────────
PYTHON_CANDIDATES=(
    "/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
    "/opt/homebrew/bin/python3"
    "/usr/bin/python3"
)
SCRIPT="edge_bot.py"
PORT=8082
BOT_LOG="nohup.out"
WRAPPER_LOG="overnight.log"
PID_FILE=".wrapper.pid"
RESTART_DELAY=5
BACKOFF_MAX=60          # cap exponential backoff
LOG_MAX_BYTES=$((50 * 1024 * 1024))   # rotate nohup.out over 50MB
CONNECTIVITY_HOSTS=("clob.polymarket.com" "gamma-api.polymarket.com")

# ─── Signal handling: DO NOT die on SIGHUP/SIGTERM ───────
# Only .stop_bot marker exits the loop cleanly.
trap 'log "received SIGHUP — ignoring (use .stop_bot to stop)"' HUP
trap 'log "received SIGTERM — ignoring (use .stop_bot to stop)"' TERM
trap 'log "received SIGINT — ignoring (use .stop_bot to stop)"' INT

log() {
    echo "$(date '+%Y-%m-%d %H:%M:%S') [WRAPPER] $*"
}

# ─── Prevent multiple wrappers ────────────────────────────
if [ -f "$PID_FILE" ]; then
    old_pid=$(cat "$PID_FILE" 2>/dev/null)
    if [ -n "$old_pid" ] && ps -p "$old_pid" > /dev/null 2>&1; then
        log "another wrapper already running (pid $old_pid) — exiting"
        exit 1
    fi
fi
echo $$ > "$PID_FILE"
log "wrapper started (pid $$)"

# Clear stop marker if stale
rm -f .stop_bot

# ─── Pick python ─────────────────────────────────────────
PYTHON=""
for candidate in "${PYTHON_CANDIDATES[@]}"; do
    if [ -x "$candidate" ] && "$candidate" -c "import aiohttp" 2>/dev/null; then
        PYTHON="$candidate"
        break
    fi
done
if [ -z "$PYTHON" ]; then
    log "FATAL: no python with aiohttp found — tried: ${PYTHON_CANDIDATES[*]}"
    rm -f "$PID_FILE"
    exit 1
fi
log "using python: $PYTHON"

# ─── Helpers ─────────────────────────────────────────────
rotate_log() {
    if [ -f "$BOT_LOG" ]; then
        size=$(stat -f%z "$BOT_LOG" 2>/dev/null || stat -c%s "$BOT_LOG" 2>/dev/null || echo 0)
        if [ "$size" -gt "$LOG_MAX_BYTES" ]; then
            ts=$(date '+%Y%m%d_%H%M%S')
            mv "$BOT_LOG" "${BOT_LOG}.${ts}"
            log "rotated $BOT_LOG (${size} bytes) -> ${BOT_LOG}.${ts}"
            # keep only last 5 rotations
            ls -t ${BOT_LOG}.* 2>/dev/null | tail -n +6 | xargs rm -f 2>/dev/null || true
        fi
    fi
}

wait_for_network() {
    # Block until at least one host resolves. Never gives up.
    local tries=0
    while true; do
        for host in "${CONNECTIVITY_HOSTS[@]}"; do
            if host "$host" > /dev/null 2>&1 || \
               ping -c 1 -t 2 "$host" > /dev/null 2>&1 || \
               curl -s --max-time 3 -o /dev/null "https://$host" 2>/dev/null; then
                if [ $tries -gt 0 ]; then
                    log "network recovered after ${tries} checks"
                fi
                return 0
            fi
        done
        tries=$((tries + 1))
        if [ $((tries % 6)) -eq 1 ]; then
            log "waiting for network (check #$tries)..."
        fi
        sleep 10
    done
}

kill_stale_bot() {
    # Only kill edge_bot processes NOT spawned by us.
    # (Our child shares our process group; others don't.)
    our_pgid=$(ps -o pgid= -p $$ | tr -d ' ')
    for pid in $(pgrep -f "python.*$SCRIPT" 2>/dev/null); do
        [ "$pid" = "$$" ] && continue
        pid_pgid=$(ps -o pgid= -p "$pid" 2>/dev/null | tr -d ' ')
        if [ "$pid_pgid" != "$our_pgid" ]; then
            log "killing stale bot pid=$pid (pgid=$pid_pgid, ours=$our_pgid)"
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 1
}

cleanup() {
    log "wrapper exiting (pid $$)"
    rm -f "$PID_FILE"
}
trap 'cleanup' EXIT

# ─── Main loop: truly infinite ───────────────────────────
kill_stale_bot

restart_count=0
consecutive_fast_failures=0
backoff=$RESTART_DELAY

while true; do
    if [ -f .stop_bot ]; then
        log "stop marker found — exiting"
        rm -f .stop_bot
        break
    fi

    rotate_log
    wait_for_network

    restart_count=$((restart_count + 1))
    log "starting bot (restart #$restart_count)"

    start_ts=$(date +%s)
    "$PYTHON" "$SCRIPT" --port "$PORT" >> "$BOT_LOG" 2>&1
    exit_code=$?
    end_ts=$(date +%s)
    duration=$((end_ts - start_ts))

    log "bot exited code=$exit_code after ${duration}s"

    # Exponential backoff only on fast failures (<30s up)
    if [ "$duration" -lt 30 ]; then
        consecutive_fast_failures=$((consecutive_fast_failures + 1))
        backoff=$((RESTART_DELAY * (1 << (consecutive_fast_failures > 4 ? 4 : consecutive_fast_failures))))
        [ $backoff -gt $BACKOFF_MAX ] && backoff=$BACKOFF_MAX
        log "fast failure (#$consecutive_fast_failures) — backing off ${backoff}s"
    else
        consecutive_fast_failures=0
        backoff=$RESTART_DELAY
    fi

    if [ -f .stop_bot ]; then
        log "stop marker found — exiting"
        rm -f .stop_bot
        break
    fi

    sleep $backoff
done
