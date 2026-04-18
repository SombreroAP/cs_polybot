#!/bin/bash
# 5-min monitor — runs for 5 hours, reports status to _monitor.log
# and auto-restarts the backtest whenever it stops without live games being active.

cd /Users/andrew/Documents/Claude/Projects/Trading/esports
PY=/Applications/Xcode.app/Contents/Developer/usr/bin/python3
LOG=_monitor.log
LIVE_GAME_FLAG=_live_game_active.flag   # touch-file: when set, do NOT run backtest

# Run for up to 5.5 hours (just in case)
END_TS=$(( $(date +%s) + 5*3600 + 1800 ))

echo "[$(date '+%H:%M:%S')] === MONITOR STARTED ===" >> $LOG

while [ "$(date +%s)" -lt "$END_TS" ]; do
    NOW=$(date '+%H:%M:%S')

    # 1. Check live bot (port 8082)
    LIVE_OK="no"
    LIVE_ACTIVE_MATCHES=0
    LIVE_INFO=$(curl -s --max-time 5 http://localhost:8082/api/state 2>/dev/null | \
        $PY -c "
import json, sys
try:
    d = json.load(sys.stdin)
    am = d.get('active_matches', [])
    live_count = sum(1 for m in am if str(m.get('status','')).lower() in ('live','running','in_progress','active'))
    ts = d.get('trade_stats', {})
    print(f'ok|{len(am)}|{live_count}|{ts.get(\"total\",0)}|{ts.get(\"open\",0)}|{ts.get(\"balance\",0):.2f}|{ts.get(\"pnl\",0):+.2f}')
except Exception as e:
    print(f'err|{e}')
" 2>/dev/null)
    if echo "$LIVE_INFO" | grep -q "^ok|"; then
        LIVE_OK="yes"
        IFS='|' read -r _ NAM LAM TOT OPEN BAL PNL <<< "$LIVE_INFO"
        LIVE_ACTIVE_MATCHES="$LAM"
    fi

    # 2. Decide: live game active? (if any active_matches in 'live/running' state)
    if [ "$LIVE_ACTIVE_MATCHES" -gt 0 ]; then
        touch "$LIVE_GAME_FLAG"
    else
        rm -f "$LIVE_GAME_FLAG"
    fi

    # 3. Check backtest (port 8083)
    BT_INFO=$(curl -s --max-time 5 http://localhost:8083/api/state 2>/dev/null | \
        $PY -c "
import json, sys
try:
    d = json.load(sys.stdin)
    print(f'ok|{d[\"status\"]}|{d[\"matches_done\"]}|{d[\"matches_total\"]}|{d[\"balance\"]:.2f}|{d[\"pnl\"]:+.2f}|{d[\"total_trades\"]}|{d[\"wins\"]}|{d[\"losses\"]}|{d[\"qwen_calls\"]}')
except:
    print('err')
" 2>/dev/null)

    BT_OK="no"
    BT_STATUS=""; BT_DONE=0; BT_TOTAL=0; BT_BAL=0; BT_PNL=0
    if echo "$BT_INFO" | grep -q "^ok|"; then
        BT_OK="yes"
        IFS='|' read -r _ BT_STATUS BT_DONE BT_TOTAL BT_BAL BT_PNL BT_TRADES BT_W BT_L BT_QC <<< "$BT_INFO"
    fi

    # 4. Should backtest be running now?
    # - If LIVE game active → do NOT run backtest (leave CPU/qwen for live bot)
    # - If no live game and backtest not running → start a new run
    BT_RUNNING=$(pgrep -f "replay_backtest.py" | head -1)
    SHOULD_BACKTEST="yes"
    if [ -f "$LIVE_GAME_FLAG" ]; then SHOULD_BACKTEST="no"; fi

    if [ "$SHOULD_BACKTEST" = "yes" ] && [ -z "$BT_RUNNING" ]; then
        echo "[$NOW] no backtest running and no live games → starting new run" >> $LOG
        nohup $PY replay_backtest.py \
            --files-list _audit/good_files.txt \
            --duration 5400 --calls-per-match 40 \
            --port 8083 --starting-balance 1000 \
            --tag auto > replay_backtest.log 2>&1 &
        NEW_PID=$!
        echo "[$NOW]   new PID=$NEW_PID" >> $LOG
        sleep 3
    fi

    if [ "$SHOULD_BACKTEST" = "no" ] && [ -n "$BT_RUNNING" ]; then
        echo "[$NOW] live games active → stopping backtest to free resources" >> $LOG
        pkill -f replay_backtest.py
        sleep 2
    fi

    # 5. Report
    echo "[$NOW] live: bot=$LIVE_OK matches=$NAM live_games=$LIVE_ACTIVE_MATCHES trades=$TOT open=$OPEN bal=\$$BAL pnl=$PNL | bt: ok=$BT_OK $BT_STATUS $BT_DONE/$BT_TOTAL bal=\$$BT_BAL pnl=\$$BT_PNL trades=$BT_TRADES ($BT_W W / $BT_L L) qwen=$BT_QC | bt_pid=$BT_RUNNING" >> $LOG

    sleep 300  # 5 min
done

echo "[$(date '+%H:%M:%S')] === MONITOR ENDED ===" >> $LOG
