#!/bin/bash
PYTHON="/Applications/Xcode.app/Contents/Developer/usr/bin/python3"
cd "$(dirname "$0")"

echo "=== RUNNING ALL TEST DATA FILES ==="
TOTAL_PNL=0
TOTAL_WINS=0
TOTAL_LOSSES=0
TOTAL_TRADES=0

for FILE in "Test Data"/*.jsonl; do
    MATCH=$(basename "$FILE" .jsonl)
    echo ""
    echo "============================================"
    echo "TESTING: $MATCH"
    echo "============================================"
    
    # Run replay (no dashboard, just results)
    $PYTHON -c "
import asyncio, json, sys, os
sys.path.insert(0, '.')
os.environ['OLLAMA_URL'] = 'http://192.168.76.196:11435'
os.environ['OLLAMA_MODEL'] = 'gemma4'

from test_replay import replay_match, state, ReplayState

# Reset state
import test_replay
test_replay.state = ReplayState()
state = test_replay.state

async def run():
    await replay_match('$FILE')
    wins = sum(1 for t in state.trades if t['pnl'] > 0)
    losses = len(state.trades) - wins
    print(f'RESULT|{wins}|{losses}|{state.total_pnl:.2f}|{state.gemma_calls}|{state.balance:.2f}')

asyncio.run(run())
" 2>&1 | tail -5
    
done

echo ""
echo "============================================"
echo "ALL TESTS COMPLETE"
echo "============================================"

# Restart live bot
echo "Restarting live bot..."
rm -f .pause_analyzer .stop_bot
nohup ./run_overnight.sh > overnight.log 2>&1 &
echo "Live bot restarted"
