#!/usr/bin/env python3
"""
Test Replay Engine — replays historical match data through Gemma 4.

Reads JSONL files from Test Data/ folder and simulates the match in real-time,
feeding each event to Gemma for buy/skip decisions. Tracks simulated P&L.

Runs on port 8083 with its own dashboard.
"""
import asyncio
import aiohttp
import json
import logging
import os
import sys
import time
import argparse
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional
from flask import Flask, jsonify, render_template_string

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("replay")

# Config
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://192.168.76.196:11435")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma4")
SPEED_MULTIPLIER = 50  # 50x speed — 1 hour match in ~72 seconds

SYSTEM_PROMPT = """You are an esports latency arbitrage trader analyzing CS2 matches.
You see game events 2-15 seconds before the betting market reacts.

Given the game state and recent events, decide: BUY the favored team, or SKIP.
Consider: round score, economy, momentum, kill streaks, map position.
In CS2: gun round wins > eco wins. Match point situations are high value.
Factor in the SPREAD — buying at ask means instant loss if price doesn't move.

Reply ONLY with JSON: {"action":"buy_a" or "buy_b" or "skip","confidence":0.0-1.0,"reason":"brief"}"""


@dataclass
class ReplayState:
    match_id: int = 0
    team_a: str = ""
    team_b: str = ""
    map_name: str = ""
    score_a: int = 0  # series
    score_b: int = 0
    round_a: int = 0  # current map
    round_b: int = 0
    economy_a: int = 0
    economy_b: int = 0
    bid_a: float = 0
    ask_a: float = 0
    bid_b: float = 0
    ask_b: float = 0
    last_trade: float = 0
    events: list = field(default_factory=list)
    decisions: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    balance: float = 1000.0
    total_pnl: float = 0.0
    gemma_calls: int = 0
    gemma_latency_ms: list = field(default_factory=list)
    current_time: str = ""
    progress: float = 0.0
    status: str = "loading"
    open_positions: list = field(default_factory=list)


state = ReplayState()


async def call_gemma(prompt: str) -> Optional[dict]:
    """Call Gemma 4 for a trading decision."""
    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_predict": 120},
    }
    t0 = time.time()
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{OLLAMA_URL}/api/chat", json=payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                data = await resp.json()
        text = data.get("message", {}).get("content", "")
        latency = (time.time() - t0) * 1000
        state.gemma_calls += 1
        state.gemma_latency_ms.append(latency)

        if "```" in text:
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        text = text.strip()
        result = json.loads(text)
        result["latency_ms"] = latency
        return result
    except Exception as e:
        logger.error(f"Gemma error: {e}")
        return None


def check_positions(state: ReplayState):
    """Check open positions for TP/SL."""
    for pos in list(state.open_positions):
        if pos["team"] == "a":
            current_bid = state.bid_a
        else:
            current_bid = state.bid_b

        if current_bid <= 0:
            continue

        pnl_pct = (current_bid - pos["entry"]) / pos["entry"]
        pos["current"] = current_bid
        pos["pnl_pct"] = pnl_pct

        if pnl_pct >= 0.075:  # TP
            pnl = pos["stake"] * pnl_pct
            state.balance += pos["stake"] + pnl
            state.total_pnl += pnl
            state.trades.append({
                "team": pos["team_name"], "entry": pos["entry"], "exit": current_bid,
                "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 1),
                "reason": "take_profit", "time": state.current_time,
            })
            state.open_positions.remove(pos)
            logger.info(f"TP: {pos['team_name']} +${pnl:.2f} ({pnl_pct*100:+.1f}%)")
        elif pnl_pct <= -0.125 and time.time() - pos.get("opened_at", 0) > 15:  # SL after grace
            pnl = pos["stake"] * pnl_pct
            state.balance += pos["stake"] + pnl
            state.total_pnl += pnl
            state.trades.append({
                "team": pos["team_name"], "entry": pos["entry"], "exit": current_bid,
                "pnl": round(pnl, 2), "pnl_pct": round(pnl_pct * 100, 1),
                "reason": "stop_loss", "time": state.current_time,
            })
            state.open_positions.remove(pos)
            logger.info(f"SL: {pos['team_name']} ${pnl:.2f} ({pnl_pct*100:+.1f}%)")


async def replay_match(filepath: str):
    """Replay a match file through Gemma 4."""
    logger.info(f"Loading {os.path.basename(filepath)}...")
    state.status = "loading"

    lines = []
    with open(filepath) as f:
        for line in f:
            lines.append(json.loads(line))

    # Get meta
    meta = lines[0]["raw"] if lines[0]["message_type"] == "_META" else {}
    state.match_id = meta.get("match_id", 0)
    state.team_a = meta.get("team1", "Team A")
    state.team_b = meta.get("team2", "Team B")
    state.status = "replaying"

    logger.info(f"Replaying: {state.team_a} vs {state.team_b} ({len(lines):,} events)")

    # Sort by timestamp
    lines.sort(key=lambda x: x.get("ts", 0))
    start_ts = lines[0]["ts"]
    end_ts = lines[-1]["ts"]
    duration = end_ts - start_ts

    replay_start = time.time()
    last_gemma_call = 0
    round_events = []

    for i, event in enumerate(lines):
        state.progress = i / len(lines) * 100
        state.current_time = event.get("ts_iso", "")[:19]

        # Speed control — replay at SPEED_MULTIPLIER
        elapsed_real = time.time() - replay_start
        elapsed_game = event["ts"] - start_ts
        target_real = elapsed_game / SPEED_MULTIPLIER
        if target_real > elapsed_real:
            await asyncio.sleep(target_real - elapsed_real)

        msg_type = event["message_type"]
        raw = event.get("raw", {})

        # Update orderbook
        if msg_type == "_PM_best_bid_ask":
            # Determine which token
            if raw.get("bestBid", 0) > 0:
                # Heuristic: lower-priced token is the one with lower bid
                bid = raw["bestBid"]
                ask = raw["bestAsk"]
                if state.bid_a == 0 or abs(bid - state.bid_a) < abs(bid - state.bid_b):
                    state.bid_a = bid
                    state.ask_a = ask
                else:
                    state.bid_b = bid
                    state.ask_b = ask
            check_positions(state)

        elif msg_type == "_PM_book":
            bid = raw.get("bestBid", 0)
            ask = raw.get("bestAsk", 0)
            if bid > 0:
                if state.bid_a == 0 or abs(bid - state.bid_a) < abs(bid - state.bid_b):
                    state.bid_a = bid
                    state.ask_a = ask
                else:
                    state.bid_b = bid
                    state.ask_b = ask

        elif msg_type == "SNAPSHOT_MATCH_UPDATE":
            t1 = raw.get("team_one", {})
            t2 = raw.get("team_two", {})
            new_ra = t1.get("score", 0) or 0
            new_rb = t2.get("score", 0) or 0
            state.map_name = raw.get("map_name", "")
            # Update economy from snapshot
            eq1 = t1.get("equipment_value", 0) or 0
            eq2 = t2.get("equipment_value", 0) or 0
            if eq1 > 0 or eq2 > 0:
                state.economy_a = eq1
                state.economy_b = eq2

            if new_ra != state.round_a or new_rb != state.round_b:
                winner = state.team_a if new_ra > state.round_a else state.team_b
                state.round_a = new_ra
                state.round_b = new_rb

                round_events.append({
                    "type": "round_end",
                    "desc": f"Round to {winner} ({new_ra}-{new_rb}) on {state.map_name}",
                    "time": state.current_time,
                })
                state.events.append({
                    "type": "round_end", "team": winner,
                    "desc": f"{winner} wins round ({new_ra}-{new_rb})",
                    "time": state.current_time,
                    "bid_a": state.bid_a, "ask_a": state.ask_a,
                })
                if len(state.events) > 100:
                    state.events = state.events[-80:]

                logger.info(f"ROUND: {state.team_a} {new_ra}-{new_rb} {state.team_b} | bid={state.bid_a:.2f} ask={state.ask_a:.2f}")

                # Call Gemma on meaningful rounds
                if time.time() - last_gemma_call > 2:
                    last_gemma_call = time.time()
                    prompt = f"""CS2 | {state.team_a} vs {state.team_b}
Map: {state.map_name} | Round: {state.round_a}-{state.round_b}
Series: {state.score_a}-{state.score_b}
Economy: {state.team_a}=${state.economy_a:,} | {state.team_b}=${state.economy_b:,}
Orderbook: {state.team_a} bid={state.bid_a*100:.0f}c ask={state.ask_a*100:.0f}c | {state.team_b} bid={state.bid_b*100:.0f}c ask={state.ask_b*100:.0f}c
Spread: {(state.ask_a-state.bid_a)*100:.0f}c

RECENT EVENTS:
""" + "\n".join([f"- {e['desc']}" for e in round_events[-6:]]) + f"""

Should we trade? Which team?"""

                    result = await call_gemma(prompt)
                    if result:
                        decision = {
                            "action": result.get("action", "skip"),
                            "confidence": result.get("confidence", 0),
                            "reason": result.get("reason", ""),
                            "time": state.current_time,
                            "round": f"{state.round_a}-{state.round_b}",
                            "latency_ms": result.get("latency_ms", 0),
                            "bid_a": state.bid_a, "ask_a": state.ask_a,
                        }
                        state.decisions.append(decision)
                        if len(state.decisions) > 50:
                            state.decisions = state.decisions[-40:]

                        # Execute trade
                        action = result.get("action", "skip")
                        conf = result.get("confidence", 0)
                        if action in ("buy_a", "buy_b") and conf >= 0.7 and len(state.open_positions) < 2:
                            team = "a" if action == "buy_a" else "b"
                            team_name = state.team_a if team == "a" else state.team_b
                            entry = state.ask_a if team == "a" else state.ask_b
                            if 0.15 < entry < 0.87 and entry > 0:
                                stake = 25.0
                                state.balance -= stake
                                state.open_positions.append({
                                    "team": team, "team_name": team_name,
                                    "entry": entry, "stake": stake,
                                    "current": entry, "pnl_pct": 0,
                                    "reason": result.get("reason", ""),
                                    "opened_at": time.time(),
                                })
                                logger.info(f"BUY {team_name} @ {entry:.3f} | conf={conf:.0%} | {result.get('reason','')[:50]}")

        elif msg_type in ("GAME_EVENT_TEAMS_ECONOMY_INFO", "SNAPSHOT_MATCH_UPDATE"):
            t1 = raw.get("team_one", {})
            t2 = raw.get("team_two", {})
            eq1 = t1.get("equipment_value", 0) or 0
            eq2 = t2.get("equipment_value", 0) or 0
            if eq1 > 0 or eq2 > 0:
                state.economy_a = eq1
                state.economy_b = eq2

        elif msg_type == "GAME_EVENT_PLAYER_KILL":
            killer = raw.get("killer_display_name", "?")
            victim = raw.get("victim_display_name", "?")
            weapon = raw.get("weapon_name", "?")
            round_events.append({
                "type": "kill",
                "desc": f"{killer} killed {victim} with {weapon}",
                "time": state.current_time,
            })

        elif msg_type == "GAME_EVENT_MATCH_END_ROUND":
            # Map might have ended
            fixture = raw.get("match_fixture", {})
            new_sa = fixture.get("team_one_score", state.score_a)
            new_sb = fixture.get("team_two_score", state.score_b)
            if new_sa != state.score_a or new_sb != state.score_b:
                state.score_a = new_sa
                state.score_b = new_sb
                winner = state.team_a if new_sa > state.score_a else state.team_b
                logger.info(f"MAP WIN: {winner} | Series: {state.score_a}-{state.score_b}")
                round_events = []  # reset for new map

    # Close remaining positions at final price
    for pos in list(state.open_positions):
        current = state.bid_a if pos["team"] == "a" else state.bid_b
        pnl = pos["stake"] * ((current - pos["entry"]) / pos["entry"])
        state.balance += pos["stake"] + pnl
        state.total_pnl += pnl
        state.trades.append({
            "team": pos["team_name"], "entry": pos["entry"], "exit": current,
            "pnl": round(pnl, 2), "pnl_pct": round((current - pos["entry"]) / pos["entry"] * 100, 1),
            "reason": "end_of_match", "time": state.current_time,
        })
    state.open_positions = []
    state.status = "complete"
    state.progress = 100

    # Summary
    wins = sum(1 for t in state.trades if t["pnl"] > 0)
    losses = len(state.trades) - wins
    avg_latency = sum(state.gemma_latency_ms) / len(state.gemma_latency_ms) if state.gemma_latency_ms else 0
    logger.info(f"\n{'='*60}")
    logger.info(f"REPLAY COMPLETE: {state.team_a} vs {state.team_b}")
    logger.info(f"Trades: {len(state.trades)} ({wins}W/{losses}L)")
    logger.info(f"P&L: ${state.total_pnl:+.2f} | Balance: ${state.balance:.2f}")
    logger.info(f"Gemma calls: {state.gemma_calls} | Avg latency: {avg_latency:.0f}ms")
    logger.info(f"{'='*60}")


# ─── Flask Dashboard ─────────────────────────────────────────────────────

app = Flask(__name__)

DASHBOARD_HTML = """<!DOCTYPE html>
<html><head><title>Replay Test</title>
<style>
body{background:#0a0a0f;color:#e0e0e0;font-family:'Inter',sans-serif;margin:0;padding:10px}
.card{background:#12121f;border:1px solid #1e1e2e;border-radius:6px;padding:12px;margin-bottom:8px}
.green{color:#4ade80}.red{color:#f87171}.yellow{color:#fbbf24}.dim{color:#555}
.big{font-size:24px;font-weight:700}
h2{margin:0 0 8px;font-size:14px;color:#888;text-transform:uppercase}
table{width:100%;border-collapse:collapse;font-size:11px}
th{text-align:left;color:#555;padding:4px}td{padding:4px;border-top:1px solid #1a1a2e}
.tp{background:#16a34a22;color:#4ade80;padding:2px 6px;border-radius:3px;font-size:9px}
.sl{background:#dc262622;color:#f87171;padding:2px 6px;border-radius:3px;font-size:9px}
</style></head><body>
<h1 style="color:#fbbf24;">⚡ REPLAY TEST — Gemma 4 on RTX 5090</h1>
<div style="display:flex;gap:10px;flex-wrap:wrap;">
<div class="card" style="flex:1;min-width:150px"><h2>Match</h2><div id="match" class="big">Loading...</div></div>
<div class="card" style="flex:1;min-width:100px"><h2>Balance</h2><div id="bal" class="big green">$1,000</div></div>
<div class="card" style="flex:1;min-width:100px"><h2>P&L</h2><div id="pnl" class="big">$0</div></div>
<div class="card" style="flex:1;min-width:80px"><h2>Trades</h2><div id="trades" class="big">0</div></div>
<div class="card" style="flex:1;min-width:80px"><h2>Gemma Calls</h2><div id="gcalls" class="big">0</div></div>
<div class="card" style="flex:1;min-width:100px"><h2>Progress</h2><div id="progress" class="big">0%</div></div>
</div>
<div class="card"><h2>Score & Market</h2>
<div id="score" style="font-size:18px;">0-0</div>
<div id="market" class="dim">bid/ask loading...</div>
</div>
<div class="card"><h2>Open Positions</h2><div id="positions">None</div></div>
<div class="card"><h2>Gemma 4 Decisions</h2><div id="decisions" style="max-height:300px;overflow-y:auto;font-size:11px;">Waiting...</div></div>
<div class="card"><h2>Trade History</h2>
<table><thead><tr><th>Time</th><th>Exit</th><th>Team</th><th>Entry</th><th>Exit</th><th>P&L</th><th>Reason</th></tr></thead>
<tbody id="history"></tbody></table>
</div>
<div class="card"><h2>Live Events</h2><div id="events" style="max-height:200px;overflow-y:auto;font-size:10px;font-family:monospace;">Waiting...</div></div>
<script>
function poll(){
fetch('/api/state').then(r=>r.json()).then(d=>{
document.getElementById('match').textContent=d.team_a+' vs '+d.team_b;
document.getElementById('bal').textContent='$'+d.balance.toFixed(2);
document.getElementById('bal').className='big '+(d.total_pnl>=0?'green':'red');
document.getElementById('pnl').textContent=(d.total_pnl>=0?'+':'')+('$'+d.total_pnl.toFixed(2));
document.getElementById('pnl').className='big '+(d.total_pnl>=0?'green':'red');
document.getElementById('trades').textContent=d.trades.length;
document.getElementById('gcalls').textContent=d.gemma_calls;
document.getElementById('progress').textContent=d.progress.toFixed(0)+'% ('+d.status+')';
document.getElementById('score').innerHTML='<span style="font-size:28px;">'+d.round_a+'-'+d.round_b+'</span> <span class="dim">on '+d.map_name+' | Series: '+d.score_a+'-'+d.score_b+'</span>';
document.getElementById('market').innerHTML=d.team_a+': bid='+((d.bid_a*100).toFixed(0))+'c ask='+((d.ask_a*100).toFixed(0))+'c | '+d.team_b+': bid='+((d.bid_b*100).toFixed(0))+'c ask='+((d.ask_b*100).toFixed(0))+'c';
var pos=d.open_positions;
document.getElementById('positions').innerHTML=pos.length?pos.map(p=>'<div>'+p.team_name+' @ '+p.entry.toFixed(3)+' | now: '+(p.current||0).toFixed(3)+' | '+(p.pnl_pct*100).toFixed(1)+'%</div>').join(''):'None';
var decs=d.decisions||[];
document.getElementById('decisions').innerHTML=decs.slice().reverse().map(function(dd){
var c=dd.action.indexOf('buy')>-1?'#4ade80':'#f87171';
return '<div style="padding:3px 0;border-bottom:1px solid #111;"><span class="dim">'+dd.time.substring(11)+'</span> <span style="color:'+c+';font-weight:700;">'+dd.action.toUpperCase()+'</span> '+((dd.confidence*100).toFixed(0))+'% | '+dd.reason.substring(0,80)+' <span class="dim">'+((dd.latency_ms||0).toFixed(0))+'ms</span></div>';
}).join('');
var trades=d.trades||[];
document.getElementById('history').innerHTML=trades.slice().reverse().map(function(t){
var badge=t.reason==='take_profit'?'<span class="tp">TP</span>':'<span class="sl">SL</span>';
var pl='<span class="'+(t.pnl>=0?'green':'red')+'">'+(t.pnl>=0?'+':'')+('$'+t.pnl.toFixed(2))+'</span>';
return '<tr><td class="dim">'+t.time.substring(11)+'</td><td>'+badge+'</td><td>'+t.team+'</td><td>$'+t.entry.toFixed(3)+'</td><td>$'+t.exit.toFixed(3)+'</td><td>'+pl+'</td><td class="dim">'+t.reason+'</td></tr>';
}).join('');
var evts=d.events||[];
document.getElementById('events').innerHTML=evts.slice().reverse().slice(0,30).map(function(e){
return '<div><span class="dim">'+e.time.substring(11)+'</span> '+e.desc+'</div>';
}).join('');
}).catch(function(){});
}
poll();setInterval(poll,500);
</script></body></html>"""


@app.route("/")
def index():
    return DASHBOARD_HTML

@app.route("/api/state")
def api_state():
    return jsonify({
        "team_a": state.team_a, "team_b": state.team_b,
        "map_name": state.map_name,
        "score_a": state.score_a, "score_b": state.score_b,
        "round_a": state.round_a, "round_b": state.round_b,
        "economy_a": state.economy_a, "economy_b": state.economy_b,
        "bid_a": state.bid_a, "ask_a": state.ask_a,
        "bid_b": state.bid_b, "ask_b": state.ask_b,
        "balance": state.balance, "total_pnl": state.total_pnl,
        "gemma_calls": state.gemma_calls,
        "events": state.events[-50:],
        "decisions": state.decisions[-30:],
        "trades": state.trades,
        "open_positions": state.open_positions,
        "progress": state.progress,
        "status": state.status,
    })


def run_dashboard(port):
    from waitress import serve
    serve(app, host="0.0.0.0", port=port, threads=2, _quiet=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("file", help="JSONL file to replay")
    parser.add_argument("--port", type=int, default=8083)
    parser.add_argument("--speed", type=int, default=50, help="Speed multiplier (default: 50x)")
    args = parser.parse_args()

    global SPEED_MULTIPLIER
    SPEED_MULTIPLIER = args.speed

    # Start dashboard
    import threading
    threading.Thread(target=run_dashboard, args=(args.port,), daemon=True).start()
    logger.info(f"Replay dashboard: http://localhost:{args.port}")

    await asyncio.sleep(1)
    await replay_match(args.file)

    # Keep running for dashboard
    logger.info("Replay complete — dashboard still running. Ctrl+C to exit.")
    while True:
        await asyncio.sleep(1)


if __name__ == "__main__":
    asyncio.run(main())
