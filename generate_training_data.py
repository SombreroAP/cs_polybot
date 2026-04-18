#!/usr/bin/env python3
"""
Generate training data from our trade history + match events.

Builds prompt→response pairs in the EXACT format our bot uses,
labeled with actual outcomes (win/loss) so the fine-tuned model
learns what ACTUALLY works vs what doesn't.

Output: data/training/prompt_format_training.jsonl
Each line: {"input": "<our exact prompt>", "output": "<correct JSON response>"}
"""
import json
import os
import sys
import sqlite3
import glob
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DB_PATH = "data/trades.db"
RECORDINGS_DIR = "data/recordings"
OUTPUT_DIR = "data/training"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_trades():
    """Load all resolved trades with outcomes."""
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT order_id, team, game, amount, fill_price, sell_price, pnl,
               exit_reason, trigger_event, analysis, market_price_at_entry,
               our_price_at_entry, timestamp
        FROM positions WHERE resolved = 1 AND pnl != 0
        ORDER BY timestamp
    """).fetchall()
    db.close()
    return [dict(r) for r in rows]

def load_match_events():
    """Load match events grouped by match_id."""
    db = sqlite3.connect(DB_PATH, timeout=30)
    db.row_factory = sqlite3.Row
    rows = db.execute("""
        SELECT match_id, game, event_type, team, description,
               team_a, team_b, score_a, score_b, round_a, round_b,
               kills_a, kills_b, gold_lead, economy_a, economy_b,
               market_price, timestamp
        FROM match_events ORDER BY timestamp
    """).fetchall()
    db.close()

    events = defaultdict(list)
    for r in rows:
        events[r["match_id"]].append(dict(r))
    return events

def load_recordings():
    """Load price data from recordings."""
    prices = {}  # match_id -> [{bid, ask, ts}, ...]
    for f in glob.glob(f"{RECORDINGS_DIR}/*.jsonl"):
        match_id = os.path.basename(f).split("_")[0]
        try:
            for line in open(f):
                d = json.loads(line)
                if d["message_type"] == "_PM_best_bid_ask":
                    raw = d["raw"]
                    bid = raw.get("bestBid", 0)
                    ask = raw.get("bestAsk", 0)
                    if bid > 0.05 and ask < 0.95:  # skip dust
                        if match_id not in prices:
                            prices[match_id] = []
                        prices[match_id].append({
                            "bid": bid, "ask": ask, "ts": d["ts"]
                        })
        except Exception:
            pass
    return prices

def build_prompt(trade, events, prices):
    """Build the EXACT same prompt format our bot sends."""
    game = trade["game"].upper()
    team = trade["team"]
    fill = trade["fill_price"]
    mkt = trade["market_price_at_entry"]
    our = trade["our_price_at_entry"]

    # Find matching events near trade time
    nearby = []
    for mid, evts in events.items():
        for e in evts:
            if abs(e["timestamp"] - trade["timestamp"]) < 120:  # within 2 min
                if e["team_a"] and (team.lower() in e["team_a"].lower() or
                    team.lower() in e["team_b"].lower()):
                    nearby.append(e)

    nearby.sort(key=lambda e: e["timestamp"])

    lines = []
    lines.append(f"GAME: {game} | {nearby[0]['team_a'] if nearby else '?'} vs {nearby[0]['team_b'] if nearby else '?'}")

    if nearby:
        e = nearby[-1]  # latest event
        if e["round_a"] or e["round_b"]:
            lines.append(f"Series: {e['score_a']}-{e['score_b']} | Map/Round: {e['round_a']}-{e['round_b']}")
        if e["economy_a"] or e["economy_b"]:
            lines.append(f"Economy: {e['team_a']}=${e['economy_a']:,} | {e['team_b']}=${e['economy_b']:,}")
        if game in ("DOTA2", "LOL") and (e["kills_a"] or e["kills_b"]):
            lines.append(f"Kills: {e['team_a']} {e['kills_a']} - {e['kills_b']} {e['team_b']}")
            if e["gold_lead"]:
                gold_team = e['team_a'] if e['gold_lead'] > 0 else e['team_b']
                lines.append(f"GOLD LEAD: {gold_team} +{abs(e['gold_lead']):,}")

    lines.append(f"MARKET: Price: {mkt*100:.0f}c | Spread: {abs(fill-mkt)*100:.0f}%")

    if fill > 0:
        lines.append(f"Orderbook: bid={mkt*100:.0f}c ask={fill*100:.0f}c")

    # Events
    if nearby:
        lines.append(f"Events({len(nearby)}):")
        for e in nearby[-5:]:
            lines.append(f"- {e['description'][:50]}")

    lines.append(f"\nBUY {team}? Reply with ONLY this JSON, nothing else:")
    lines.append('{"action":"buy or skip","confidence":0.0-1.0,"reason":"one sentence"}')

    return "\n".join(lines)

def build_response(trade):
    """Build the CORRECT response based on actual outcome."""
    won = trade["pnl"] > 0
    pnl_pct = abs(trade["pnl"] / trade["amount"] * 100) if trade["amount"] > 0 else 0

    if won:
        # Trade was profitable — correct action was BUY
        conf = min(0.95, 0.6 + pnl_pct / 100)  # higher P&L = higher confidence
        reason = trade.get("trigger_event", "")
        if reason:
            # Extract reason from trigger text
            parts = reason.split("|")
            reason = parts[1].strip()[:60] if len(parts) >= 2 else reason[:60]
        else:
            reason = f"Profitable trade: +{pnl_pct:.0f}% in {trade['exit_reason']}"
        return json.dumps({
            "action": "buy",
            "confidence": round(conf, 2),
            "reason": reason
        })
    else:
        # Trade lost — correct action was SKIP
        if pnl_pct > 50:
            reason = "Token resolved or massive price crash — should have skipped"
        elif pnl_pct > 20:
            reason = "Large loss from spread or market reversal — risk too high"
        elif trade["fill_price"] > 0.75:
            reason = "High price entry with limited upside — should skip above 75c"
        else:
            reason = f"Market moved against: -{pnl_pct:.0f}% loss"
        return json.dumps({
            "action": "skip",
            "confidence": round(min(0.9, 0.5 + pnl_pct / 200), 2),
            "reason": reason
        })

def generate_from_recordings(prices, events):
    """Generate additional training pairs from price movement data."""
    pairs = []

    for mid, price_list in prices.items():
        if len(price_list) < 10:
            continue

        # Find events for this match
        match_events = events.get(mid, [])
        if not match_events:
            continue

        for evt in match_events:
            if evt["event_type"] not in ("round_end", "teamfight_won", "kill_streak", "map_win"):
                continue

            evt_ts = evt["timestamp"]

            # Find price BEFORE event
            pre_prices = [p for p in price_list if p["ts"] < evt_ts and p["ts"] > evt_ts - 30]
            # Find price AFTER event
            post_prices = [p for p in price_list if p["ts"] > evt_ts and p["ts"] < evt_ts + 60]

            if not pre_prices or not post_prices:
                continue

            pre_bid = pre_prices[-1]["bid"]
            post_bid = post_prices[-1]["bid"]
            price_move = (post_bid - pre_bid) / pre_bid * 100 if pre_bid > 0 else 0

            # Build prompt from event data
            game = evt["game"].upper()
            lines = [f"GAME: {game} | {evt['team_a']} vs {evt['team_b']}"]
            if evt["round_a"] or evt["round_b"]:
                lines.append(f"Round: {evt['round_a']}-{evt['round_b']}")
            if evt["economy_a"]:
                lines.append(f"Economy: {evt['team_a']}=${evt['economy_a']:,} | {evt['team_b']}=${evt['economy_b']:,}")
            if evt["kills_a"] or evt["kills_b"]:
                lines.append(f"Kills: {evt['kills_a']}-{evt['kills_b']}")
            if evt["gold_lead"]:
                lines.append(f"Gold lead: {evt['gold_lead']:+,}")

            pre_ask = pre_prices[-1]["ask"]
            spread = pre_ask - pre_bid
            lines.append(f"Orderbook: bid={pre_bid*100:.0f}c ask={pre_ask*100:.0f}c spr={spread*100:.0f}%")
            lines.append(f"Events(1):\n- {evt['description'][:50]}")

            buy_team = evt["team_a"] if evt["team"] == "a" else evt["team_b"] if evt["team"] == "b" else evt["team_a"]
            lines.append(f"\nBUY {buy_team}? Reply with ONLY this JSON, nothing else:")
            lines.append('{"action":"buy or skip","confidence":0.0-1.0,"reason":"one sentence"}')

            prompt = "\n".join(lines)

            # Label based on actual price movement
            if price_move > 3:
                response = json.dumps({
                    "action": "buy",
                    "confidence": min(0.95, 0.6 + price_move / 50),
                    "reason": f"Price moved +{price_move:.1f}% after event — profitable buy"
                })
            elif price_move < -3:
                response = json.dumps({
                    "action": "skip",
                    "confidence": min(0.9, 0.5 + abs(price_move) / 50),
                    "reason": f"Price dropped {price_move:.1f}% after event — skip"
                })
            else:
                response = json.dumps({
                    "action": "skip",
                    "confidence": 0.6,
                    "reason": f"Price barely moved ({price_move:+.1f}%) — not enough edge"
                })

            pairs.append({"input": prompt, "output": response})

    return pairs

def main():
    print("=== GENERATING TRAINING DATA ===\n")

    # Load data
    trades = load_trades()
    events = load_match_events()
    prices = load_recordings()

    print(f"Trades: {len(trades)}")
    print(f"Match events: {sum(len(v) for v in events.values()):,} across {len(events)} matches")
    print(f"Price recordings: {sum(len(v) for v in prices.values()):,} across {len(prices)} matches")

    # Generate from trades
    trade_pairs = []
    for trade in trades:
        try:
            prompt = build_prompt(trade, events, prices)
            response = build_response(trade)
            trade_pairs.append({"input": prompt, "output": response})
        except Exception as e:
            pass

    print(f"\nTrade-based pairs: {len(trade_pairs)}")
    wins = sum(1 for t in trades if t["pnl"] > 0)
    losses = len(trades) - wins
    print(f"  Wins: {wins} → 'buy' examples")
    print(f"  Losses: {losses} → 'skip' examples")

    # Generate from recordings + events
    recording_pairs = generate_from_recordings(prices, events)
    print(f"Recording-based pairs: {len(recording_pairs)}")

    # Combine
    all_pairs = trade_pairs + recording_pairs
    print(f"\nTOTAL training pairs: {len(all_pairs)}")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "prompt_format_training.jsonl")
    with open(output_path, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    size = os.path.getsize(output_path)
    print(f"Saved to: {output_path} ({size/1e6:.1f}MB)")

    # Show samples
    print("\n=== SAMPLE BUY EXAMPLE ===")
    buy_ex = next((p for p in trade_pairs if '"buy"' in p["output"]), None)
    if buy_ex:
        print(f"Input:\n{buy_ex['input'][:200]}...")
        print(f"Output: {buy_ex['output']}")

    print("\n=== SAMPLE SKIP EXAMPLE ===")
    skip_ex = next((p for p in trade_pairs if '"skip"' in p["output"]), None)
    if skip_ex:
        print(f"Input:\n{skip_ex['input'][:200]}...")
        print(f"Output: {skip_ex['output']}")

if __name__ == "__main__":
    main()
