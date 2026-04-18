#!/usr/bin/env python3
"""
Prepare Training Data for Gemma 4 Fine-Tuning.

Reads JSONL match recordings (Test Data/ and data/recordings/) and generates
training pairs: game_state → correct_action based on what ACTUALLY happened
to the price after each event.

Output: data/training/training_data.jsonl (ready for Unsloth/QLoRA)

Usage:
    python prepare_training_data.py              # process all files
    python prepare_training_data.py --dir "Test Data"  # specific folder
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "training")


def process_match(filepath: str) -> list:
    """Process a single match file into training pairs."""
    training_pairs = []

    # Read all events
    events = []
    with open(filepath) as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    if len(events) < 10:
        return []

    # Extract metadata
    meta = {}
    for e in events:
        if e["message_type"] == "_META":
            meta = e["raw"]
            break

    team_a = meta.get("team1", "Team A")
    team_b = meta.get("team2", "Team B")
    game = meta.get("game", "cs2")

    # Build timeline: timestamp → (game_state, price)
    price_timeline = []  # [(ts, bid_a, ask_a)]
    game_events = []     # [(ts, event_type, raw)]

    for e in events:
        ts = e["ts"]
        msg = e["message_type"]
        raw = e["raw"]

        if msg in ("_PM_best_bid_ask", "_PM_book"):
            bid = raw.get("bestBid", 0)
            ask = raw.get("bestAsk", 0)
            if bid > 0:
                price_timeline.append((ts, bid, ask))

        elif msg in ("GAME_EVENT_MATCH_END_ROUND", "SNAPSHOT_MATCH_UPDATE",
                      "GAME_EVENT_PLAYER_KILL", "GAME_EVENT_TEAMS_ECONOMY_INFO"):
            game_events.append((ts, msg, raw))

    if not price_timeline or not game_events:
        return []

    # For each game event, find what happened to the price in the next 30-90 seconds
    for i, (evt_ts, evt_type, evt_raw) in enumerate(game_events):
        # Skip non-actionable events
        if evt_type == "GAME_EVENT_PLAYER_KILL":
            continue  # individual kills are too noisy for training

        # Find price at event time
        price_at_event = None
        for ts, bid, ask in price_timeline:
            if ts <= evt_ts:
                price_at_event = (bid, ask)
            else:
                break

        if not price_at_event:
            continue

        bid_now, ask_now = price_at_event

        # Skip if price is extreme (resolved/illiquid)
        if bid_now < 0.10 or bid_now > 0.90 or ask_now <= 0:
            continue

        # Find price 30s and 60s later
        price_30s = None
        price_60s = None
        for ts, bid, ask in price_timeline:
            if ts > evt_ts + 25 and price_30s is None:
                price_30s = bid
            if ts > evt_ts + 55 and price_60s is None:
                price_60s = bid
            if price_30s and price_60s:
                break

        if price_30s is None:
            continue

        # Determine correct action based on what ACTUALLY happened
        move_30s = (price_30s - bid_now) / bid_now * 100
        move_60s = ((price_60s - bid_now) / bid_now * 100) if price_60s else move_30s

        # Extract game state from event
        t1 = evt_raw.get("team_one", {})
        t2 = evt_raw.get("team_two", {})
        score_a = t1.get("score", 0) or 0
        score_b = t2.get("score", 0) or 0
        series_a = t1.get("match_score", 0) or 0
        series_b = t2.get("match_score", 0) or 0
        eco_a = t1.get("equipment_value", 0) or 0
        eco_b = t2.get("equipment_value", 0) or 0
        map_name = evt_raw.get("map_name", "")

        spread = ask_now - bid_now
        spread_pct = spread / ask_now * 100 if ask_now > 0 else 0

        # Build the prompt (what the model would see)
        prompt_lines = [
            f"GAME: {game.upper()} | {team_a} vs {team_b}",
            f"Map: {map_name} | Round: {score_a}-{score_b} | Series: {series_a}-{series_b}",
        ]
        if eco_a or eco_b:
            prompt_lines.append(f"Economy: {team_a}=${eco_a:,} | {team_b}=${eco_b:,}")
        prompt_lines.append(f"Market: bid={bid_now*100:.0f}c ask={ask_now*100:.0f}c spread={spread_pct:.0f}%")
        prompt_lines.append(f"Should we BUY {team_a}?")
        prompt = "\n".join(prompt_lines)

        # Determine correct answer
        # If price moved up >5% in 30-60s → should have bought team_a
        # If price moved down >5% → should have bought team_b (or skipped)
        # Otherwise → skip (no clear edge)
        if move_30s > 5 or move_60s > 7:
            correct_action = "buy_a"
            correct_conf = min(0.95, 0.7 + move_60s / 50)
            correct_reason = f"Price moved +{move_60s:.1f}% in 60s — clear upward momentum"
        elif move_30s < -5 or move_60s < -7:
            correct_action = "buy_b"
            correct_conf = min(0.95, 0.7 + abs(move_60s) / 50)
            correct_reason = f"Price dropped {abs(move_60s):.1f}% in 60s — downward momentum, bet opponent"
        elif spread_pct > 12:
            correct_action = "skip"
            correct_conf = 0.8
            correct_reason = f"Spread {spread_pct:.0f}% too wide — execution cost exceeds potential gain"
        else:
            correct_action = "skip"
            correct_conf = 0.6
            correct_reason = f"Price moved only {move_60s:+.1f}% — no clear edge"

        response = json.dumps({
            "action": correct_action,
            "confidence": round(correct_conf, 2),
            "reason": correct_reason,
        })

        training_pairs.append({
            "prompt": prompt,
            "response": response,
            "metadata": {
                "match": os.path.basename(filepath),
                "event_type": evt_type,
                "price_move_30s": round(move_30s, 2),
                "price_move_60s": round(move_60s, 2),
                "bid_at_event": bid_now,
                "spread_pct": round(spread_pct, 1),
            }
        })

    return training_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default=None, help="Specific directory to process")
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all JSONL files
    files = []
    if args.dir:
        files = glob.glob(os.path.join(args.dir, "*.jsonl"))
    else:
        files = glob.glob("Test Data/*.jsonl") + glob.glob("data/recordings/*.jsonl")

    print(f"Processing {len(files)} match files...")

    all_pairs = []
    for filepath in files:
        name = os.path.basename(filepath)
        pairs = process_match(filepath)
        if pairs:
            all_pairs.extend(pairs)
            buys = sum(1 for p in pairs if "buy" in p["response"])
            skips = len(pairs) - buys
            print(f"  {name[:50]:50s} → {len(pairs)} pairs ({buys} buy, {skips} skip)")

    # Write training data
    output_file = os.path.join(OUTPUT_DIR, "training_data.jsonl")
    with open(output_file, "w") as f:
        for pair in all_pairs:
            f.write(json.dumps(pair) + "\n")

    # Also write in Unsloth/Alpaca format for direct fine-tuning
    alpaca_file = os.path.join(OUTPUT_DIR, "alpaca_format.jsonl")
    system_prompt = "You are an esports latency arbitrage trader. Analyze the game state and reply with JSON: {\"action\":\"buy_a/buy_b/skip\",\"confidence\":0.0-1.0,\"reason\":\"brief\"}"
    with open(alpaca_file, "w") as f:
        for pair in all_pairs:
            alpaca = {
                "instruction": system_prompt,
                "input": pair["prompt"],
                "output": pair["response"],
            }
            f.write(json.dumps(alpaca) + "\n")

    # Stats
    buys = sum(1 for p in all_pairs if "buy" in p["response"])
    skips = len(all_pairs) - buys
    avg_move = sum(abs(p["metadata"]["price_move_60s"]) for p in all_pairs) / len(all_pairs) if all_pairs else 0

    print(f"\n{'='*60}")
    print(f"TRAINING DATA READY")
    print(f"{'='*60}")
    print(f"Total pairs: {len(all_pairs)}")
    print(f"  Buy signals: {buys} ({buys/len(all_pairs)*100:.0f}%)")
    print(f"  Skip signals: {skips} ({skips/len(all_pairs)*100:.0f}%)")
    print(f"Avg price move: {avg_move:.1f}%")
    print(f"\nOutput files:")
    print(f"  {output_file}")
    print(f"  {alpaca_file}")
    print(f"\nTo fine-tune on your RTX 5090:")
    print(f"  1. Copy {alpaca_file} to your Windows PC")
    print(f"  2. pip install unsloth")
    print(f"  3. Run the fine-tuning script (see finetune_gemma4.py)")


if __name__ == "__main__":
    main()
