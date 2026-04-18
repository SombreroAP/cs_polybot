#!/usr/bin/env python3
"""
Replay recorded bo3.gg match through the NEW rich-data pipeline.

Feeds snapshots one-by-one into CS2Bo3Feed._process_snapshot (which now emits
KILL/OPEN_KILL/HP_DAMAGE/CLUTCH_WIN events and stashes alive/HP/damage/buy-type
on match_state.extra).

At key moments — every ROUND_END, every MAP_WIN, every OPEN_KILL, every
CLUTCH_WIN, and at a configurable sample rate of regular snapshots — we call
EdgeAnalyst.should_buy() with the current rich state + the most recent real
Polymarket orderbook from the recording. Prints qwen's reply.

Usage:
    python3 replay_rich.py "Test Data/117218_Fire_Flux_vs_Lph_Gaming_converted.jsonl"
    python3 replay_rich.py <jsonl> --sample-every 30
"""
import argparse
import asyncio
import json
import logging
import os
import sys
import time
from collections import defaultdict

# Silence noisy feed + market discovery logs; keep our own output readable
logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                    datefmt="%H:%M:%S")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from feeds.cs2_bo3 import CS2Bo3Feed
from feeds.base import GameEvent, EventType, MatchState
from edge_analyst import EdgeAnalyst


TRIGGER_EVENTS = {
    EventType.ROUND_END,
    EventType.MAP_WIN,
    EventType.OPEN_KILL,
    EventType.CLUTCH_WIN,
    EventType.KILL_STREAK,
    EventType.MATCH_END,
}


async def replay(jsonl_path: str, sample_every: int, limit_calls: int, live_speed: bool):
    print(f"\n=== REPLAY: {jsonl_path} ===\n")

    feed = CS2Bo3Feed()
    analyst = EdgeAnalyst()

    # Collect events instead of firing them to real callbacks
    emitted_events: list[GameEvent] = []
    def collector(ev: GameEvent):
        emitted_events.append(ev)
    feed.on_event(collector)

    # Parse META + orderbook + snapshots
    match_id: str = ""
    team_a: str = ""
    team_b: str = ""
    token_map: dict = {}
    # latest bid/ask per token
    books: dict[str, dict] = {}
    # Which token is "team A" for the Match Winner market
    team_a_token: str = ""
    team_b_token: str = ""

    snapshots_read = 0
    books_read = 0
    qwen_calls = 0

    with open(jsonl_path) as fh:
        lines = [l for l in fh if l.strip()]

    print(f"{len(lines)} messages in file. First pass reading META + orderbooks...")

    # ─── First pass: extract META + build token→team map ───────────────────
    for raw in lines:
        try:
            d = json.loads(raw)
        except Exception:
            continue
        mt = d.get("message_type", "")
        r = d.get("raw", {}) or {}
        if mt == "_META":
            match_id = str(r.get("match_id", ""))
            team_a = r.get("team1", "")
            team_b = r.get("team2", "")
            token_map = r.get("token_map", {})
            for tok, info in token_map.items():
                if info.get("market_type") == "Match Winner":
                    if info.get("outcome_index") == 0:
                        team_a_token = tok
                    elif info.get("outcome_index") == 1:
                        team_b_token = tok
            print(f"MATCH {match_id}: {team_a} vs {team_b}")
            print(f"  team_a token ({team_a}): {team_a_token[:20]}...")
            print(f"  team_b token ({team_b}): {team_b_token[:20]}...")
            break

    if not match_id:
        print("No _META found — cannot replay.")
        return

    # Subscribe match so feed state exists
    await feed.subscribe_match(match_id)

    # Now do the real replay
    print("\n--- Streaming snapshots + orderbook updates ---\n")
    last_decision_ts = 0.0

    for raw in lines:
        try:
            d = json.loads(raw)
        except Exception:
            continue
        mt = d.get("message_type", "")
        r = d.get("raw", {}) or {}

        # Orderbook updates
        if mt in ("_PM_book", "_PM_best_bid_ask", "_PRICE_SNAPSHOT"):
            tok = r.get("tokenId") or r.get("token_id")
            if not tok:
                continue
            bid = r.get("bestBid") if r.get("bestBid") is not None else r.get("bid")
            ask = r.get("bestAsk") if r.get("bestAsk") is not None else r.get("ask")
            if bid is None and ask is None:
                continue
            books.setdefault(tok, {})
            if bid is not None: books[tok]["bid"] = float(bid)
            if ask is not None: books[tok]["ask"] = float(ask)
            books[tok]["ts"] = d.get("ts", 0)
            books_read += 1
            continue

        # Match snapshot
        if mt == "SNAPSHOT_MATCH_UPDATE":
            snapshots_read += 1
            # Feed the snapshot into the feed — this now emits rich events
            before = len(emitted_events)
            feed._process_snapshot(match_id, r)
            new_events = emitted_events[before:]

            # Print new events with compact formatting
            for ev in new_events:
                if ev.event_type == EventType.OPEN_KILL or ev.event_type == EventType.CLUTCH_WIN or ev.event_type == EventType.MAP_WIN:
                    tag = "★"
                elif ev.event_type in (EventType.KILL, EventType.HP_DAMAGE):
                    tag = " "
                else:
                    tag = "·"
                desc = ev.description[:70]
                print(f"  {tag} [{ev.event_type.value:14}] team={ev.team} | {desc}")

            # Decide: call qwen?
            should_call = False
            reason = ""
            for ev in new_events:
                if ev.event_type in TRIGGER_EVENTS:
                    should_call = True
                    reason = f"on {ev.event_type.value}"
                    break
            # Sample rate fallback
            if not should_call and sample_every > 0 and snapshots_read % sample_every == 0:
                should_call = True
                reason = f"sample (every {sample_every})"

            if should_call and qwen_calls < limit_calls:
                # Build game_state + market_state exactly like latency.py does
                state = feed.get_match_state(match_id)
                if not state:
                    continue
                ex = state.extra or {}

                # Determine buy team (favor the team trending up)
                # For simplicity: buy whichever team's win_prob > 0.5 per the feed model
                prob_a = state.win_probability_a
                if prob_a >= 0.5:
                    buy_team_letter = "a"
                    buy_token = team_a_token
                    buy_name = team_a
                else:
                    buy_team_letter = "b"
                    buy_token = team_b_token
                    buy_name = team_b

                book = books.get(buy_token, {})
                if not book or not book.get("bid"):
                    continue  # no market data yet — skip
                bid = float(book.get("bid") or 0)
                ask = float(book.get("ask") or 0)
                mid = (bid + ask) / 2 if (bid and ask) else (bid or ask)
                spread = max(0.0, ask - bid) if (ask > bid) else 0.0

                # skip if market is obviously untradable
                if ask <= 0 or bid <= 0 or ask >= 0.98 or spread > 0.3:
                    continue

                game_state = {
                    "game": "cs2",
                    "team_a": state.team_a,
                    "team_b": state.team_b,
                    "score_a": state.score_a,
                    "score_b": state.score_b,
                    "round_a": state.round_score_a,
                    "round_b": state.round_score_b,
                    "buy_team": buy_name,
                    "economy_a": ex.get("team_a_money", 0),
                    "economy_b": ex.get("team_b_money", 0),
                    "side": ex.get("team_a_current_side", ""),
                    "round_kills_a": ex.get("round_kills_a", 0),
                    "round_kills_b": ex.get("round_kills_b", 0),
                    "model_price": prob_a if buy_team_letter == "a" else 1 - prob_a,
                    "alive_a": ex.get("alive_a", 5),
                    "alive_b": ex.get("alive_b", 5),
                    "avg_hp_a": ex.get("avg_hp_a", 100),
                    "avg_hp_b": ex.get("avg_hp_b", 100),
                    "damage_a": ex.get("damage_a", 0),
                    "damage_b": ex.get("damage_b", 0),
                    "has_awp_a": bool(ex.get("has_awp_a", False)),
                    "has_awp_b": bool(ex.get("has_awp_b", False)),
                    "has_defuse_kit_a": bool(ex.get("has_defuse_kit_a", False)),
                    "bomb_carrier": ex.get("bomb_carrier", ""),
                    "buy_type_a": ex.get("buy_type_a", ""),
                    "buy_type_b": ex.get("buy_type_b", ""),
                    "map_name": ex.get("map_name", ""),
                }

                market_state = {
                    "price": mid, "bid": bid, "ask": ask, "spread": spread,
                    "volume": 10000, "liquidity": 5000, "market_type": "series",
                    "question": f"{state.team_a} vs {state.team_b}",
                    "momentum_10": {}, "momentum_30": {}, "momentum_60": {},
                    "staleness_seconds": 0,
                    "balance": 1000, "open_positions": 0,
                }
                window = [
                    {"description": e.description[:80], "type": e.event_type.value,
                     "team": e.team}
                    for e in new_events[:5]
                ]

                header = (
                    f"\n[{snapshots_read:4d}] series {state.score_a}-{state.score_b} "
                    f"rd {state.round_score_a}-{state.round_score_b} | "
                    f"alive {ex.get('alive_a',5)}v{ex.get('alive_b',5)} | "
                    f"hp {ex.get('avg_hp_a',100)}/{ex.get('avg_hp_b',100)} | "
                    f"buy_a={ex.get('buy_type_a','?')} buy_b={ex.get('buy_type_b','?')} | "
                    f"ask_{buy_name}={ask:.2f} | {reason}"
                )
                print(header)

                qwen_calls += 1
                t0 = time.time()
                try:
                    decision = await analyst.should_buy(window, game_state, market_state)
                except Exception as e:
                    print(f"  QWEN ERROR: {e}")
                    continue
                dt_ms = int((time.time() - t0) * 1000)

                if decision is None:
                    print(f"  → QWEN SKIPPED (cached/limited)  {dt_ms}ms")
                else:
                    action = decision.action.upper()
                    marker = "✅ BUY" if action == "BUY" else "⛔ SKIP"
                    print(f"  → {marker} | conf={decision.confidence:.2f}  bet=${decision.bet_size:.0f}  "
                          f"tp={decision.tp_pct*100:.0f}%  sl={decision.sl_pct*100:.0f}%  {dt_ms}ms")
                    print(f"     reason: {decision.reason}")

                if qwen_calls >= limit_calls:
                    print(f"\n[LIMIT] Reached {limit_calls} qwen calls — stopping.")
                    break

            if live_speed:
                await asyncio.sleep(0.05)

    print(f"\n=== REPLAY DONE ===")
    print(f"Snapshots: {snapshots_read}")
    print(f"Orderbook updates: {books_read}")
    print(f"Events emitted: {len(emitted_events)}")
    # Event type distribution
    dist = defaultdict(int)
    for e in emitted_events:
        dist[e.event_type.value] += 1
    for k in sorted(dist.keys()):
        print(f"  {k:16} = {dist[k]}")
    print(f"Qwen calls: {qwen_calls}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("jsonl", help="recorded match jsonl file")
    p.add_argument("--sample-every", type=int, default=0,
                   help="call qwen every N snapshots in addition to trigger events (0 = off)")
    p.add_argument("--limit", type=int, default=12, help="max qwen calls (for cost/time)")
    p.add_argument("--live", action="store_true", help="throttle to near-real-time")
    args = p.parse_args()
    asyncio.run(replay(args.jsonl, args.sample_every, args.limit, args.live))


if __name__ == "__main__":
    main()
