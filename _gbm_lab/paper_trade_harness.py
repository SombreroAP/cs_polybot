"""
Paper-trading harness for the drift-conditioned MM strategy.

Subscribes to Polymarket's WebSocket book updates for one or more tokens,
runs the MMStrategy on each book update, and LOGS what orders it would
have placed/cancelled. Does NOT actually place orders.

Designed to be a confidence-builder before going live: run this for a day,
compare logged "would-have-traded" decisions to actual book transitions,
and verify the strategy behaves sensibly in real time.

Usage (after Polymarket order placement is wired):
    pip install py-clob-client websocket-client
    python paper_trade_harness.py --tokens TOKEN_ID_A TOKEN_ID_B

This file is intentionally a skeleton — the user / future Claude session
should wire the actual WebSocket connection and confirm strategy behaviour
before swapping the logger for live order placement.
"""
import argparse
import json
import logging
import os
import signal
import sys
import time
from dataclasses import asdict
from typing import Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_strategy_production import MMConfig, MMStrategy


# ─── Logging setup ─────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("paper_trade")


# ─── Per-token strategies registry ─────────────────────────────────────────
strategies: Dict[str, MMStrategy] = {}


def get_or_create_strategy(token_id: str, cfg: MMConfig) -> MMStrategy:
    if token_id not in strategies:
        strategies[token_id] = MMStrategy(token_id=token_id, cfg=cfg)
        log.info(f"[NEW] strategy for token {token_id[:16]}…")
    return strategies[token_id]


# ─── Book-update handler ───────────────────────────────────────────────────
def on_book_update(token_id: str, best_bid: float, best_ask: float, ts: float):
    """Called for every Polymarket book event we observe."""
    strat = get_or_create_strategy(token_id, MMConfig())
    decision = strat.on_book_update(best_bid, best_ask, ts)
    if decision is None:
        return
    # Log the would-be action. In real version, this would call the CLOB client.
    if decision["action"] == "cancel_all":
        log.info(f"[{token_id[:8]}] CANCEL_ALL (reason={decision.get('reason')})  "
                 f"bb={best_bid:.4f} ba={best_ask:.4f}")
    elif decision["action"] == "cancel_and_replace":
        log.info(f"[{token_id[:8]}] QUOTE  "
                 f"bb={best_bid:.4f}→our_bid={decision.get('new_bid_price')}  "
                 f"ba={best_ask:.4f}→our_ask={decision.get('new_ask_price')}  "
                 f"size={decision['size']}  inv={strat.inventory}")


# ─── Trade-event handler (for tracking fills if WS provides them) ─────────
def on_fill(token_id: str, side: str, price: float, size: float = 1.0, ts: float = 0):
    strat = strategies.get(token_id)
    if not strat:
        log.warning(f"Fill on unknown token {token_id}")
        return
    strat.on_fill(side, price, size, ts)
    log.info(f"[{token_id[:8]}] FILL {side} @ {price:.4f}  inv={strat.inventory}  cash={strat.cash:.4f}")


# ─── Stats reporter (periodic) ────────────────────────────────────────────
def print_stats():
    log.info("=" * 60)
    log.info(f"Paper-trade stats: {len(strategies)} tokens tracked")
    total_fills = sum(s.n_fills for s in strategies.values())
    total_cash = sum(s.cash for s in strategies.values())
    log.info(f"  total fills: {total_fills}  total cash (excl inv): ${total_cash:+.4f}")
    if total_fills > 0:
        log.info(f"  avg per fill: {total_cash / total_fills * 100:+.2f}¢")
    # Top-3 active
    for s in sorted(strategies.values(), key=lambda x: -x.n_fills)[:3]:
        log.info(f"  {s.token_id[:12]}: {s.n_fills} fills  cash={s.cash:+.4f}  inv={s.inventory}")


# ─── WebSocket wiring stub ────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", nargs="+",
                    help="Token IDs to subscribe to (one per token)")
    ap.add_argument("--mock-replay",
                    help="Path to a processed/*.jsonl file to replay instead of live WS")
    ap.add_argument("--stats-every", type=int, default=60,
                    help="Print stats every N seconds")
    args = ap.parse_args()

    if args.mock_replay:
        # Replay mode — useful for verifying strategy behaviour without live WS
        log.info(f"Mock-replay mode: {args.mock_replay}")
        last_stats = time.time()
        with open(args.mock_replay) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") == "book":
                    tok = r.get("tokenId")
                    bid = r.get("bid")
                    ask = r.get("ask")
                    ts = r.get("ts")
                    if tok and bid is not None and ask is not None:
                        on_book_update(tok, float(bid), float(ask), ts)
                if time.time() - last_stats > args.stats_every:
                    print_stats()
                    last_stats = time.time()
        print_stats()
        return

    # Live WS mode (skeleton — needs py-clob-client / websocket-client wired)
    log.warning("Live WebSocket mode not yet implemented.")
    log.warning("To run paper trade live:")
    log.warning("  1. pip install py-clob-client websocket-client")
    log.warning("  2. Use the official Polymarket WSS endpoint")
    log.warning("  3. For each price_change message, call on_book_update()")
    log.warning("  4. For each trade message that includes our orders, call on_fill()")
    log.warning("For now, use --mock-replay <processed-file.jsonl> to test.")


def handle_sigint(_sig, _frame):
    print_stats()
    log.info("Bye.")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, handle_sigint)
    main()
