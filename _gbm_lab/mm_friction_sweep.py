"""Slippage-sensitivity sweep on the best MM strategy.

If real Polygon gas + friction on Polymarket CLOB is much lower than my
conservative 0.3¢/share assumption, the best strategy is already net positive.
This script reruns the best config across friction values 0–0.5¢/share and
reports the break-even point.
"""
import os, sys, glob, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collections import defaultdict
from mm_event_aware import parse_match_file, simulate_with_event_cooldown


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_events = {}
    all_books = {}
    for p in paths:
        ev, bk = parse_match_file(p)
        if ev or bk:
            all_events[p] = ev
            all_books[p] = bk
    print(f"  parsed.")

    # Best strategy from event-aware sweep: behind_inside_2c spread>=10¢ no cooldown
    qd, ms, cd = -0.02, 0.10, 0
    print(f"\nStrategy: behind_inside_2c (delta={qd:+.2f}) spread>=10¢, no event cooldown")
    print(f"{'slippage_¢/share':>18} {'fills':>6} {'tokens':>7} {'total_$':>9} {'¢/fill':>8}")
    print("-" * 60)
    for slip_cents in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
        slippage = slip_cents / 100.0
        total_pnl = 0.0
        total_fills = 0
        tokens_active = 0
        for p, evs in all_events.items():
            for tok, pts in all_books[p].items():
                r = simulate_with_event_cooldown(pts, evs, cd,
                                                 quote_delta=qd, min_spread=ms,
                                                 slippage=slippage)
                if r is None or r["n_fills"] == 0:
                    continue
                total_pnl += r["pnl_usd"]
                total_fills += r["n_fills"]
                tokens_active += 1
        per_fill = total_pnl / max(1, total_fills) * 100
        sign = "+" if total_pnl >= 0 else ""
        print(f"{slip_cents:>17.2f}¢ {total_fills:>6} {tokens_active:>7} "
              f"{sign}{total_pnl:>+8.1f} {per_fill:>+7.2f}¢")


if __name__ == "__main__":
    main()
