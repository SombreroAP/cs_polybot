"""
Inventory-skewing MM simulator.

When inventory is long, quote a more aggressive ASK (closer to mid) and a less
aggressive BID. When short, the reverse. This reduces inventory carrying cost
by encouraging offsetting fills.

Parameter: skew_per_share = how much (in $) we shift the quote per unit of
inventory. Positive value means quotes shift in the direction that offsets.
"""
import os, sys, glob, json
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_event_aware import parse_match_file


def simulate_with_skew(pts,
                      quote_delta=-0.02,
                      min_spread=0.10,
                      max_inv=5.0,
                      slippage=0.003,
                      skew_per_share=0.005,
                      quote_size=1.0):
    if len(pts) < 2:
        return None

    cash = 0.0
    inv = 0.0
    fills = []
    max_abs_inv = 0.0

    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0

    mids = [(t, (b + a) / 2) for t, b, a in pts]
    def mid_at(t_target):
        lo, hi = 0, len(mids) - 1
        while lo < hi:
            m = (lo + hi + 1) // 2
            if mids[m][0] <= t_target: lo = m
            else: hi = m - 1
        return mids[lo][1]

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i + 1]
        spread = ask - bid

        # Fill detection
        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            price = bid_q_price
            cash -= price * quote_size
            inv += quote_size
            fills.append((t_next, "bid_fill", price, mid_at(t_next + 30)))
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size
            inv -= quote_size
            fills.append((t_next, "ask_fill", price, mid_at(t_next + 30)))

        max_abs_inv = max(max_abs_inv, abs(inv))

        if spread < min_spread:
            have_bid_q = have_ask_q = False
            continue

        # Inventory-skewed quotes
        # Long inventory → shift quotes DOWN (lower ask = more attractive to offset; lower bid = less attractive)
        # Short inventory → shift quotes UP
        skew = -inv * skew_per_share  # shifts quotes down when long
        new_bid_q = bid_next + quote_delta + skew
        new_ask_q = ask_next - quote_delta + skew
        if new_bid_q >= new_ask_q:
            have_bid_q = have_ask_q = False
        else:
            have_bid_q = inv + quote_size <= max_inv
            have_ask_q = inv - quote_size >= -max_inv
            bid_q_price = new_bid_q
            ask_q_price = new_ask_q

    t_last, bid_last, ask_last = pts[-1]
    if inv > 0: cash += inv * bid_last
    elif inv < 0: cash -= (-inv) * ask_last
    cash -= len(fills) * slippage * quote_size

    return {"n_fills": len(fills), "pnl_usd": cash, "max_abs_inv": max_abs_inv}


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_books = {}
    for p in paths:
        _, bk = parse_match_file(p)
        if bk: all_books[p] = bk
    print("  parsed.")

    print(f"\nInventory-skew sweep on behind_inside_2c spread>=10¢")
    print(f"{'skew_per_share':>16} {'fills':>6} {'tokens':>7} {'total_$':>9} {'¢/fill':>8} {'max_inv':>8}")
    print("-" * 70)
    for skew_cents in [0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0]:
        skew = skew_cents / 100.0
        total_pnl = 0.0
        total_fills = 0
        tokens_active = 0
        max_inv_overall = 0
        for p, books in all_books.items():
            for tok, pts in books.items():
                r = simulate_with_skew(pts, skew_per_share=skew)
                if r is None or r["n_fills"] == 0: continue
                total_pnl += r["pnl_usd"]
                total_fills += r["n_fills"]
                tokens_active += 1
                max_inv_overall = max(max_inv_overall, r["max_abs_inv"])
        per_fill = total_pnl / max(1, total_fills) * 100
        sign = "+" if total_pnl >= 0 else ""
        print(f"{skew_cents:>15.2f}¢ {total_fills:>6} {tokens_active:>7} "
              f"{sign}{total_pnl:>+8.1f} {per_fill:>+7.2f}¢ {max_inv_overall:>7.1f}")


if __name__ == "__main__":
    main()
