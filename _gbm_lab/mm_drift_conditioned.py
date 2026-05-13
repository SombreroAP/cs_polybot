"""
Drift-conditioned MM: pull quotes when recent mid-drift exceeds a threshold.

Hypothesis: large recent mid drift signals informed flow about to take more
liquidity. Pulling quotes during these periods reduces toxic fills.
"""
import os, sys, glob, bisect
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_event_aware import parse_match_file


def simulate_drift_conditioned(pts, lookback_s, max_drift_cents,
                              quote_delta=-0.02, min_spread=0.10,
                              max_inv=5.0, slippage=0.003, quote_size=1.0):
    if len(pts) < 2: return None

    cash = 0.0; inv = 0.0
    fills = []
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0

    keys = [t for t, _, _ in pts]
    mids = [(t, (b+a)/2) for t, b, a in pts]

    def mid_at(t_target):
        i = bisect.bisect_right(keys, t_target) - 1
        if i < 0: return mids[0][1]
        return mids[i][1]

    def recent_drift_cents(t_now):
        # Mid drift over last lookback_s
        past_mid = mid_at(t_now - lookback_s)
        cur_mid = mid_at(t_now)
        return abs(cur_mid - past_mid) * 100

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid

        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            price = bid_q_price
            cash -= price * quote_size; inv += quote_size
            fills.append((t_next, "bid_fill", price))
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size; inv -= quote_size
            fills.append((t_next, "ask_fill", price))

        # Filters
        if spread < min_spread:
            have_bid_q = have_ask_q = False
            continue
        recent_drift = recent_drift_cents(t_next)
        if recent_drift > max_drift_cents:
            have_bid_q = have_ask_q = False
            continue

        new_bid_q = bid_next + quote_delta
        new_ask_q = ask_next - quote_delta
        if new_bid_q >= new_ask_q:
            have_bid_q = have_ask_q = False
        else:
            have_bid_q = inv + quote_size <= max_inv
            have_ask_q = inv - quote_size >= -max_inv
            bid_q_price = new_bid_q; ask_q_price = new_ask_q

    _, bid_last, ask_last = pts[-1]
    if inv > 0: cash += inv * bid_last
    elif inv < 0: cash -= (-inv) * ask_last
    cash -= len(fills) * slippage * quote_size

    return {"n_fills": len(fills), "pnl_usd": cash}


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_books = {}
    for p in paths:
        _, bk = parse_match_file(p)
        if bk: all_books[p] = bk
    print(f"  parsed.")

    print(f"\nDrift-conditioned MM sweep (strategy: behind_inside_2c spread>=10¢)")
    print(f"slippage=0 (gross), to see true edge")
    print(f"{'lookback':>10} {'max_drift':>10} {'fills':>6} {'total_$':>9} {'¢/fill':>8}")
    print("-" * 60)
    for lookback in [10, 30, 60]:
        for max_d in [1.0, 2.0, 3.0, 5.0, 10.0, 999.0]:  # 999 = no filter
            total_pnl = 0.0; total_fills = 0
            for p, books in all_books.items():
                for tok, pts in books.items():
                    r = simulate_drift_conditioned(pts, lookback, max_d, slippage=0.0)
                    if r is None or r["n_fills"] == 0: continue
                    total_pnl += r["pnl_usd"]
                    total_fills += r["n_fills"]
            per = total_pnl / max(1, total_fills) * 100
            sign = "+" if total_pnl >= 0 else ""
            print(f"{lookback:>8}s {max_d:>9.1f}¢ {total_fills:>6} {sign}{total_pnl:>+8.1f} {per:>+7.2f}¢")
        print()


if __name__ == "__main__":
    main()
