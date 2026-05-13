"""
Volatility-aware drift filter — use drift relative to current spread.

Hypothesis: in wide-spread markets, 1¢ absolute drift is noise. In narrow
markets, 1¢ is signal. Using max_drift = K * spread instead of fixed 1¢ may
generalise better.
"""
import os, sys, glob, bisect
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_event_aware import parse_match_file


def simulate_relative_drift(pts, lookback_s, drift_ratio,
                            quote_delta=-0.02, min_spread=0.10,
                            max_inv=5.0, slippage=0.003, quote_size=1.0):
    """drift_ratio = drift_threshold / current_spread."""
    if len(pts) < 2: return None
    cash = 0.0; inv = 0.0; n_fills = 0
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0

    keys = [t for t, _, _ in pts]
    mids = [(t, (b+a)/2) for t, b, a in pts]

    def mid_at(t):
        i = bisect.bisect_right(keys, t) - 1
        if i < 0: return mids[0][1]
        return mids[i][1]

    def drift(t_now):
        return abs(mid_at(t_now) - mid_at(t_now - lookback_s))

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid

        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            cash -= bid_q_price * quote_size; inv += quote_size; n_fills += 1
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            cash += ask_q_price * quote_size; inv -= quote_size; n_fills += 1

        if spread < min_spread:
            have_bid_q = have_ask_q = False
            continue
        # Relative drift filter
        if drift(t_next) > drift_ratio * spread:
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
    cash -= n_fills * slippage * quote_size
    return {"n_fills": n_fills, "pnl_usd": cash}


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_books = {}
    for p in paths:
        _, bk = parse_match_file(p)
        if bk: all_books[p] = bk
    print(f"  parsed.")

    # Baseline (absolute drift) for comparison
    print(f"\nBaseline: absolute drift filter (gross, slippage=0)")
    from mm_drift_conditioned import simulate_drift_conditioned
    for lookback, max_d in [(30, 1.0), (60, 1.0), (30, 2.0)]:
        total_pnl = 0; total_fills = 0
        for p, books in all_books.items():
            for tok, pts in books.items():
                r = simulate_drift_conditioned(pts, lookback, max_d, slippage=0.0)
                if r is None or r["n_fills"] == 0: continue
                total_pnl += r["pnl_usd"]; total_fills += r["n_fills"]
        per = total_pnl / max(1, total_fills) * 100
        print(f"  absolute lookback={lookback}s max_drift={max_d:.1f}¢: "
              f"fills={total_fills} total=${total_pnl:+.1f} ¢/fill={per:+.2f}")

    print(f"\nRelative drift filter (drift / current_spread)")
    print(f"{'lookback':>8} {'ratio':>6} {'fills':>6} {'total_$':>9} {'¢/fill':>8}")
    print("-" * 50)
    for lookback in [30, 60]:
        for ratio in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            total_pnl = 0; total_fills = 0
            for p, books in all_books.items():
                for tok, pts in books.items():
                    r = simulate_relative_drift(pts, lookback, ratio, slippage=0.0)
                    if r is None or r["n_fills"] == 0: continue
                    total_pnl += r["pnl_usd"]; total_fills += r["n_fills"]
            per = total_pnl / max(1, total_fills) * 100
            sign = "+" if total_pnl >= 0 else ""
            print(f"{lookback:>7}s {ratio:>5.2f} {total_fills:>6} "
                  f"{sign}{total_pnl:>+8.1f} {per:>+7.2f}¢")
        print()


if __name__ == "__main__":
    main()
