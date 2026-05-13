"""
Phase-filtered MM: only quote in non-live phases.
"""
import os, sys, glob, bisect
import numpy as np
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_phase_analysis import parse_match_with_phase, phase_at


def simulate_phase_filtered(pts, phases, allowed_phases,
                            quote_delta=-0.02, min_spread=0.10,
                            max_inv=5.0, slippage=0.003, quote_size=1.0):
    if len(pts) < 2: return None

    cash = 0.0; inv = 0.0
    fills = []
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0

    mids = [(t, (b+a)/2) for t, b, a in pts]
    def mid_at(t_target):
        lo, hi = 0, len(mids) - 1
        while lo < hi:
            m = (lo + hi + 1) // 2
            if mids[m][0] <= t_target: lo = m
            else: hi = m - 1
        return mids[lo][1]

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid

        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            price = bid_q_price
            cash -= price * quote_size; inv += quote_size
            fills.append((t_next, "bid_fill", price, mid_at(t_next + 30)))
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size; inv -= quote_size
            fills.append((t_next, "ask_fill", price, mid_at(t_next + 30)))

        # Phase + spread filters
        ph_now = phase_at(phases, t_next)
        if spread < min_spread or ph_now not in allowed_phases:
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
    all_data = []
    for p in paths:
        ev, bk, ph = parse_match_with_phase(p)
        if bk: all_data.append((ev, bk, ph))
    print(f"  parsed.")

    # Different phase filter combos
    phase_sets = [
        ("ALL phases (baseline)",     {"UNKNOWN","IN_PROGRESS","BUY_TIME","FINISHED","PAUSED"}),
        ("UNKNOWN only",              {"UNKNOWN"}),
        ("UNKNOWN + PAUSED",          {"UNKNOWN","PAUSED"}),
        ("not IN_PROGRESS",           {"UNKNOWN","BUY_TIME","FINISHED","PAUSED"}),
        ("not (IN_PROGRESS|BUY_TIME)", {"UNKNOWN","FINISHED","PAUSED"}),
    ]
    print(f"\nPhase filter sweep (strategy: behind_inside_2c spread>=10¢)")
    print(f"{'phase filter':<30} {'fills':>6} {'total_$':>9} {'¢/fill':>8}")
    print("-" * 60)
    for label, allowed in phase_sets:
        total_pnl = 0.0; total_fills = 0
        for (ev, bk, ph) in all_data:
            for tok, pts in bk.items():
                r = simulate_phase_filtered(pts, ph, allowed)
                if r is None or r["n_fills"] == 0: continue
                total_pnl += r["pnl_usd"]; total_fills += r["n_fills"]
        per = total_pnl / max(1, total_fills) * 100
        sign = "+" if total_pnl >= 0 else ""
        print(f"{label:<30} {total_fills:>6} {sign}{total_pnl:>+8.1f} {per:>+7.2f}¢")

    # Same sweep but with zero-friction (gross)
    print(f"\nGross (slippage=0) — shows true edge:")
    print(f"{'phase filter':<30} {'fills':>6} {'total_$':>9} {'¢/fill':>8}")
    print("-" * 60)
    for label, allowed in phase_sets:
        total_pnl = 0.0; total_fills = 0
        for (ev, bk, ph) in all_data:
            for tok, pts in bk.items():
                r = simulate_phase_filtered(pts, ph, allowed, slippage=0.0)
                if r is None or r["n_fills"] == 0: continue
                total_pnl += r["pnl_usd"]; total_fills += r["n_fills"]
        per = total_pnl / max(1, total_fills) * 100
        sign = "+" if total_pnl >= 0 else ""
        print(f"{label:<30} {total_fills:>6} {sign}{total_pnl:>+8.1f} {per:>+7.2f}¢")


if __name__ == "__main__":
    main()
