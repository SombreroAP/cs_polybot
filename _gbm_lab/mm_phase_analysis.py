"""
Per-game-phase MM toxicity analysis.

For each book point, identify the round_phase from the nearest snapshot.
Bucket fills by phase and report per-fill PnL + adverse selection. Identifies
which phases are favorable / hostile for market-making.
"""
import os, sys, glob, json, bisect
import numpy as np
from collections import defaultdict, Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_event_aware import parse_match_file


def parse_match_with_phase(path):
    """Return (events, books_by_token, phase_timeline).
    phase_timeline: list of (ts, round_phase) sorted by ts."""
    events = []
    books = defaultdict(list)
    phases = []
    prev_round_total = None; prev_map_total = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception: continue
            typ = r.get("type")
            if typ == "snapshot":
                t1 = r.get("team_one") or {}
                t2 = r.get("team_two") or {}
                rt = (t1.get("score") or 0) + (t2.get("score") or 0)
                mt = (t1.get("match_score") or 0) + (t2.get("match_score") or 0)
                ts = r.get("ts")
                rp = r.get("round_phase") or "UNKNOWN"
                if rp != "UNKNOWN":
                    phases.append((ts, rp))
                if prev_round_total is not None and rt > prev_round_total:
                    events.append((ts, "round_end"))
                if prev_map_total is not None and mt > prev_map_total:
                    events.append((ts, "map_win"))
                prev_round_total = rt; prev_map_total = mt
            elif typ == "book":
                tok = r.get("tokenId")
                bid = r.get("bid"); ask = r.get("ask")
                if tok and bid is not None and ask is not None:
                    books[tok].append((r.get("ts"), float(bid), float(ask)))
    events.sort()
    phases.sort()
    for tok in books: books[tok].sort()
    return events, dict(books), phases


def phase_at(phases, t):
    if not phases: return "UNKNOWN"
    keys = [p[0] for p in phases]
    i = bisect.bisect_right(keys, t)
    if i == 0: return "UNKNOWN"
    return phases[i-1][1]


def simulate_with_phase_tracking(pts, phases, quote_delta=-0.02, min_spread=0.10,
                                 max_inv=5.0, slippage=0.003, quote_size=1.0):
    """Same MM logic but also returns per-phase breakdown."""
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
            mid30 = mid_at(t_next + 30)
            fills.append((t_next, "bid_fill", price, mid30, phase_at(phases, t_next)))
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size; inv -= quote_size
            mid30 = mid_at(t_next + 30)
            fills.append((t_next, "ask_fill", price, mid30, phase_at(phases, t_next)))

        if spread < min_spread:
            have_bid_q = have_ask_q = False
            continue

        new_bid_q = bid_next + quote_delta
        new_ask_q = ask_next - quote_delta
        if new_bid_q >= new_ask_q:
            have_bid_q = have_ask_q = False
        else:
            have_bid_q = inv + quote_size <= max_inv
            have_ask_q = inv - quote_size >= -max_inv
            bid_q_price = new_bid_q
            ask_q_price = new_ask_q

    _, bid_last, ask_last = pts[-1]
    if inv > 0: cash += inv * bid_last
    elif inv < 0: cash -= (-inv) * ask_last
    cash -= len(fills) * slippage * quote_size

    # Per-phase aggregates
    phase_stats = defaultdict(lambda: {"n": 0, "adverse_sum": 0.0})
    for (ts, side, price, mid30, ph) in fills:
        adverse = price - mid30 if side == "bid_fill" else mid30 - price
        phase_stats[ph]["n"] += 1
        phase_stats[ph]["adverse_sum"] += adverse

    return {"pnl_usd": cash, "n_fills": len(fills), "phase_stats": dict(phase_stats)}


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_data = []
    for p in paths:
        ev, bk, ph = parse_match_with_phase(p)
        if bk: all_data.append((ev, bk, ph))
    print(f"  {len(all_data)} files parsed.")

    # Run sim and accumulate per-phase stats
    total_phase = defaultdict(lambda: {"n": 0, "adverse_sum": 0.0})
    total_pnl = 0.0; total_fills = 0
    for (ev, bk, ph) in all_data:
        for tok, pts in bk.items():
            r = simulate_with_phase_tracking(pts, ph)
            if r is None: continue
            total_pnl += r["pnl_usd"]
            total_fills += r["n_fills"]
            for phase, stats in r["phase_stats"].items():
                total_phase[phase]["n"] += stats["n"]
                total_phase[phase]["adverse_sum"] += stats["adverse_sum"]

    print(f"\nTotal: {total_fills} fills, ${total_pnl:+.2f}")
    print(f"\n{'phase':>20} {'n_fills':>10} {'%':>6} {'mean_adverse_30s':>17}")
    print("-" * 60)
    rows = sorted(total_phase.items(), key=lambda kv: -kv[1]["n"])
    for ph, st in rows:
        n = st["n"]
        if n < 5: continue
        adverse_avg_cents = (st["adverse_sum"] / n) * 100
        pct = n / max(1, total_fills) * 100
        print(f"{ph:>20} {n:>10} {pct:>5.1f}% {adverse_avg_cents:>+15.2f}¢")


if __name__ == "__main__":
    main()
