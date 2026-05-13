"""
Event-aware MM simulator.

For each processed match file:
  1. Walk snapshots to build an event timeline: each (round_total or map_total)
     increment is a 'round_end' or 'map_win' event at that ts.
  2. Walk the book stream for tokens in this match, but ONLY quote when the
     time-since-last-event exceeds a configurable cooldown.

Hypothesis: most adverse selection happens within the first 10-30 seconds
after a decisive event (round_end / map_win) — that's when informed flow
trades. Avoiding those windows should improve realised PnL.
"""
import json, os, glob
import numpy as np
from collections import defaultdict
from mm_simulator import simulate_token


def parse_match_file(path):
    """Return (events: [(ts, type)], books_by_token: {tok: [(ts, bid, ask)]})."""
    events = []
    books = defaultdict(list)
    prev_round_total = None
    prev_map_total = None
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            typ = r.get("type")
            if typ == "snapshot":
                t1 = r.get("team_one") or {}
                t2 = r.get("team_two") or {}
                rt = (t1.get("score") or 0) + (t2.get("score") or 0)
                mt = (t1.get("match_score") or 0) + (t2.get("match_score") or 0)
                ts = r.get("ts")
                if prev_round_total is not None and rt > prev_round_total:
                    events.append((ts, "round_end"))
                if prev_map_total is not None and mt > prev_map_total:
                    events.append((ts, "map_win"))
                prev_round_total = rt
                prev_map_total = mt
            elif typ == "book":
                tok = r.get("tokenId")
                bid = r.get("bid")
                ask = r.get("ask")
                if tok and bid is not None and ask is not None:
                    books[tok].append((r.get("ts"), float(bid), float(ask)))
    events.sort()
    for tok in books:
        books[tok].sort()
    return events, dict(books)


def simulate_with_event_cooldown(pts, events, cooldown_s,
                                 quote_delta=0.01, quote_size=1.0,
                                 max_inv=5.0, slippage=0.003, min_spread=0.0):
    """Like simulate_token but with an event-cooldown filter applied."""
    if len(pts) < 2:
        return None

    import bisect
    event_ts = [e[0] for e in events]

    def secs_since_event(t):
        i = bisect.bisect_right(event_ts, t)
        if i == 0:
            return float("inf")  # no prior events
        return t - event_ts[i - 1]

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

        # Fill detection (same as before)
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

        # Filters: spread + event cooldown
        if spread < min_spread:
            have_bid_q = have_ask_q = False
            continue
        if secs_since_event(t_next) < cooldown_s:
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

    t_last, bid_last, ask_last = pts[-1]
    if inv > 0:
        cash += inv * bid_last; inv = 0
    elif inv < 0:
        cash -= (-inv) * ask_last; inv = 0
    cash -= len(fills) * slippage * quote_size

    adverse = [(price - m30) if side == "bid_fill" else (m30 - price)
               for (_, side, price, m30) in fills]
    return {
        "n_fills": len(fills),
        "pnl_usd": cash,
        "max_abs_inv": max_abs_inv,
        "mean_adverse_30s_cents": np.mean(adverse) * 100 if adverse else 0,
    }


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Processing {len(paths)} match files…")

    # Parse all files once
    all_events = {}
    all_books = {}
    for p in paths:
        ev, bk = parse_match_file(p)
        if ev or bk:
            all_events[p] = ev
            all_books[p] = bk

    print(f"  {len(all_events)} files parsed")
    total_events = sum(len(v) for v in all_events.values())
    total_books_pts = sum(sum(len(b) for b in d.values()) for d in all_books.values())
    print(f"  {total_events} score-change events identified")
    print(f"  {total_books_pts} book points across all tokens\n")

    configs = [
        ("baseline (no event filter)         delta=+1¢ ms=0¢", +0.01, 0.0, 0),
        ("event_cooldown 5s                  delta=+1¢ ms=0¢", +0.01, 0.0, 5),
        ("event_cooldown 15s                 delta=+1¢ ms=0¢", +0.01, 0.0, 15),
        ("event_cooldown 30s                 delta=+1¢ ms=0¢", +0.01, 0.0, 30),
        ("event_cooldown 60s                 delta=+1¢ ms=0¢", +0.01, 0.0, 60),
        ("behind_inside_2c spread>=10¢ base                  ", -0.02, 0.10, 0),
        ("behind_inside_2c spread>=10¢ cooldown 15s          ", -0.02, 0.10, 15),
        ("behind_inside_2c spread>=10¢ cooldown 30s          ", -0.02, 0.10, 30),
        ("behind_inside_2c spread>=10¢ cooldown 60s          ", -0.02, 0.10, 60),
        ("inside_touch spread>=20¢ base                       ", 0.00, 0.20, 0),
        ("inside_touch spread>=20¢ cooldown 30s               ", 0.00, 0.20, 30),
        ("inside_touch spread>=20¢ cooldown 60s               ", 0.00, 0.20, 60),
    ]

    print("=" * 110)
    print(f"{'strategy':<60} {'fills':>6} {'tokens':>7} {'total_$':>9} {'¢/fill':>8} {'adv30s¢':>9}")
    print("=" * 110)

    for label, qd, ms, cd in configs:
        total_pnl = 0.0
        total_fills = 0
        tokens_active = 0
        all_adverse = []
        for p, evs in all_events.items():
            for tok, pts in all_books[p].items():
                if cd > 0:
                    r = simulate_with_event_cooldown(pts, evs, cd,
                                                     quote_delta=qd, min_spread=ms)
                else:
                    r = simulate_with_event_cooldown(pts, [], 0,
                                                     quote_delta=qd, min_spread=ms)
                if r is None or r["n_fills"] == 0:
                    continue
                total_pnl += r["pnl_usd"]
                total_fills += r["n_fills"]
                tokens_active += 1
                all_adverse.append(r["mean_adverse_30s_cents"])
        avg_per_fill = total_pnl / max(1, total_fills) * 100
        mean_adverse = np.mean(all_adverse) if all_adverse else 0
        print(f"{label:<60} {total_fills:>6} {tokens_active:>7} "
              f"{total_pnl:>+8.1f} {avg_per_fill:>+7.2f}¢ {mean_adverse:>+8.2f}¢")


if __name__ == "__main__":
    main()
