"""
Market-making viability analysis.

Q1: How wide are spreads in our data, and how often do they fully compress?
Q2: How much does the mid drift on different time horizons?
Q3: If we quote at mid (earning ~half spread per fill), does spread/2 beat
    expected adverse selection?

This is a first-pass analytical study — no order placement.
"""
import json, os, glob, math
import numpy as np
from collections import defaultdict


def load_book_streams(processed_dir="data/processed"):
    streams = defaultdict(list)
    for path in glob.glob(os.path.join(processed_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("type") != "book":
                    continue
                tok = r.get("tokenId")
                bid = r.get("bid")
                ask = r.get("ask")
                if not tok or bid is None or ask is None:
                    continue
                streams[tok].append((r["ts"], bid, ask))
    for tok in streams:
        streams[tok].sort(key=lambda x: x[0])
    return streams


def main():
    print("Loading book streams from data/processed/…")
    streams = load_book_streams()
    n_tokens = len(streams)
    total_pts = sum(len(v) for v in streams.values())
    print(f"  {n_tokens} tokens, {total_pts} book points\n")

    # Q1: spread distribution per book point
    spreads = []
    mids = []
    for tok, pts in streams.items():
        for _, bid, ask in pts:
            if 0 < bid < 1 and 0 < ask < 1 and ask > bid:
                spreads.append(ask - bid)
                mids.append((bid + ask) / 2)
    spreads.sort()
    print(f"=== Q1: Spread distribution ===")
    print(f"  n = {len(spreads)}")
    print(f"  p10  = {spreads[int(len(spreads)*0.10)]*100:5.1f}¢")
    print(f"  p25  = {spreads[int(len(spreads)*0.25)]*100:5.1f}¢")
    print(f"  p50  = {spreads[int(len(spreads)*0.50)]*100:5.1f}¢")
    print(f"  p75  = {spreads[int(len(spreads)*0.75)]*100:5.1f}¢")
    print(f"  p90  = {spreads[int(len(spreads)*0.90)]*100:5.1f}¢")
    print(f"  mean = {np.mean(spreads)*100:5.1f}¢")

    # Q2: mid drift distribution at multiple horizons.
    # For each book update, compute |mid(t+H) - mid(t)| for H in {10s, 30s, 60s, 300s}.
    print(f"\n=== Q2: Mid drift over time horizons ===")
    import bisect

    horizons = [10, 30, 60, 300, 600]
    drifts = {h: [] for h in horizons}
    # Track signed drift too, to see if there's directional bias
    signed_drifts = {h: [] for h in horizons}

    for tok, pts in streams.items():
        if len(pts) < 2:
            continue
        keys = [t for t, _, _ in pts]
        for i, (t, bid, ask) in enumerate(pts):
            mid0 = (bid + ask) / 2
            for h in horizons:
                # Find earliest pt with ts >= t+h
                target = t + h
                j = bisect.bisect_left(keys, target)
                if j >= len(pts):
                    continue
                _, b1, a1 = pts[j]
                mid1 = (b1 + a1) / 2
                drift = mid1 - mid0
                drifts[h].append(abs(drift))
                signed_drifts[h].append(drift)

    print(f"{'horizon':>8} {'n':>8} {'mean_|drift|':>14} {'p50':>10} {'p90':>10}")
    for h in horizons:
        d = sorted(drifts[h])
        if not d:
            continue
        print(f"{h:>7}s {len(d):>8}  {np.mean(d)*100:>11.2f}¢ {d[len(d)//2]*100:>8.2f}¢ "
              f"{d[int(len(d)*0.9)]*100:>8.2f}¢")

    # Q3: Compare "spread captured" (~half-spread per fill) to expected drift.
    # Simple model: a round trip = both bid and ask fill within H seconds.
    # Estimated spread captured per round-trip = current spread (we earn the full spread on a paired fill).
    # Expected adverse selection per round-trip ≈ 2 × mean drift over H (we pay it both ways).
    print(f"\n=== Q3: Spread-vs-drift comparison ===")
    print(f"Hypothesis: at each book point, compare current spread to expected "
          f"|mid drift| at horizon H. If spread > 2 × drift → favorable MM economics.\n")
    print(f"{'horizon':>8} {'mean_spread':>12} {'mean_2×drift':>14} {'spread/(2drift)':>16} {'EV/round-trip':>14}")
    for h in [30, 60, 300]:
        # Per-point pair: spread at t vs 2 × |drift over h|
        ev_pairs = []
        # Resample: walk streams again, computing per-point
        for tok, pts in streams.items():
            if len(pts) < 2: continue
            keys = [t for t, _, _ in pts]
            for i, (t, bid, ask) in enumerate(pts):
                if ask <= bid: continue
                spread = ask - bid
                mid0 = (bid + ask) / 2
                j = bisect.bisect_left(keys, t + h)
                if j >= len(pts): continue
                _, b1, a1 = pts[j]
                mid1 = (b1 + a1) / 2
                drift = abs(mid1 - mid0)
                ev = spread - 2 * drift  # round-trip EV if both sides fill
                ev_pairs.append((spread, drift, ev))
        if not ev_pairs:
            continue
        mean_spread = np.mean([p[0] for p in ev_pairs])
        mean_2drift = np.mean([2 * p[1] for p in ev_pairs])
        ratio = mean_spread / max(1e-9, mean_2drift)
        mean_ev = np.mean([p[2] for p in ev_pairs])
        positive_ev_pct = sum(1 for p in ev_pairs if p[2] > 0) / len(ev_pairs)
        print(f"{h:>7}s {mean_spread*100:>10.2f}¢ {mean_2drift*100:>13.2f}¢ "
              f"{ratio:>15.2f} {mean_ev*100:>+12.2f}¢   ({positive_ev_pct*100:.0f}% of pts +EV)")

    # Q4: same but only on tight-spread points (≤ 3¢)
    print(f"\n=== Q4: Same as Q3 but restricted to tight-spread points (≤ 3¢) ===")
    print(f"{'horizon':>8} {'mean_spread':>12} {'mean_2×drift':>14} {'spread/(2drift)':>16} {'EV/round-trip':>14}")
    for h in [30, 60, 300]:
        ev_pairs = []
        for tok, pts in streams.items():
            if len(pts) < 2: continue
            keys = [t for t, _, _ in pts]
            for i, (t, bid, ask) in enumerate(pts):
                spread = ask - bid
                if spread <= 0 or spread > 0.03: continue  # tight only
                mid0 = (bid + ask) / 2
                j = bisect.bisect_left(keys, t + h)
                if j >= len(pts): continue
                _, b1, a1 = pts[j]
                mid1 = (b1 + a1) / 2
                drift = abs(mid1 - mid0)
                ev = spread - 2 * drift
                ev_pairs.append((spread, drift, ev))
        if not ev_pairs: continue
        mean_spread = np.mean([p[0] for p in ev_pairs])
        mean_2drift = np.mean([2 * p[1] for p in ev_pairs])
        ratio = mean_spread / max(1e-9, mean_2drift)
        mean_ev = np.mean([p[2] for p in ev_pairs])
        positive_ev_pct = sum(1 for p in ev_pairs if p[2] > 0) / len(ev_pairs)
        print(f"{h:>7}s {mean_spread*100:>10.2f}¢ {mean_2drift*100:>13.2f}¢ "
              f"{ratio:>15.2f} {mean_ev*100:>+12.2f}¢   ({positive_ev_pct*100:.0f}% of pts +EV)")


if __name__ == "__main__":
    main()
