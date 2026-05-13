"""
Final validation of the drift-conditioned MM strategy.

Runs the best config (behind_inside_2c, spread>=10¢, drift filter lookback=30s
max_drift=1¢) and reports:
  - Per-fill bootstrap CI on PnL
  - Daily PnL distribution
  - Robustness across slippage assumptions
  - Per-token PnL distribution
"""
import os, sys, glob, bisect, random
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mm_event_aware import parse_match_file


def simulate_token_with_drift(pts, lookback_s=30, max_drift_cents=1.0,
                              quote_delta=-0.02, min_spread=0.10,
                              max_inv=5.0, slippage=0.003, quote_size=1.0,
                              drift_mode="absolute", drift_ratio=0.30):
    """Returns full per-fill list so we can compute distributions."""
    if len(pts) < 2:
        return None

    cash = 0.0; inv = 0.0
    fills = []  # list of (ts, side, price, pnl_per_share_after_pairing)
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0
    open_bids = []; open_asks = []

    keys = [t for t, _, _ in pts]
    mids = [(t, (b+a)/2) for t, b, a in pts]

    def mid_at(t):
        i = bisect.bisect_right(keys, t) - 1
        if i < 0: return mids[0][1]
        return mids[i][1]

    def recent_drift_cents(t_now):
        past_mid = mid_at(t_now - lookback_s)
        cur_mid = mid_at(t_now)
        return abs(cur_mid - past_mid) * 100

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid

        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            price = bid_q_price
            cash -= price * quote_size
            inv += quote_size
            fills.append({"ts": t_next, "side": "bid_fill", "price": price})
            open_bids.append(price)
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            price = ask_q_price
            cash += price * quote_size
            inv -= quote_size
            fills.append({"ts": t_next, "side": "ask_fill", "price": price})
            open_asks.append(price)

        # Filters
        if drift_mode == "relative":
            threshold_cents = drift_ratio * spread * 100
        else:
            threshold_cents = max_drift_cents
        if spread < min_spread or recent_drift_cents(t_next) > threshold_cents:
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

    return {"n_fills": len(fills), "pnl_usd": cash, "fills": fills}


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    if not values: return None, None, None
    rng = rng or random.Random(42)
    n = len(values)
    mean = sum(values) / n
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(n_boot * (alpha/2))]
    hi = boots[int(n_boot * (1 - alpha/2))]
    return mean, lo, hi


def main():
    paths = glob.glob("data/processed/*.jsonl")
    print(f"Parsing {len(paths)} match files…")
    all_books = {}
    for p in paths:
        _, bk = parse_match_file(p)
        if bk: all_books[p] = bk
    print(f"  parsed.")

    print("\n" + "="*78)
    print("FINAL VALIDATION — drift-conditioned MM strategy (relative drift mode)")
    print("="*78)
    print(f"Config: behind_inside_2c, spread>=10¢, drift_lookback=30s, drift_ratio=0.30")
    # Override the default sim to use relative-drift mode
    global _drift_kwargs
    _drift_kwargs = {"drift_mode": "relative", "drift_ratio": 0.30}

    # Robustness across slippage
    print(f"\n--- Robustness across slippage (per-share friction) ---")
    print(f"{'slippage':>10} {'fills':>6} {'tokens_active':>15} {'total_$':>9} {'¢/fill':>8} "
          f"{'CI_lo':>7} {'CI_hi':>7}")
    print("-" * 75)
    all_pnls_at_005 = []
    for slip_cents in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
        slip = slip_cents / 100
        total_pnl = 0.0; total_fills = 0; tokens_active = 0
        per_fill_pnls = []
        for p, books in all_books.items():
            for tok, pts in books.items():
                r = simulate_token_with_drift(pts, slippage=slip, **_drift_kwargs)
                if r is None or r["n_fills"] == 0: continue
                total_pnl += r["pnl_usd"]; total_fills += r["n_fills"]; tokens_active += 1
                # per-fill cash pnl ~ approximation: total cash / n_fills as estimate
                # but better: use cash/fills as the per-fill sample
                avg = r["pnl_usd"] / r["n_fills"]
                per_fill_pnls.extend([avg] * r["n_fills"])
        per_fill_cents = total_pnl / max(1, total_fills) * 100
        if per_fill_pnls:
            mean_pf = sum(per_fill_pnls)/len(per_fill_pnls)
            _, lo, hi = bootstrap_ci(per_fill_pnls)
            sign = "+" if total_pnl >= 0 else ""
            print(f"{slip_cents:>9.2f}¢ {total_fills:>6} {tokens_active:>15} "
                  f"{sign}{total_pnl:>+8.1f} {per_fill_cents:>+7.2f}¢ "
                  f"{lo*100:>+6.2f}¢ {hi*100:>+6.2f}¢")
        if slip_cents == 0.05:
            all_pnls_at_005 = per_fill_pnls

    # Daily PnL distribution (at realistic 0.05¢ slippage)
    print(f"\n--- Per-day PnL distribution (slippage=0.05¢/share) ---")
    daily = defaultdict(float)
    daily_fills = defaultdict(int)
    for p, books in all_books.items():
        for tok, pts in books.items():
            r = simulate_token_with_drift(pts, slippage=0.0005, **_drift_kwargs)
            if r is None or r["n_fills"] == 0: continue
            for f in r["fills"]:
                day = datetime.fromtimestamp(f["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
                daily_fills[day] += 1
            # Distribute token's PnL evenly across fills (rough approximation)
            avg = r["pnl_usd"] / r["n_fills"]
            for f in r["fills"]:
                day = datetime.fromtimestamp(f["ts"], tz=timezone.utc).strftime("%Y-%m-%d")
                daily[day] += avg
    days = sorted(daily.keys())
    print(f"{'day':>12} {'fills':>6} {'pnl_$':>9}")
    for d in days[-15:]:
        sign = "+" if daily[d] >= 0 else ""
        print(f"{d:>12} {daily_fills[d]:>6} {sign}{daily[d]:>+8.3f}")
    profitable_days = sum(1 for v in daily.values() if v > 0)
    print(f"\nDays with positive PnL: {profitable_days}/{len(daily)} ({profitable_days/max(1,len(daily))*100:.0f}%)")
    print(f"Total over {len(daily)} days: ${sum(daily.values()):+.2f}")

    # Per-token distribution
    print(f"\n--- Per-token PnL distribution ---")
    token_pnls = []
    for p, books in all_books.items():
        for tok, pts in books.items():
            r = simulate_token_with_drift(pts, slippage=0.0005, **_drift_kwargs)
            if r is None or r["n_fills"] == 0: continue
            token_pnls.append(r["pnl_usd"])
    token_pnls.sort()
    if token_pnls:
        n = len(token_pnls)
        winners = sum(1 for p in token_pnls if p > 0)
        print(f"  tokens: {n}  winners: {winners} ({winners/n*100:.0f}%)")
        print(f"  mean: ${sum(token_pnls)/n:+.4f}  median: ${token_pnls[n//2]:+.4f}")
        print(f"  p10: ${token_pnls[n//10]:+.4f}  p90: ${token_pnls[int(n*0.9)]:+.4f}")
        print(f"  worst: ${min(token_pnls):+.3f}  best: ${max(token_pnls):+.3f}")


if __name__ == "__main__":
    main()
