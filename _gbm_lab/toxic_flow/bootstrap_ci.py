"""
Bootstrap 95% confidence intervals on the CatBoost v2 τ=1.5¢ result.

The A/B simulator reports total PnL. To get a per-fill CI we need the
individual fill PnLs. This script re-runs the sim, captures every fill's
hindsight PnL, then bootstraps the mean.
"""
import os, sys, glob, bisect, json, pickle, random, math
from datetime import datetime, timezone
import numpy as np
from collections import defaultdict


QUOTE_DELTA = -0.02
MIN_SPREAD = 0.10
MAX_INV = 5.0
SLIPPAGE = 0.0005
QUOTE_SIZE = 1.0


def parse_books(path):
    streams = defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") != "book":
                continue
            tok = r.get("tokenId")
            bid = r.get("bid"); ask = r.get("ask"); ts = r.get("ts")
            if tok and bid is not None and ask is not None and ts is not None:
                streams[tok].append((float(ts), float(bid), float(ask)))
    for tok in streams:
        streams[tok].sort()
    return dict(streams)


def features_v2(t, mid, spread, ts_arr, mid_arr, bid_arr, ask_arr, comp_ts, comp_mid_arr):
    def latest_le_idx(arr, target):
        i = bisect.bisect_right(arr, target) - 1
        return i if i >= 0 else None
    def mid_at(t_target):
        i = latest_le_idx(ts_arr, t_target)
        return float(mid_arr[i]) if i is not None else None
    def cmid_at(t_target):
        if comp_ts is None:
            return None
        i = latest_le_idx(comp_ts, t_target)
        return float(comp_mid_arr[i]) if i is not None else None
    def vol_window(t_now, window):
        lo = bisect.bisect_left(ts_arr, t_now - window)
        hi = bisect.bisect_right(ts_arr, t_now)
        if hi - lo < 2:
            return 0.0
        return float(np.std(mid_arr[lo:hi]))

    f = {"spread": spread}
    for w in [5, 10, 30, 60, 300]:
        mp = mid_at(t - w); v = (mid - mp) if mp is not None else 0.0
        f[f"past_drift_{w}s"] = v
        f[f"abs_past_drift_{w}s"] = abs(v)
    mp = mid_at(t - 1.0)
    f["velocity_1s"] = (mid - mp) if mp is not None else 0.0
    a5 = mid_at(t - 5); a10 = mid_at(t - 10)
    f["accel_5_10"] = ((mid - a5) - (a5 - a10)) if (a5 is not None and a10 is not None) else 0.0
    for w in [30, 60, 300]:
        f[f"vol_{w}s"] = vol_window(t, w)
    i30 = latest_le_idx(ts_arr, t - 30)
    f["spread_change_30s"] = spread - (float(ask_arr[i30] - bid_arr[i30]) if i30 is not None else spread)
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    h = dt.hour + dt.minute / 60.0
    f["tod_sin"] = math.sin(2 * math.pi * h / 24)
    f["tod_cos"] = math.cos(2 * math.pi * h / 24)
    dow = dt.weekday()
    f["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    f["dow_cos"] = math.cos(2 * math.pi * dow / 7)
    cn = cmid_at(t); c30 = cmid_at(t - 30)
    if cn is not None and c30 is not None:
        f["comp_drift_30s"] = cn - c30
        f["abs_comp_drift_30s"] = abs(cn - c30)
        f["comp_mid"] = cn; f["comp_sum"] = mid + cn
        f["comp_sum_dev"] = abs((mid + cn) - 1.0)
    else:
        f["comp_drift_30s"] = 0.0; f["abs_comp_drift_30s"] = 0.0
        f["comp_mid"] = 0.5; f["comp_sum"] = 1.0; f["comp_sum_dev"] = 0.0
    return f


def simulate_collect_fills(pts, gate, slip=SLIPPAGE):
    """Like simulate_token but returns the list of per-fill PnLs for bootstrapping."""
    if len(pts) < 2:
        return []
    cash = 0.0; inv = 0.0
    fills = []  # per-fill cash deltas
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0
    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid
        # Fill detection
        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            cash -= bid_q_price * QUOTE_SIZE; inv += QUOTE_SIZE
            fills.append(-bid_q_price * QUOTE_SIZE - slip * QUOTE_SIZE)
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            cash += ask_q_price * QUOTE_SIZE; inv -= QUOTE_SIZE
            fills.append(ask_q_price * QUOTE_SIZE - slip * QUOTE_SIZE)
        if spread < MIN_SPREAD or i < 1:
            have_bid_q = have_ask_q = False
            continue
        try:
            if gate(i):
                have_bid_q = have_ask_q = False
                continue
        except Exception:
            have_bid_q = have_ask_q = False
            continue
        new_bid_q = bid_next + QUOTE_DELTA
        new_ask_q = ask_next - QUOTE_DELTA
        if new_bid_q >= new_ask_q:
            have_bid_q = have_ask_q = False
        else:
            have_bid_q = inv + QUOTE_SIZE <= MAX_INV
            have_ask_q = inv - QUOTE_SIZE >= -MAX_INV
            bid_q_price = new_bid_q
            ask_q_price = new_ask_q
    # Liquidate
    _, bid_last, ask_last = pts[-1]
    if inv > 0:
        fills.append(inv * bid_last)
    elif inv < 0:
        fills.append(-(-inv) * ask_last)
    return fills


def bootstrap_ci(values, n_boot=2000, alpha=0.05, seed=42):
    if not values:
        return None, None, None
    rng = random.Random(seed)
    n = len(values)
    means = []
    for _ in range(n_boot):
        s = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(s) / n)
    means.sort()
    mean = sum(values) / n
    lo = means[int(n_boot * (alpha / 2))]
    hi = means[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def main():
    paths = sorted(glob.glob("data/processed/*.jsonl"))
    print(f"Loading {len(paths)} files…")
    all_streams = {}
    file_tokens = {}
    for p in paths:
        s = parse_books(p)
        for tok, pts in s.items():
            all_streams[tok] = pts
        if len(s) == 2:
            t = list(s.keys())
            if all(len(s[x]) >= 2 for x in t):
                file_tokens[p] = t
    complement = {}
    for p, ts in file_tokens.items():
        a, b = ts
        complement[a] = b; complement[b] = a
    print(f"  {len(all_streams)} tokens, {len(complement)} complement pairs\n")

    # Load CatBoost v2 model
    with open("models/toxic_flow_v2_cb.pkl", "rb") as f:
        d = pickle.load(f)
    model = d["model"]
    features = d["features"]
    threshold_cents = 1.5
    print(f"Model: catboost_v2  features={len(features)}  threshold={threshold_cents}¢\n")

    all_fills = []
    n_tokens_traded = 0
    for tok, pts in all_streams.items():
        ts_arr = np.array([p[0] for p in pts])
        bid_arr = np.array([p[1] for p in pts])
        ask_arr = np.array([p[2] for p in pts])
        mid_arr = (bid_arr + ask_arr) / 2
        comp_tok = complement.get(tok)
        if comp_tok and comp_tok in all_streams:
            cpts = all_streams[comp_tok]
            cts = np.array([p[0] for p in cpts])
            cmid = (np.array([p[1] for p in cpts]) + np.array([p[2] for p in cpts])) / 2
        else:
            cts = None; cmid = None

        def gate(i):
            t = float(ts_arr[i])
            mid = float(mid_arr[i])
            spread = float(ask_arr[i] - bid_arr[i])
            feats = features_v2(t, mid, spread, ts_arr, mid_arr, bid_arr, ask_arr, cts, cmid)
            x = np.array([[feats[f] for f in features]])
            pred = float(model.predict(x)[0])
            return pred * 100 > threshold_cents

        fills = simulate_collect_fills(pts, gate)
        # Pair up into per-roundtrip PnL
        # For bootstrap, treat individual fill cash flows; sum across all = total PnL
        if fills:
            n_tokens_traded += 1
        all_fills.extend(fills)

    n = len(all_fills)
    total = sum(all_fills)
    print(f"Total cash flows recorded: {n}  total PnL: ${total:+.2f}")
    print(f"Active tokens: {n_tokens_traded}")
    if n == 0:
        return

    # Per-token PnL bootstrap (more honest unit than per-flow)
    by_token = defaultdict(float)
    # We don't have token labels on the flows; redo with token bookkeeping
    print("\nRebuilding with per-token totals…")
    token_pnls = []
    for tok, pts in all_streams.items():
        ts_arr = np.array([p[0] for p in pts]); bid_arr = np.array([p[1] for p in pts])
        ask_arr = np.array([p[2] for p in pts]); mid_arr = (bid_arr + ask_arr) / 2
        comp_tok = complement.get(tok)
        if comp_tok and comp_tok in all_streams:
            cpts = all_streams[comp_tok]
            cts = np.array([p[0] for p in cpts])
            cmid = (np.array([p[1] for p in cpts]) + np.array([p[2] for p in cpts])) / 2
        else:
            cts = None; cmid = None
        def gate2(i):
            t = float(ts_arr[i]); mid = float(mid_arr[i]); spread = float(ask_arr[i] - bid_arr[i])
            feats = features_v2(t, mid, spread, ts_arr, mid_arr, bid_arr, ask_arr, cts, cmid)
            x = np.array([[feats[f] for f in features]])
            return float(model.predict(x)[0]) * 100 > threshold_cents
        flows = simulate_collect_fills(pts, gate2)
        if flows:
            token_pnls.append(sum(flows))

    print(f"Per-token PnLs: n={len(token_pnls)}")
    if token_pnls:
        token_mean, token_lo, token_hi = bootstrap_ci(token_pnls)
        print(f"  mean per-token PnL: ${token_mean:+.4f}")
        print(f"  95% CI: [${token_lo:+.4f}, ${token_hi:+.4f}]")
        winners = sum(1 for p in token_pnls if p > 0)
        print(f"  winners: {winners}/{len(token_pnls)} ({winners*100/len(token_pnls):.1f}%)")

    # Total PnL projection summary
    print(f"\nSummary:")
    print(f"  Per-token mean: ${token_mean:+.4f}")
    print(f"  95% CI: [${token_lo:+.4f}, ${token_hi:+.4f}]")
    print(f"  Total tokens (26-day backtest): {len(token_pnls)}")
    print(f"  Implied 26-day total PnL: ${sum(token_pnls):+.2f}")
    print(f"  Daily rate (1-share size): ${sum(token_pnls)/26:+.4f}")
    print(f"  At $100 quote size: ${100 * sum(token_pnls)/26:+.2f} / day")
    print(f"  At $1000 quote size: ${1000 * sum(token_pnls)/26:+.2f} / day")


if __name__ == "__main__":
    main()
