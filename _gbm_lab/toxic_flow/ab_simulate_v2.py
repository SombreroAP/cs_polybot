"""
A/B simulation comparing v1 and v2 models, all thresholds, against heuristic.
"""
import os, sys, glob, bisect, json, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ab_simulate import (parse_books, simulate_token, run_strategy,
                          heuristic_gate, QUOTE_DELTA, MIN_SPREAD, MAX_INV,
                          SLIPPAGE, QUOTE_SIZE, HEURISTIC_RATIO,
                          DRIFT_LOOKBACK_S, FUTURE_HORIZON_S)


FEATURES_V2 = [
    "spread", "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "vol_30s", "vol_60s", "vol_300s",
    "spread_change_30s",
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    "comp_drift_30s", "abs_comp_drift_30s", "comp_mid",
    "comp_sum", "comp_sum_dev",
]


def features_for_obs_v2(pts, i, ts_arr, mid_arr, bid_arr, ask_arr,
                        comp_ts=None, comp_mid_arr=None):
    """v2 features including volatility, day-of-week, complement."""
    import math
    t = float(ts_arr[i])
    bid = float(bid_arr[i]); ask = float(ask_arr[i])
    mid = float(mid_arr[i]); spread = ask - bid

    def mid_at(t_target):
        j = bisect.bisect_right(ts_arr, t_target) - 1
        return float(mid_arr[j]) if j >= 0 else None

    def cmid_at(t_target):
        if comp_ts is None:
            return None
        j = bisect.bisect_right(comp_ts, t_target) - 1
        return float(comp_mid_arr[j]) if j >= 0 else None

    def vol_window(t_now, window):
        lo = bisect.bisect_left(ts_arr, t_now - window)
        hi = bisect.bisect_right(ts_arr, t_now)
        if hi - lo < 2:
            return 0.0
        return float(np.std(mid_arr[lo:hi]))

    feats = {"spread": spread}
    for w in [5, 10, 30, 60, 300]:
        mp = mid_at(t - w); v = (mid - mp) if mp is not None else 0.0
        feats[f"past_drift_{w}s"] = v
        feats[f"abs_past_drift_{w}s"] = abs(v)
    mp = mid_at(t - 1.0)
    feats["velocity_1s"] = (mid - mp) if mp is not None else 0.0
    a5 = mid_at(t - 5); a10 = mid_at(t - 10)
    feats["accel_5_10"] = ((mid - a5) - (a5 - a10)) if (a5 is not None and a10 is not None) else 0.0
    for w in [30, 60, 300]:
        feats[f"vol_{w}s"] = vol_window(t, w)
    i_30 = bisect.bisect_right(ts_arr, t - 30) - 1
    feats["spread_change_30s"] = spread - (float(ask_arr[i_30] - bid_arr[i_30]) if i_30 >= 0 else spread)
    dt = datetime.fromtimestamp(t, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    feats["tod_sin"] = math.sin(2 * math.pi * hour / 24)
    feats["tod_cos"] = math.cos(2 * math.pi * hour / 24)
    dow = dt.weekday()
    feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)
    cn = cmid_at(t); c30 = cmid_at(t - 30)
    if cn is not None and c30 is not None:
        feats["comp_drift_30s"] = cn - c30
        feats["abs_comp_drift_30s"] = abs(cn - c30)
        feats["comp_mid"] = cn
        feats["comp_sum"] = mid + cn
        feats["comp_sum_dev"] = abs((mid + cn) - 1.0)
    else:
        feats["comp_drift_30s"] = 0.0
        feats["abs_comp_drift_30s"] = 0.0
        feats["comp_mid"] = 0.5
        feats["comp_sum"] = 1.0
        feats["comp_sum_dev"] = 0.0
    return feats


def make_model_gate_v2(model_path, threshold_cents, complement_map, all_streams):
    with open(model_path, "rb") as f:
        d = pickle.load(f)
    model = d["model"]
    features = d.get("features", FEATURES_V2)

    def gate_factory(token_id, pts, ts_arr, mid_arr, bid_arr, ask_arr):
        comp_tok = complement_map.get(token_id)
        if comp_tok and comp_tok in all_streams:
            cpts = all_streams[comp_tok]
            cts = np.array([p[0] for p in cpts])
            cmids = np.array([(p[1] + p[2]) / 2 for p in cpts])
        else:
            cts = None; cmids = None

        def gate(feats_unused_idx, spread):
            return False  # filled in below
        # Closure for full simulate path: we need a different signature
        return ts_arr, mid_arr, bid_arr, ask_arr, cts, cmids, features

    # Wrap to a simulator-compatible interface
    def make_gate_for_token(token_id, pts):
        ts_arr = np.array([p[0] for p in pts])
        bid_arr = np.array([p[1] for p in pts])
        ask_arr = np.array([p[2] for p in pts])
        mid_arr = (bid_arr + ask_arr) / 2
        comp_tok = complement_map.get(token_id)
        if comp_tok and comp_tok in all_streams:
            cpts = all_streams[comp_tok]
            cts = np.array([p[0] for p in cpts])
            cmids = np.array([(p[1] + p[2]) / 2 for p in cpts])
        else:
            cts = None; cmids = None

        def gate(i):
            feats = features_for_obs_v2(pts, i, ts_arr, mid_arr, bid_arr, ask_arr, cts, cmids)
            x = np.array([[feats[f] for f in features]])
            pred = float(model.predict(x)[0])
            return pred * 100 > threshold_cents

        return gate
    return make_gate_for_token


def simulate_token_v2(pts, gate_for_token):
    """Like simulate_token but with index-based gate."""
    if len(pts) < 2:
        return None
    cash = 0.0; inv = 0.0; n_fills = 0; pulls = 0
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0
    gate = gate_for_token

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i+1]
        spread = ask - bid
        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            cash -= bid_q_price * QUOTE_SIZE; inv += QUOTE_SIZE; n_fills += 1
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            cash += ask_q_price * QUOTE_SIZE; inv -= QUOTE_SIZE; n_fills += 1
        if spread < MIN_SPREAD:
            have_bid_q = have_ask_q = False
            continue
        if i < 1:
            have_bid_q = have_ask_q = False
            continue
        try:
            if gate(i):
                pulls += 1
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
    _, bid_last, ask_last = pts[-1]
    if inv > 0: cash += inv * bid_last
    elif inv < 0: cash -= (-inv) * ask_last
    cash -= n_fills * SLIPPAGE * QUOTE_SIZE
    return {"n_fills": n_fills, "pnl": cash, "pulls": pulls}


def main():
    paths = sorted(glob.glob("data/processed/*.jsonl"))
    print(f"Loading {len(paths)} files…")
    all_streams = {}
    file_token_pairs = {}  # path -> [token_ids]
    for p in paths:
        streams = parse_books(p)
        for tok, pts in streams.items():
            all_streams[tok] = pts
        toks = list(streams.keys())
        if len(toks) == 2 and all(len(streams[t]) >= 2 for t in toks):
            file_token_pairs[p] = toks
    # Build complement map: in each file, the 2 tokens are complements
    complement = {}
    for p, toks in file_token_pairs.items():
        a, b = toks
        complement[a] = b
        complement[b] = a
    print(f"  {len(all_streams)} tokens, {len(complement)} complement pairs")

    # Heuristic baseline
    print("\n=== Heuristic ===")
    run_strategy(all_streams, heuristic_gate, "heuristic_drift_ratio_0.30")

    # v1 XGBoost
    print("\n=== v1 XGBoost (current production model) ===")
    from ab_simulate import make_model_gate
    for tau in [2.0, 3.0, 4.0]:
        gate = make_model_gate("_gbm_lab/toxic_flow/artifacts/xgboost.pkl", tau)
        run_strategy(all_streams, gate, f"xgb_v1 τ>{tau:.1f}¢")

    # v2 models
    for mf in ["lightgbm_v2.pkl", "xgboost_v2.pkl", "catboost_v2.pkl"]:
        mp = f"_gbm_lab/toxic_flow/artifacts/{mf}"
        if not os.path.exists(mp):
            continue
        print(f"\n=== {mf.replace('.pkl','')} ===")
        make_gate = make_model_gate_v2(mp, 0, complement, all_streams)
        for tau in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            make_gate = make_model_gate_v2(mp, tau, complement, all_streams)
            total_pnl = 0.0; total_fills = 0; total_pulls = 0; n_active = 0
            for tok, pts in all_streams.items():
                gate = make_gate(tok, pts)
                r = simulate_token_v2(pts, gate)
                if r is None or r["n_fills"] == 0:
                    continue
                total_pnl += r["pnl"]
                total_fills += r["n_fills"]
                total_pulls += r["pulls"]
                n_active += 1
            avg = total_pnl / max(1, total_fills) * 100
            print(f"  {mf.replace('.pkl',''):>20} τ>{tau:.1f}¢: fills={total_fills:>5} "
                  f"pulls={total_pulls:>6} pnl=${total_pnl:>+8.2f} ¢/fill={avg:>+6.2f}")


if __name__ == "__main__":
    main()
