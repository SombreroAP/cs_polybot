"""
A/B PnL simulation — drop a trained model into mm_simulator as the toxic-flow
detector, compare per-fill PnL to the original heuristic.

Strategy variants:
  - HEURISTIC: original `drift_past_30s > drift_ratio × spread` rule
  - MODEL:     predict next 60s |mid drift|; threshold by τ

Both run on the SAME 26-day book stream, time-ordered. Reports per-fill PnL,
fill count, drift-filter pull count.
"""
import os, sys, glob, bisect, json, pickle
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone


# Strategy params (mirror production)
QUOTE_DELTA = -0.02
MIN_SPREAD = 0.10
MAX_INV = 5.0
SLIPPAGE = 0.0005       # cents — realistic Polygon gas
QUOTE_SIZE = 1.0
DRIFT_LOOKBACK_S = 30.0
HEURISTIC_RATIO = 0.30
FUTURE_HORIZON_S = 60.0


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


def features_for_obs(pts_arr, i, t_token_start, ts_arr, mid_arr):
    """Compute features for observation i (matching build_dataset.py)."""
    t = ts_arr[i]
    bid = pts_arr[i][1]; ask = pts_arr[i][2]
    mid = mid_arr[i]
    spread = ask - bid

    def mid_at(t_target):
        j = bisect.bisect_right(ts_arr, t_target) - 1
        if j < 0:
            return None
        return float(mid_arr[j])

    feats = {"spread": spread, "mid": mid}
    for w in [5, 10, 30, 60, 300]:
        m_past = mid_at(t - w)
        v = (mid - m_past) if m_past is not None else 0.0
        feats[f"past_drift_{w}s"] = v
        feats[f"abs_past_drift_{w}s"] = abs(v)
    m_prev = mid_at(t - 1.0)
    feats["velocity_1s"] = (mid - m_prev) if m_prev is not None else 0.0
    m_a = mid_at(t - 5.0); m_b = mid_at(t - 10.0)
    if m_a is not None and m_b is not None:
        feats["accel_5_10"] = (mid - m_a) - (m_a - m_b)
    else:
        feats["accel_5_10"] = 0.0
    hour = datetime.fromtimestamp(t, tz=timezone.utc).hour + \
           datetime.fromtimestamp(t, tz=timezone.utc).minute / 60.0
    feats["tod_sin"] = float(np.sin(2 * np.pi * hour / 24))
    feats["tod_cos"] = float(np.cos(2 * np.pi * hour / 24))
    return feats


def simulate_token(pts, gate_fn):
    """Run MM on one token, using gate_fn(features, spread) → bool (skip if True)."""
    if len(pts) < 2:
        return None
    ts_arr = np.array([p[0] for p in pts])
    mid_arr = np.array([(p[1] + p[2]) / 2 for p in pts])
    cash = 0.0; inv = 0.0; n_fills = 0
    have_bid_q = False; bid_q_price = 0.0
    have_ask_q = False; ask_q_price = 0.0
    t_token_start = ts_arr[0]
    drift_pulls = 0

    for i in range(len(pts) - 1):
        t, bid, ask = pts[i]
        t_next, bid_next, ask_next = pts[i + 1]
        spread = ask - bid

        # Fill detection
        if have_bid_q and ask_next <= bid_q_price + 1e-9:
            cash -= bid_q_price * QUOTE_SIZE; inv += QUOTE_SIZE; n_fills += 1
        if have_ask_q and bid_next >= ask_q_price - 1e-9:
            cash += ask_q_price * QUOTE_SIZE; inv -= QUOTE_SIZE; n_fills += 1

        # Spread filter
        if spread < MIN_SPREAD:
            have_bid_q = have_ask_q = False
            continue
        # Need history to compute features
        if t - t_token_start < 30.0:
            have_bid_q = have_ask_q = False
            continue

        # Compute features and ask gate
        feats = features_for_obs(pts, i, t_token_start, ts_arr, mid_arr)
        if gate_fn(feats, spread):
            drift_pulls += 1
            have_bid_q = have_ask_q = False
            continue

        # Place quotes
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

    return {"n_fills": n_fills, "pnl": cash, "drift_pulls": drift_pulls}


# ── Gate functions ──────────────────────────────────────────────────────
def heuristic_gate(feats, spread):
    """Same logic as production: |past_drift_30s| > drift_ratio × spread."""
    return abs(feats["past_drift_30s"]) > HEURISTIC_RATIO * spread


def make_model_gate(model_path, threshold_cents):
    """Build a gate that predicts future drift and pulls quotes if > threshold."""
    with open(model_path, "rb") as f:
        d = pickle.load(f)
    model = d["model"]
    features = d["features"]

    def gate(feats, spread):
        x = np.array([[feats[f] for f in features]])
        pred = float(model.predict(x)[0])
        return pred * 100 > threshold_cents

    return gate


def make_mlp_gate(model_path, threshold_cents):
    """Build a gate using a PyTorch MLP saved as a .pt file."""
    import torch
    import torch.nn as nn
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    features = ckpt["features"]
    cfg = ckpt["config"]
    in_dim = cfg["in_dim"]
    # Reconstruct: heuristically derive hidden/depth from state_dict shape
    sd = ckpt["state_dict"]
    # Layers are net.0.weight, net.3.weight, net.6.weight, ... (Linear-ReLU-Dropout triplets)
    linear_keys = sorted([k for k in sd.keys() if k.endswith(".weight") and "net" in k])
    # Detect dims
    dims = [sd[linear_keys[0]].shape[1]]  # in_dim
    for k in linear_keys:
        dims.append(sd[k].shape[0])
    # Build a matching MLP
    layers = []
    for i in range(len(dims) - 2):
        layers += [nn.Linear(dims[i], dims[i+1]), nn.ReLU(), nn.Dropout(0.0)]
    layers.append(nn.Linear(dims[-2], dims[-1]))
    model = nn.Sequential(*layers)
    # Map keys back to net.* form
    new_sd = {}
    for i, k in enumerate(linear_keys):
        new_idx = i * 3
        new_sd[f"{new_idx}.weight"] = sd[k]
        new_sd[f"{new_idx}.bias"] = sd[k.replace(".weight", ".bias")]
    model.load_state_dict(new_sd, strict=False)
    model.eval()

    def gate(feats, spread):
        x = torch.tensor([[feats[f] for f in features]], dtype=torch.float32)
        with torch.no_grad():
            pred = float(model(x).item())
        return pred * 100 > threshold_cents

    return gate


def run_strategy(streams, gate_fn, label):
    total_pnl = 0.0; total_fills = 0; total_pulls = 0; tokens_active = 0
    for tok, pts in streams.items():
        r = simulate_token(pts, gate_fn)
        if r is None or r["n_fills"] == 0:
            continue
        total_pnl += r["pnl"]
        total_fills += r["n_fills"]
        total_pulls += r["drift_pulls"]
        tokens_active += 1
    avg = total_pnl / max(1, total_fills) * 100
    print(f"  {label:>40}: fills={total_fills:>5} pulls={total_pulls:>6} "
          f"pnl=${total_pnl:>+8.2f} ¢/fill={avg:>+6.2f}")
    return total_pnl, total_fills, avg


def main():
    paths = sorted(glob.glob("data/processed/*.jsonl"))
    print(f"Loading {len(paths)} book files…")
    all_streams = {}
    for p in paths:
        for tok, pts in parse_books(p).items():
            all_streams[tok] = pts
    print(f"  {len(all_streams)} tokens loaded, "
          f"{sum(len(v) for v in all_streams.values())} book points")

    print("\n" + "="*78)
    print("HEURISTIC baseline (production strategy)")
    print("="*78)
    print(f"  Gate: |past_drift_30s| > {HEURISTIC_RATIO} × spread")
    base_pnl, base_fills, base_avg = run_strategy(
        all_streams, heuristic_gate, "heuristic_drift_ratio_0.30"
    )

    print("\n" + "="*78)
    print("ML MODELS — sweep across drift thresholds")
    print("="*78)
    model_files = ["lightgbm.pkl", "xgboost.pkl", "catboost.pkl", "linreg.pkl"]
    for mf in model_files:
        mp = f"_gbm_lab/toxic_flow/artifacts/{mf}"
        if not os.path.exists(mp):
            continue
        print(f"\n--- {mf.replace('.pkl','')} ---")
        for tau_c in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            gate = make_model_gate(mp, threshold_cents=tau_c)
            label = f"{mf.replace('.pkl','')} τ>{tau_c:.1f}¢"
            try:
                run_strategy(all_streams, gate, label)
            except Exception as e:
                print(f"  {label}: error {e}")

    # MLP
    mlp_path = "_gbm_lab/toxic_flow/artifacts/mlp.pt"
    if os.path.exists(mlp_path):
        print(f"\n--- mlp (PyTorch) ---")
        for tau_c in [2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
            try:
                gate = make_mlp_gate(mlp_path, threshold_cents=tau_c)
                run_strategy(all_streams, gate, f"mlp τ>{tau_c:.1f}¢")
            except Exception as e:
                print(f"  mlp τ>{tau_c:.1f}¢: error {e}")

    print(f"\nBaseline ¢/fill: {base_avg:+.2f}¢  on {base_fills} fills")
    print("Look for model+τ combos with HIGHER ¢/fill AND non-trivial fill count.")


if __name__ == "__main__":
    main()
