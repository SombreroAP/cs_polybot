"""
Build a supervised dataset for toxic-flow prediction.

Task: regression. Predict |mid drift in next 60s| from current observables.

For each book observation in our processed recordings:
  - Compute features at time t (current spread, mid, past drifts at multiple
    windows, velocity, acceleration, time-of-day)
  - Compute target: |mid at t+60s - mid at t| (absolute future drift in 60s)
  - Filter out observations near the edge of a token's stream (no future data)

Output: data/training/toxic_flow.parquet  (one row per book observation)

Run: python _gbm_lab/toxic_flow/build_dataset.py
"""
import os, sys, glob, json, bisect
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone


FUTURE_HORIZON_S = 60.0
MIN_HISTORY_S = 30.0   # need at least this much past to compute drifts
SAMPLE_EVERY_S = 5.0   # don't emit a row for every micro-update; sample
PAST_WINDOWS_S = [5, 10, 30, 60, 300]


def parse_books(path):
    """Yield (ts, tokenId, bid, ask) tuples sorted by ts per token."""
    by_token = defaultdict(list)
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
            ts = r.get("ts")
            if tok and bid is not None and ask is not None and ts is not None:
                by_token[tok].append((float(ts), float(bid), float(ask)))
    for tok in by_token:
        by_token[tok].sort()
    return by_token


def make_features(stream):
    """
    stream: sorted list of (ts, bid, ask) for one token.
    Yields dicts (one per emitted sample) with features + target.
    """
    if len(stream) < 4:
        return
    ts_arr = np.array([s[0] for s in stream])
    bid_arr = np.array([s[1] for s in stream])
    ask_arr = np.array([s[2] for s in stream])
    mid_arr = (bid_arr + ask_arr) / 2.0

    def mid_at(t):
        i = bisect.bisect_right(ts_arr, t) - 1
        if i < 0:
            return None
        return float(mid_arr[i])

    last_emit = -1e9
    t0_token = ts_arr[0]
    t_max = ts_arr[-1]

    for i in range(len(stream)):
        t = float(ts_arr[i])
        bid = float(bid_arr[i])
        ask = float(ask_arr[i])
        if ask <= bid:
            continue
        # Require enough past history
        if t - t0_token < MIN_HISTORY_S:
            continue
        # Need future data
        if t_max - t < FUTURE_HORIZON_S:
            continue
        # Sample stride
        if t - last_emit < SAMPLE_EVERY_S:
            continue
        last_emit = t

        mid = float(mid_arr[i])
        spread = ask - bid

        # Multi-window past drifts
        feats = {
            "ts": t,
            "bid": bid,
            "ask": ask,
            "mid": mid,
            "spread": spread,
        }
        for w in PAST_WINDOWS_S:
            m_past = mid_at(t - w)
            feats[f"past_drift_{w}s"] = (mid - m_past) if m_past is not None else 0.0
            feats[f"abs_past_drift_{w}s"] = abs(feats[f"past_drift_{w}s"])

        # Velocity (drift over 1s — proxy: nearest prior point)
        m_prev = mid_at(t - 1.0)
        feats["velocity_1s"] = (mid - m_prev) if m_prev is not None else 0.0

        # Acceleration (delta-of-delta over 10s)
        m_a = mid_at(t - 5.0)
        m_b = mid_at(t - 10.0)
        if m_a is not None and m_b is not None:
            feats["accel_5_10"] = (mid - m_a) - (m_a - m_b)
        else:
            feats["accel_5_10"] = 0.0

        # Time-of-day (UTC hour, sin/cos to avoid wrap discontinuity)
        hour = datetime.fromtimestamp(t, tz=timezone.utc).hour + \
               datetime.fromtimestamp(t, tz=timezone.utc).minute / 60.0
        feats["tod_sin"] = float(np.sin(2 * np.pi * hour / 24))
        feats["tod_cos"] = float(np.cos(2 * np.pi * hour / 24))

        # Target: |mid in 60s - mid_now|
        m_future = mid_at(t + FUTURE_HORIZON_S)
        if m_future is None:
            continue
        feats["target_abs_drift_60s"] = abs(m_future - mid)
        feats["target_signed_drift_60s"] = m_future - mid

        yield feats


def main():
    paths = sorted(glob.glob("data/processed/*.jsonl"))
    print(f"Found {len(paths)} processed files")
    rows = []
    skipped_thin = 0
    for i, p in enumerate(paths):
        if i and i % 200 == 0:
            print(f"  [{i}/{len(paths)}]  rows so far: {len(rows)}")
        try:
            streams = parse_books(p)
        except Exception as e:
            continue
        for tok, stream in streams.items():
            try:
                for row in make_features(stream):
                    row["token_id"] = tok
                    rows.append(row)
            except Exception:
                skipped_thin += 1
                continue

    print(f"\nTotal rows: {len(rows)}  skipped_thin: {skipped_thin}")
    if not rows:
        print("No rows emitted — check data path and filters.")
        return

    df = pd.DataFrame(rows)
    print(f"\nFeature columns: {list(df.columns)}")
    print(f"Target stats: mean={df['target_abs_drift_60s'].mean()*100:.2f}c  "
          f"p50={df['target_abs_drift_60s'].median()*100:.2f}c  "
          f"p90={df['target_abs_drift_60s'].quantile(0.9)*100:.2f}c  "
          f"p99={df['target_abs_drift_60s'].quantile(0.99)*100:.2f}c")

    out_path = "data/training/toxic_flow.parquet"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"\nWrote {out_path} ({len(df):,} rows, {os.path.getsize(out_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
