"""
Dataset v2 with richer features:
  - All v1 features (spread, mid, multi-window past drifts, velocity, accel, time-of-day)
  - VOLATILITY features: rolling std of mid over 30/60/300s windows
  - DAY-OF-WEEK (one-hot or sin/cos)
  - SPREAD HISTORY: how spread has changed (widening = informed flow signal)
  - COMPLEMENTARY-TOKEN features: same metrics for the OPPOSITE binary outcome.
    Binary markets: mid_A + mid_B ≈ 1, so any move in A is mirrored in B.
    But the SPEED of mirror (lag) can reveal which side has informed flow.

For complementary-token features, we need to know which tokens belong to the
same market. Polymarket binary markets have token_a + token_b summing to ~$1.
We detect pairs from the recordings by matching: same match_id appears as both
a and b tokens in `_META` records, and the matched-pair mids should sum to ~1.
"""
import os, sys, glob, json, bisect, math
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timezone


FUTURE_HORIZON_S = 60.0
MIN_HISTORY_S = 30.0
SAMPLE_EVERY_S = 5.0
PAST_WINDOWS_S = [5, 10, 30, 60, 300]
VOL_WINDOWS_S = [30, 60, 300]


def parse_books_and_meta(path):
    """Return (books_by_token, complement_map: token_id -> other_token_id).

    Within a single processed file we have ALL book records for ONE match.
    Binary markets have exactly 2 tokens (YES + NO). If the file contains
    exactly 2 tokens with non-trivial book data, they are complements.
    """
    by_token = defaultdict(list)
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("type") == "book":
                tok = r.get("tokenId")
                bid = r.get("bid"); ask = r.get("ask"); ts = r.get("ts")
                if tok and bid is not None and ask is not None and ts is not None:
                    by_token[tok].append((float(ts), float(bid), float(ask)))
    for tok in by_token:
        by_token[tok].sort()
    # Build complement map: if exactly 2 tokens present, pair them
    complement = {}
    tokens_with_data = [t for t, s in by_token.items() if len(s) >= 2]
    if len(tokens_with_data) == 2:
        a, b = tokens_with_data
        complement[a] = b
        complement[b] = a
    return dict(by_token), complement


def make_features(stream, comp_stream=None):
    """
    stream: sorted [(ts, bid, ask)] for one token.
    comp_stream: optional sorted [(ts, bid, ask)] for the complementary token.
    """
    if len(stream) < 4:
        return
    ts_arr = np.array([s[0] for s in stream])
    bid_arr = np.array([s[1] for s in stream])
    ask_arr = np.array([s[2] for s in stream])
    mid_arr = (bid_arr + ask_arr) / 2.0

    if comp_stream and len(comp_stream) >= 2:
        cts_arr = np.array([s[0] for s in comp_stream])
        cmid_arr = (np.array([s[1] for s in comp_stream]) +
                    np.array([s[2] for s in comp_stream])) / 2.0
    else:
        cts_arr = None
        cmid_arr = None

    def mid_at(t):
        i = bisect.bisect_right(ts_arr, t) - 1
        return float(mid_arr[i]) if i >= 0 else None

    def cmid_at(t):
        if cts_arr is None:
            return None
        i = bisect.bisect_right(cts_arr, t) - 1
        return float(cmid_arr[i]) if i >= 0 else None

    def vol_window(t_now, window):
        """Std of mids within (t-window, t]."""
        i_lo = bisect.bisect_left(ts_arr, t_now - window)
        i_hi = bisect.bisect_right(ts_arr, t_now)
        if i_hi - i_lo < 2:
            return 0.0
        return float(np.std(mid_arr[i_lo:i_hi]))

    last_emit = -1e9
    t0_token = ts_arr[0]
    t_max = ts_arr[-1]

    for i in range(len(stream)):
        t = float(ts_arr[i])
        bid = float(bid_arr[i]); ask = float(ask_arr[i])
        if ask <= bid:
            continue
        if t - t0_token < MIN_HISTORY_S:
            continue
        if t_max - t < FUTURE_HORIZON_S:
            continue
        if t - last_emit < SAMPLE_EVERY_S:
            continue
        last_emit = t

        mid = float(mid_arr[i])
        spread = ask - bid

        feats = {
            "ts": t,
            "bid": bid, "ask": ask, "mid": mid, "spread": spread,
        }
        # Past drifts
        for w in PAST_WINDOWS_S:
            m_past = mid_at(t - w)
            v = (mid - m_past) if m_past is not None else 0.0
            feats[f"past_drift_{w}s"] = v
            feats[f"abs_past_drift_{w}s"] = abs(v)
        m_prev = mid_at(t - 1.0)
        feats["velocity_1s"] = (mid - m_prev) if m_prev is not None else 0.0
        m_a = mid_at(t - 5.0); m_b = mid_at(t - 10.0)
        feats["accel_5_10"] = ((mid - m_a) - (m_a - m_b)) if (m_a is not None and m_b is not None) else 0.0

        # Volatility
        for w in VOL_WINDOWS_S:
            feats[f"vol_{w}s"] = vol_window(t, w)

        # Spread velocity (widening = informed flow)
        # nearest prior spread sample 30s ago — approximate using ask-bid at that bisect index
        i_30 = bisect.bisect_right(ts_arr, t - 30.0) - 1
        if i_30 >= 0:
            prev_spread = ask_arr[i_30] - bid_arr[i_30]
            feats["spread_change_30s"] = spread - float(prev_spread)
        else:
            feats["spread_change_30s"] = 0.0

        # Time-of-day + day-of-week
        dt = datetime.fromtimestamp(t, tz=timezone.utc)
        hour = dt.hour + dt.minute / 60.0
        feats["tod_sin"] = math.sin(2 * math.pi * hour / 24)
        feats["tod_cos"] = math.cos(2 * math.pi * hour / 24)
        dow = dt.weekday()  # 0=Mon
        feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
        feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)

        # COMPLEMENTARY-TOKEN features
        if cts_arr is not None:
            cm_now = cmid_at(t)
            cm_30 = cmid_at(t - 30.0)
            if cm_now is not None and cm_30 is not None:
                feats["comp_drift_30s"] = cm_now - cm_30
                feats["abs_comp_drift_30s"] = abs(cm_now - cm_30)
                feats["comp_mid"] = cm_now
                feats["comp_sum"] = mid + cm_now  # should be ~1 in healthy binary
                feats["comp_sum_dev"] = abs((mid + cm_now) - 1.0)
            else:
                feats["comp_drift_30s"] = 0.0
                feats["abs_comp_drift_30s"] = 0.0
                feats["comp_mid"] = 0.5
                feats["comp_sum"] = 1.0
                feats["comp_sum_dev"] = 0.0
        else:
            feats["comp_drift_30s"] = 0.0
            feats["abs_comp_drift_30s"] = 0.0
            feats["comp_mid"] = 0.5
            feats["comp_sum"] = 1.0
            feats["comp_sum_dev"] = 0.0

        # Target
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
    for i, p in enumerate(paths):
        if i and i % 200 == 0:
            print(f"  [{i}/{len(paths)}]  rows: {len(rows)}")
        try:
            streams, complement = parse_books_and_meta(p)
        except Exception:
            continue
        for tok, stream in streams.items():
            comp_tok = complement.get(tok)
            comp_stream = streams.get(comp_tok) if comp_tok else None
            try:
                for row in make_features(stream, comp_stream):
                    row["token_id"] = tok
                    rows.append(row)
            except Exception:
                continue

    print(f"\nTotal rows: {len(rows)}")
    if not rows:
        print("No rows.")
        return
    df = pd.DataFrame(rows)
    print(f"Columns: {len(df.columns)}  features: {[c for c in df.columns if not c.startswith('target') and c not in ('ts','token_id','bid','ask','mid')]}")
    print(f"Target: mean={df['target_abs_drift_60s'].mean()*100:.2f}c  "
          f"p50={df['target_abs_drift_60s'].median()*100:.2f}c  "
          f"p90={df['target_abs_drift_60s'].quantile(0.9)*100:.2f}c")

    out = "data/training/toxic_flow_v2.parquet"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"Wrote {out} ({len(df):,} rows, {os.path.getsize(out)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
