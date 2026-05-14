"""
Lazy-loaded toxic-flow predictor.

Wraps a trained XGBoost regressor (or other sklearn-compatible model) that
takes a feature vector and predicts the absolute mid drift over the next 60
seconds. Used by mm_strategy.py when drift_mode="model".

Falls back to a no-op (returns 0 drift) if the model file is missing —
mm_strategy treats that as "no toxic signal, allow quoting".
"""
from __future__ import annotations

import logging
import math
import os
import pickle
import threading
from datetime import datetime, timezone
from typing import Optional


log = logging.getLogger("mm_toxic")


# Features must match the model's training-time feature list. The
# predictor reads `features` from the pickle so both v1 and v2 work.
EXPECTED_FEATURES = [
    "spread", "mid",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "tod_sin", "tod_cos",
]

# v2 feature set (used when model is catboost_v2 etc)
V2_FEATURES = [
    "spread",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "vol_30s", "vol_60s", "vol_300s",
    "spread_change_30s",
    "tod_sin", "tod_cos",
    "dow_sin", "dow_cos",
    "comp_drift_30s", "abs_comp_drift_30s", "comp_mid",
    "comp_sum", "comp_sum_dev",
]


class _ModelHandle:
    def __init__(self, path: str):
        self.path = path
        self.model = None
        self.features = None
        self.load_err: Optional[str] = None
        self._lock = threading.Lock()
        self._maybe_load()

    def _maybe_load(self):
        if not os.path.exists(self.path):
            self.load_err = f"model file not found: {self.path}"
            log.warning(f"[MM-TOXIC] {self.load_err}")
            return
        try:
            with open(self.path, "rb") as f:
                d = pickle.load(f)
            self.model = d["model"]
            self.features = d.get("features", EXPECTED_FEATURES)
            log.info(f"[MM-TOXIC] loaded {self.path}  features={len(self.features)}")
        except Exception as e:
            self.load_err = str(e)
            log.warning(f"[MM-TOXIC] failed to load {self.path}: {e}")

    def predict_drift(self, feats: dict) -> Optional[float]:
        """Return predicted |mid drift in next 60s| (in $) or None if no model."""
        if self.model is None:
            return None
        try:
            import numpy as np
            x = np.array([[float(feats.get(f, 0.0)) for f in self.features]])
            return float(self.model.predict(x)[0])
        except Exception as e:
            log.debug(f"[MM-TOXIC] predict error: {e}")
            return None


# Singleton accessor
_singleton: Optional[_ModelHandle] = None
_singleton_lock = threading.Lock()


def get_predictor(model_path: str) -> _ModelHandle:
    global _singleton
    with _singleton_lock:
        if _singleton is None or _singleton.path != model_path:
            _singleton = _ModelHandle(model_path)
    return _singleton


# ────────────────────────────────────────────────────────────────────────
# Feature extraction from a rolling mid history
# ────────────────────────────────────────────────────────────────────────
def extract_features(mid_history, current_bid: float, current_ask: float,
                     t_now: float, *, spread_history=None,
                     comp_mid_history=None) -> dict:
    """Build a feature dict for the toxic-flow predictor.

    mid_history: list of (ts, mid) tuples, oldest→newest.
    spread_history (optional): list of (ts, spread) — enables spread_change_30s.
    comp_mid_history (optional): same shape for the complementary token —
                                 enables comp_drift_30s / comp_sum / comp_sum_dev.
    """
    mid = (current_bid + current_ask) / 2
    spread = current_ask - current_bid

    def latest_le(history, t_target):
        if not history:
            return None
        best = None
        for ts, m in history:
            if ts <= t_target:
                best = m
            else:
                break
        return best

    def std_in_window(history, t_now, window):
        vals = [m for ts, m in history if t_now - window <= ts <= t_now]
        if len(vals) < 2:
            return 0.0
        import statistics
        return float(statistics.pstdev(vals))

    feats = {"spread": spread, "mid": mid}
    for w in [5, 10, 30, 60, 300]:
        past = latest_le(mid_history, t_now - w)
        v = (mid - past) if past is not None else 0.0
        feats[f"past_drift_{w}s"] = v
        feats[f"abs_past_drift_{w}s"] = abs(v)

    prev = latest_le(mid_history, t_now - 1.0)
    feats["velocity_1s"] = (mid - prev) if prev is not None else 0.0

    a5 = latest_le(mid_history, t_now - 5.0)
    a10 = latest_le(mid_history, t_now - 10.0)
    feats["accel_5_10"] = ((mid - a5) - (a5 - a10)) if (a5 is not None and a10 is not None) else 0.0

    # v2 features
    for w in [30, 60, 300]:
        feats[f"vol_{w}s"] = std_in_window(mid_history, t_now, w)

    if spread_history:
        prev_spread = latest_le(spread_history, t_now - 30.0)
        feats["spread_change_30s"] = spread - prev_spread if prev_spread is not None else 0.0
    else:
        feats["spread_change_30s"] = 0.0

    dt = datetime.fromtimestamp(t_now, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    feats["tod_sin"] = math.sin(2 * math.pi * hour / 24)
    feats["tod_cos"] = math.cos(2 * math.pi * hour / 24)
    dow = dt.weekday()
    feats["dow_sin"] = math.sin(2 * math.pi * dow / 7)
    feats["dow_cos"] = math.cos(2 * math.pi * dow / 7)

    if comp_mid_history:
        cm_now = latest_le(comp_mid_history, t_now)
        cm_30 = latest_le(comp_mid_history, t_now - 30.0)
        if cm_now is not None and cm_30 is not None:
            feats["comp_drift_30s"] = cm_now - cm_30
            feats["abs_comp_drift_30s"] = abs(cm_now - cm_30)
            feats["comp_mid"] = cm_now
            feats["comp_sum"] = mid + cm_now
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

    return feats
