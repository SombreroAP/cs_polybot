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


# Features must match what _gbm_lab/toxic_flow/build_dataset.py produced
EXPECTED_FEATURES = [
    "spread", "mid",
    "past_drift_5s", "past_drift_10s", "past_drift_30s",
    "past_drift_60s", "past_drift_300s",
    "abs_past_drift_5s", "abs_past_drift_10s", "abs_past_drift_30s",
    "abs_past_drift_60s", "abs_past_drift_300s",
    "velocity_1s", "accel_5_10",
    "tod_sin", "tod_cos",
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
                     t_now: float) -> dict:
    """Build a feature dict from the strategy's rolling mid history.

    mid_history: list of (ts, mid) tuples, ordered oldest→newest, last
                 entry being t_now.
    """
    mid = (current_bid + current_ask) / 2
    spread = current_ask - current_bid

    def mid_at(t_target):
        # Find latest mid with ts <= t_target
        best = None
        for ts, m in mid_history:
            if ts <= t_target:
                best = m
            else:
                break
        return best

    feats = {"spread": spread, "mid": mid}
    for w in [5, 10, 30, 60, 300]:
        past = mid_at(t_now - w)
        v = (mid - past) if past is not None else 0.0
        feats[f"past_drift_{w}s"] = v
        feats[f"abs_past_drift_{w}s"] = abs(v)

    prev = mid_at(t_now - 1.0)
    feats["velocity_1s"] = (mid - prev) if prev is not None else 0.0

    a5 = mid_at(t_now - 5.0); a10 = mid_at(t_now - 10.0)
    if a5 is not None and a10 is not None:
        feats["accel_5_10"] = (mid - a5) - (a5 - a10)
    else:
        feats["accel_5_10"] = 0.0

    dt = datetime.fromtimestamp(t_now, tz=timezone.utc)
    hour = dt.hour + dt.minute / 60.0
    feats["tod_sin"] = math.sin(2 * math.pi * hour / 24)
    feats["tod_cos"] = math.cos(2 * math.pi * hour / 24)

    return feats
