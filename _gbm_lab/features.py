"""
Feature engineering for the GBM betting model.

Goes beyond the basic point-in-time features by walking each match's SFT rows
in chronological order and computing rolling / context features:
  - recent round-score volatility (lead changes in last K rounds)
  - momentum direction (sign of last 3 round outcomes)
  - market trajectory (price change vs N seconds ago)
  - pre-match prior (first market price we saw for this match)
  - prior-deviation (how much has the market moved from its opening price)
  - time-into-match (seconds since first decision in this match)
  - round-of-map (round_number normalised to [0,1])
  - map-momentum (winning current map by margin)
"""
import json
import math
from collections import defaultdict, deque
from typing import Iterator


def _safe(x, default=0.0):
    if x is None:
        return default
    try:
        return float(x)
    except Exception:
        return default


def _first_market_prices(market: dict):
    """Return (bid, ask) of the first token in market_before — defaults 0.5/0.5."""
    if not market:
        return 0.5, 0.5
    first = next(iter(market.values()), {})
    return _safe(first.get("bid"), 0.5), _safe(first.get("ask"), 0.5)


# ── Per-match rolling state ─────────────────────────────────────────────────
class MatchContext:
    """Maintains rolling stats for a single match as we walk through its rows."""

    def __init__(self):
        self.first_ts = None
        self.first_bid = None
        self.first_ask = None
        # round-outcome history: +1 if A won the round, -1 if B
        self.round_outcomes: deque = deque(maxlen=10)
        self.last_score_a = 0
        self.last_score_b = 0
        self.last_match_score_a = 0
        self.last_match_score_b = 0
        # price history: list of (ts, mid)
        self.price_hist: deque = deque(maxlen=20)
        self.lead_changes_count = 0
        self.last_score_diff_sign = 0

    def update_and_extract(self, row: dict) -> dict:
        """Update state from the row and return a dict of extra features."""
        state = row.get("state") or {}
        t1 = state.get("team_one") or {}
        t2 = state.get("team_two") or {}
        score_a = int(_safe(t1.get("score")))
        score_b = int(_safe(t2.get("score")))
        match_score_a = int(_safe(t1.get("match_score")))
        match_score_b = int(_safe(t2.get("match_score")))
        ts = _safe(row.get("decision_ts"))
        bid, ask = _first_market_prices(row.get("market_before") or {})
        mid = (bid + ask) / 2

        # Capture extra price-history for richer features
        self._last_mid_for_velocity = getattr(self, "_last_mid_for_velocity", mid)
        self._last_ts_for_velocity = getattr(self, "_last_ts_for_velocity", ts)
        # n-step lookback (raw rows) — useful when ts gaps are uneven
        self._mid_history = getattr(self, "_mid_history", deque(maxlen=10))
        self._bid_history = getattr(self, "_bid_history", deque(maxlen=10))
        self._ask_history = getattr(self, "_ask_history", deque(maxlen=10))
        self._mid_history.append(mid)
        self._bid_history.append(bid)
        self._ask_history.append(ask)

        # First-time init
        if self.first_ts is None:
            self.first_ts = ts
            self.first_bid = bid
            self.first_ask = ask

        # Round-outcome tracking (positive delta = A won this round)
        d_a = score_a - self.last_score_a
        d_b = score_b - self.last_score_b
        if d_a > 0 and d_b == 0:
            self.round_outcomes.append(+1)
        elif d_b > 0 and d_a == 0:
            self.round_outcomes.append(-1)
        # else: no round flip in this row, skip

        # Lead-change counter
        sd = score_a - score_b
        sd_sign = (sd > 0) - (sd < 0)
        if self.last_score_diff_sign != 0 and sd_sign != 0 and sd_sign != self.last_score_diff_sign:
            self.lead_changes_count += 1
        if sd_sign != 0:
            self.last_score_diff_sign = sd_sign

        self.last_score_a = score_a
        self.last_score_b = score_b
        self.last_match_score_a = match_score_a
        self.last_match_score_b = match_score_b

        # Price-trajectory window: drop entries older than 60s
        self.price_hist.append((ts, mid))
        cutoff = ts - 60
        while self.price_hist and self.price_hist[0][0] < cutoff:
            self.price_hist.popleft()
        if len(self.price_hist) >= 2:
            p0 = self.price_hist[0][1]
            mid_change_60s = mid - p0
        else:
            mid_change_60s = 0.0

        # Momentum: net of last 3 round outcomes
        recent3 = list(self.round_outcomes)[-3:]
        momentum_3 = sum(recent3)
        momentum_5 = sum(list(self.round_outcomes)[-5:])
        # Round-outcome volatility: how many sign flips in the window
        flips = 0
        for i in range(1, len(self.round_outcomes)):
            if self.round_outcomes[i] != self.round_outcomes[i - 1]:
                flips += 1
        volatility = flips / max(1, len(self.round_outcomes) - 1)

        round_number = int(_safe(state.get("round_number")))
        round_of_map = min(1.0, round_number / 24.0)  # 0..1

        # Map margin within current map
        map_margin = score_a - score_b

        # Series state (Bo3): match_score 1-0 = leading series, 0-1 = trailing, etc.
        series_lead = match_score_a - match_score_b
        # Is this a closeout map (one team about to win the series)?
        series_total = match_score_a + match_score_b
        closeout_map = 1 if (match_score_a == 1 or match_score_b == 1) and series_total == 1 else 0

        # Multi-step price-history features (row-based lookback for stability)
        mids = list(self._mid_history)
        bids = list(self._bid_history)
        asks = list(self._ask_history)
        mid_lag_1 = mids[-2] if len(mids) >= 2 else mid
        mid_lag_3 = mids[-4] if len(mids) >= 4 else mid
        mid_lag_5 = mids[-6] if len(mids) >= 6 else mid
        mid_change_1 = mid - mid_lag_1
        mid_change_3 = mid - mid_lag_3
        mid_change_5 = mid - mid_lag_5
        # Spread dynamics
        prev_spread = (asks[-2] - bids[-2]) if len(asks) >= 2 else (ask - bid)
        spread_change_1 = (ask - bid) - prev_spread
        # Mid-revert: how far is current mid from its rolling mean?
        if len(mids) >= 3:
            roll_mean = sum(mids[-5:]) / len(mids[-5:])
            mid_revert = mid - roll_mean
        else:
            mid_revert = 0.0
        # Velocity (per-row, sign matters)
        if len(mids) >= 2:
            mid_velocity = mid - mids[-2]
        else:
            mid_velocity = 0.0
        # Acceleration (change in velocity)
        if len(mids) >= 3:
            mid_accel = (mid - mids[-2]) - (mids[-2] - mids[-3])
        else:
            mid_accel = 0.0
        # Bid/ask asymmetry: when ask moves faster than bid (or vice versa)
        if len(bids) >= 2 and len(asks) >= 2:
            ask_velocity = ask - asks[-2]
            bid_velocity = bid - bids[-2]
            ba_asymmetry = ask_velocity - bid_velocity
        else:
            ba_asymmetry = 0.0

        return {
            "ctx_time_in_match_s": ts - self.first_ts,
            "ctx_pre_match_bid": self.first_bid,
            "ctx_pre_match_ask": self.first_ask,
            "ctx_pre_match_mid": (self.first_bid + self.first_ask) / 2,
            "ctx_mid_deviation_from_prior": mid - (self.first_bid + self.first_ask) / 2,
            "ctx_mid_change_60s": mid_change_60s,
            "ctx_momentum_last3": momentum_3,
            "ctx_momentum_last5": momentum_5,
            "ctx_lead_changes_total": self.lead_changes_count,
            "ctx_round_volatility": volatility,
            "ctx_round_of_map_norm": round_of_map,
            "ctx_map_margin": map_margin,
            "ctx_series_lead": series_lead,
            "ctx_closeout_map": closeout_map,
            # NEW: price-history features
            "ctx_mid_lag_1": mid_lag_1,
            "ctx_mid_lag_3": mid_lag_3,
            "ctx_mid_lag_5": mid_lag_5,
            "ctx_mid_change_1": mid_change_1,
            "ctx_mid_change_3": mid_change_3,
            "ctx_mid_change_5": mid_change_5,
            "ctx_spread_change_1": spread_change_1,
            "ctx_mid_revert": mid_revert,
            "ctx_mid_velocity": mid_velocity,
            "ctx_mid_accel": mid_accel,
            "ctx_ba_asymmetry": ba_asymmetry,
        }


CTX_FEATURES = [
    "ctx_time_in_match_s", "ctx_pre_match_bid", "ctx_pre_match_ask", "ctx_pre_match_mid",
    "ctx_mid_deviation_from_prior", "ctx_mid_change_60s",
    "ctx_momentum_last3", "ctx_momentum_last5",
    "ctx_lead_changes_total", "ctx_round_volatility",
    "ctx_round_of_map_norm", "ctx_map_margin", "ctx_series_lead", "ctx_closeout_map",
]
# Price-history features still computed but excluded from CTX_FEATURES until we
# have enough training rows to support them without overfitting:
EXTRA_PRICE_HISTORY_FEATURES = [
    "ctx_mid_lag_1", "ctx_mid_lag_3", "ctx_mid_lag_5",
    "ctx_mid_change_1", "ctx_mid_change_3", "ctx_mid_change_5",
    "ctx_spread_change_1", "ctx_mid_revert", "ctx_mid_velocity",
    "ctx_mid_accel", "ctx_ba_asymmetry",
]


def extract_with_context(sft_path: str):
    """Generator: yields (basic_row_dict, context_features_dict) pairs in time order per match.

    Walks the SFT in append order (already roughly time-ordered within each match
    because that's how the bot wrote them), maintaining MatchContext per match_id.
    """
    contexts: dict[str, MatchContext] = defaultdict(MatchContext)
    with open(sft_path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except Exception:
                continue
            mid = str(row.get("match_id", ""))
            ctx_feats = contexts[mid].update_and_extract(row)
            yield row, ctx_feats
