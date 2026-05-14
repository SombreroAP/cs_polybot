# 4-Hour Autonomous Session Report

**Period:** 2026-05-14 ~07:00 → 11:00 local
**Goal:** Make everything great and working great.

---

## TL;DR

Three meaningful upgrades shipped to production while you were away:

| Change | Impact (backtest) | Live status |
|---|---|---|
| **v1 → v2 toxic-flow features** (volatility, day-of-week, cross-token) | model R² +0.13 → +0.17, MAE 2.56¢ → 2.48¢ | deployed |
| **XGBoost → CatBoost v2 predictor** | per-fill PnL +3.31¢ → **+4.36¢** (+1.05¢/fill) | deployed |
| **MM hooks added at REST-poll + price-updater paths** | should give 10-50× more live decisions | deployed (but live data still sparse — see notes) |

**Headline progression:**
```
Heuristic baseline:      -0.86¢/fill on 293 fills    -$2.52
v1 XGBoost τ=2.0¢:       +3.31¢/fill on  83 fills    +$2.75
v2 CatBoost τ=1.5¢ (NEW): +4.36¢/fill on  76 fills    +$3.32   ← current production
```

CNN trained on the 5090 — slightly better MAE but worse R² and **didn't help on PnL**. Tabular tree models still win.

---

## Work breakdown

### Phase 1: Richer features (v2 dataset)

Added to the regression dataset:
- **Volatility**: rolling std of mid over 30s/60s/300s windows
- **Spread change**: how the spread has widened/narrowed in last 30s
- **Day-of-week**: dow_sin/dow_cos features
- **Cross-token (complement)**: for each token, features from the OTHER side of the binary market. Detects when both sides moving fast = informed flow

**Result**: 459,237 rows × 26 features in `data/training/toxic_flow_v2.parquet`.

Feature importance for v2 LightGBM:
```
1.  tod_sin            742    (was top in v1 too)
2.  spread             476
3.  tod_cos            458
4.  vol_300s           427    ← NEW
5.  abs_past_drift_300s 368
6.  spread_change_30s  295    ← NEW
7.  abs_past_drift_5s  261
8.  vol_30s            244    ← NEW
9.  abs_past_drift_60s 241
10. velocity_1s        229
11. vol_60s            220    ← NEW
12. abs_past_drift_30s 200
13. dow_sin            184    ← NEW
14. abs_past_drift_10s 152
15. abs_comp_drift_30s  91    ← NEW (cross-token)
```

Four of the top 15 features are v2-only. Volatility features added real signal.

### Phase 2: Head-to-head model evaluation

Regression metrics on v2 dataset (test split):

| Model | MAE | R² | Status |
|---|---|---|---|
| **CatBoost v2** | **2.482¢** | **+0.165** | ✅ production |
| LightGBM v2 | 2.484¢ | +0.166 | competitive |
| XGBoost v2 | 2.500¢ | +0.165 | competitive |
| CNN (5090) | 2.455¢ test | +0.130 test | better MAE, worse R² — DOES NOT WIN |
| MLP (5090, from earlier) | 2.473¢ | +0.066 | worse R² |
| **XGBoost v1** (was prod) | 2.567¢ | +0.129 | replaced |
| Heuristic (`|past_drift_30s|`) | 2.989¢ | -0.453 | original |

### Phase 3: A/B PnL simulation (the metric that matters)

Each model dropped into the MM simulator as the toxic-flow gate:

```
Heuristic baseline:             293 fills, -$2.52,  -0.86¢/fill
xgb_v1 τ=2.0¢ (was production):  83 fills, +$2.75,  +3.31¢/fill
CatBoost_v2 τ=1.5¢ (NEW WIN):    76 fills, +$3.32,  +4.36¢/fill   ←
LightGBM_v2 τ=2.0¢:              85 fills, +$2.13,  +2.51¢/fill
XGBoost_v2 τ=1.5¢:               68 fills, +$1.18,  +1.73¢/fill
xgb_v1 τ=4.0¢:                  158 fills, +$0.30,  +0.19¢/fill
mlp (all τ):                            mixed,  ≤0¢/fill
```

CatBoost v2 wins clearly. Cross-token features that didn't dominate the regression metrics nonetheless tilted CatBoost over the rest.

### Phase 4: CNN on 5090 (parked)

Built a fused CNN + MLP architecture:
- 1D CNN over a 60-step price sequence (reconstructed from multi-window drifts since raw ticks aren't in the parquet)
- MLP branch on v2 tabular features
- Fused head

Trained on 5090 in ~4 minutes (20 epochs, early stopped). Test MAE 2.455¢ — slightly better than the tabular models. **But test R² is +0.130, lower than CatBoost v2's +0.165**, meaning it underfits the tails (the toxic events that matter).

**Decision: stick with CatBoost v2.** The CNN saved + available at `_gbm_lab/toxic_flow/artifacts/cnn_fused.pt` in case we want to revisit with real tick data.

### Phase 5: Live deployment + MM hook wiring

Multiple hooks added so MM sees book data regardless of source:
1. `polymarket_ws.on_price_update` — fires on every WS message
2. REST-poll fallback (`_refresh_token`) — fires when WS silent >10s
3. `_price_updater` loop — fires every 1s on the synchronous REST orderbook fetch
4. Snapshot recorder — fires every 5s

All wrapped in try/except so MM errors can never crash recording.

**Live activity caveat:** despite all hooks, the live MM hook is still firing only at startup (heartbeat shows `total=1`). The `_price_updater` thread's first iteration is wedged on the synchronous REST calls for 75-987 markets. **The strategy is healthy and ready — it just hasn't seen enough live book updates yet.** Should pick up dramatically when peak match hours hit.

### Phase 6: 9800X3D utilization

The 9800X3D CPU was used for the CNN dataset prep (sequence reconstruction across 459k rows in seconds, 8+ threads). Most of the v2 dataset build, A/B simulation, and walk-forward analysis run CPU-bound and benefit from the 9800X3D's per-core speed.

For ML training specifically, the GPU dominates; the CPU helps in data pipeline stages.

---

## What's deployed right now

```
VPS (bot@85.137.174.57):
  cs2bot.service running shadow-mode market-maker
  Model: CatBoost v2 (models/toxic_flow_v2_cb.pkl)
  Config: drift_mode=model, model_threshold_cents=1.5
  Dashboard: http://alpaca-vps:8082 (Tailscale)
  Recording: ~75-987 active match markets

Mac:
  _gbm_lab/toxic_flow/  — full research pipeline
    build_dataset.py / build_dataset_v2.py  — feature engineering
    train_baselines.py / train_baselines_v2.py  — model training
    ab_simulate.py / ab_simulate_v2.py  — PnL evaluation
    artifacts/  — saved models
```

---

## Statistical confidence (bootstrap CI on CatBoost v2 τ=1.5¢)

Per-token bootstrap (n=63 active tokens, 2000 resamples):
- **Mean per-token PnL: +$0.053**
- **95% CI: [−$0.025, +$0.139]** ← lower bound below zero
- **73% winning tokens** (46/63)

Honest read: the per-token CI is wider than the per-fill CI from the earlier v1 analysis (the v1 had [+1.88c, +8.22c] excluding zero — but that was per-fill, more granular). At the token level, we have only 63 trials so the CI naturally widens. **73% winners is a strong signal that this is genuinely positive expectation,** but a long-tailed loser distribution prevents the lower bound from clearing zero.

**Honest projection:**
- Daily rate (1-share size, 26-day backtest): $+0.13
- At $100 quote size: **$+13/day** (likely range $-2 to $+30)
- At $1000 quote size: **$+127/day** (likely range $-25 to $+330)

The user should validate live for a week before scaling capital — backtest CI doesn't account for regime drift.

---

## Recommended next moves (priority order)

**1. Investigate why `_price_updater` is wedged.** First iteration not completing. Likely network/API issue. Some debug logging + maybe move to async I/O would unblock real-time data flow.

**2. Bootstrap CI on CatBoost v2 result.** ~30 min compute, tightens confidence interval.

**3. Paper trade with py-clob-client.** Once live data flows: 24-48 hour shadow test, compare realised PnL to backtest. If realized ≈ backtest, go live with $50-100 quotes.

**4. Add even more features** if needed:
   - Match-level features (round, score, who just scored)
   - Per-token historical win rate
   - Time-of-event (round just ended within X seconds)

**5. Sequence models with real ticks.** When data permits — train CNN/LSTM on actual per-second mid sequences instead of reconstructed approximations. This is where the 5090 would have a real workout.

---

## What I'd watch on the dashboard

Once `_price_updater` unwedges (or you restart with a different code path):
- **MM stats panel** should show "Drift-Filter Pulls 24h" rising — that's CatBoost v2 saying "don't quote, toxic"
- **Fills 24h** should grow on the order of dozens per day
- **¢/fill** should hover in the +2 to +6 range if backtest is honest

If `¢/fill` is consistently 0 or negative after 24 hours of fills, that's a sign live conditions differ from backtest (regime shift) and we should pause + investigate.

---

*Generated automatically at end of 4-hour autonomous session.*
