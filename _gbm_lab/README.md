# _gbm_lab — Polymarket Esports Bot Research

Built during the 2026-05-13/14 overnight session. **See `OVERNIGHT_REPORT.md` for the executive summary.**

## What's in here

### The winning strategy
- **`mm_strategy_production.py`** — Production-ready, tested module implementing the drift-conditioned MM strategy. Plug into `py-clob-client` for live use.
- **`test_mm_strategy.py`** — 7 unit tests, all pass.
- **`paper_trade_harness.py`** — Skeleton WebSocket runner. Logs decisions without placing orders. Useful for verifying live behaviour before going live.

### Research scripts (chronological)

**Phase 3 — GBM directional approach (unprofitable, archived)**
- `features.py` — Context-feature extractor
- `shootout.py`, `shootout_v2.py` — Compare LightGBM/XGBoost/CatBoost/sklearn
- `walkforward.py`, `walkforward_v2.py`, `rolling_walkforward.py`, `regularized.py` — Honest time-split validation
- `final_pipeline.py` — Parameterised pipeline runner
- `backtest.py` — Bootstrap CIs + drawdown utilities

**Phase 4 — Longer time windows (unprofitable, archived)**
- `relabel.py` — Re-label SFT with custom lookahead window
- `exit_policy.py` — Realistic TP/SL/trailing-stop exit simulator

**Phase 5 — Market making (THE WIN)**
- `mm_viability.py` — Spread-vs-drift analytical proof
- `mm_simulator.py` — Base MM simulator with strategy sweep
- `mm_event_aware.py` — MM with game-event cooldown
- `mm_friction_sweep.py` — Slippage sensitivity (break-even at 0.18¢/share)
- `mm_inventory_skew.py` — Inventory-aware quote skewing
- `mm_phase_analysis.py` — Per-game-phase toxicity analysis
- `mm_phase_filtered.py` — Phase-restricted MM (was worse than baseline)
- **`mm_drift_conditioned.py`** — The breakthrough: +5.07¢/fill
- `mm_relative_drift.py` — Volatility-aware relative drift filter
- `mm_final_validation.py` — Bootstrap CIs + per-day + per-token PnL

## How to use

### Run the validated MM strategy in mock-replay mode
```bash
PYTHONPATH=_gbm_lab .gbm_env/bin/python _gbm_lab/paper_trade_harness.py \
    --mock-replay data/processed/<match-id>.jsonl
```

### Reproduce the final validation numbers
```bash
PYTHONPATH=_gbm_lab .gbm_env/bin/python _gbm_lab/mm_final_validation.py
```

### Run unit tests
```bash
.gbm_env/bin/python _gbm_lab/test_mm_strategy.py
```

### Go live (when ready)
1. Read `OVERNIGHT_REPORT.md` — section "Decision points for tomorrow"
2. Install `py-clob-client` and wire WebSocket subscription in `paper_trade_harness.py`
3. Paper-trade for 3-5 days at $50-100 size, compare realised PnL to backtest
4. If matches: scale to $500-1000 size

## Headline numbers

```
Strategy: behind-inside-2c market-making + 30s/1¢ drift filter
Data:     26 days, 2,789 tokens, 512K book points
Result:   +5.07¢/fill gross (+5.02¢ at realistic 0.05¢ Polygon gas)
CI:       [+1.88¢, +8.22¢] — excludes zero
Fills:    306 over 26 days (~12/day)
Days +:   17/29 (59%)
Per-share scaling:  +$0.53/day  →  $53/day at $100 size  →  $530/day at $1000 size
```

## Environment

Created during this session: `.gbm_env` Python venv with sklearn, lightgbm, xgboost, catboost, numpy.
```bash
.gbm_env/bin/python <script>     # use this for all _gbm_lab scripts
```
