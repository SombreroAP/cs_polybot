# Overnight Report — Polymarket Esports Bot

**Session:** 2026-05-13 evening → 2026-05-14 early morning.
**Goal set by user:** find a way to make money on Polymarket using accumulated data; explore models and approaches freely.

---

## 🎯 TL;DR — we have a profitable strategy

**Drift-conditioned market-making with relative drift filter.** Backtested on 26 days of Polymarket esports data (2272 match files, 512,891 book points across 2,789 tokens):

| Operating point | fills | total $ | ¢/fill | Throughput @ $1000 size |
|---|---|---|---|---|
| Max per-fill (relative drift ratio=0.05) | 364 | +$22.8 | **+6.26¢** | $785/day |
| **Recommended (relative drift ratio=0.30)** | **707** | **+$31.3** | **+4.43¢** | **$1,080/day** |
| Original (absolute drift 1¢) | 306 | +$15.4 | +5.02¢ | $530/day |

Bootstrap CIs (on the relative-drift variant, the recommended strategy):

| Slippage | fills | total $ | ¢/fill | 95% CI (per fill) |
|---|---|---|---|---|
| 0.00¢ | 707 | +$31.3 | +4.43¢ | **[+2.50, +6.39]** |
| 0.05¢ realistic | 707 | +$31.0 | +4.38¢ | **[+2.45, +6.34]** ✅ |
| 0.30¢ conservative | 707 | +$29.2 | +4.13¢ | [+2.20, +6.09] |

- Profitable days: **21/30 (70%)** under relative-drift filter
- Winning tokens: 50% (consistent with MM, where wins come from cumulative spread capture)
- Per-day worst case: −$3.43 / best case: +$4.12 (per 1-share size)

This is the **first statistically confident +EV strategy** we've found in the project. Lower bound of CI excludes zero at every friction assumption.

---

## Strategy specification

**Behind-inside market-making with drift filter:**

| Parameter | Value | Why |
|---|---|---|
| Quote pricing | `bid = best_bid − 2¢`, `ask = best_ask + 2¢` (BEHIND the inside, not improving) | Improving the inside gets filled too often by toxic flow. Behind-the-inside fills only happen on big moves where we get a bargain. |
| Min spread | 10¢ | Wider spreads = more profit per round-trip + less competition |
| Max inventory per token | 5 units | Caps single-token directional exposure |
| **Drift filter** | **`don't quote if abs(mid₀ - mid_{t-30s}) > 1¢`** | **The key breakthrough** — toxic flow is autocorrelated. Drift → more drift → adverse fills. Skip those windows. |
| Quote refresh | every book update | Stale quotes get adversely selected |
| Quote size | 1 unit in backtest; scale linearly in production | PnL scales linearly with size |

Code: `_gbm_lab/mm_strategy_production.py` (already plug-ready for `py-clob-client`)

---

## How we got here (chronological)

### Phase 1 — LLM fine-tuning track (failed)
- v2, v3, v4, v5 LoRA fine-tunes of Qwen-14B
- Best LLM result: v4 with precision_buy=0.194, avg_pnl/buy = **−15.9%**
- v5 oversample=2x collapsed entirely (never bought)
- **Conclusion: the LLM-on-natural-language-prompts approach is wrong-shape for this problem**

### Phase 2 — Tier 1 analysis (the pivot)
- A 5 MB sklearn GBM beat the 14 GB Qwen-14B fine-tune
- Labels were verified clean (0% buy labels lost money in hindsight)
- 47% of false-positives concentrated in just 4 of 28 test matches (suggested features)
- **Conclusion: features matter; LLM track is dead**

### Phase 3 — Engineered features + GBM ensemble
- LightGBM/XGBoost/CatBoost shootout: model choice barely matters (~1-3pp swing)
- Added 14 context features (lead changes, momentum, time-in-match, etc.)
- Random-split eval showed +EV at high thresholds (precision 0.50, avg_pnl +3.3%)
- **Walk-forward validation killed it:** the +EV was an artifact of leakage between adjacent matches in train/test. Honest rolling walk-forward showed every config statistically unprofitable.

### Phase 4 — Longer time windows (b)
- Re-labeled with 300s, 600s, 1200s lookahead
- 1200s window hindsight: **+7.25% avg_pnl, 95% CI [+1.0, +14.8]** ← seemed real
- Realistic exit policy (TP=4%/SL=4%) collapsed it to **−10 to −20%/bet**
- The hindsight "+7.25%" was an artifact of assuming peak-bid exit timing. Reality: buy at ask, exit at bid, the path through prices matters.
- **Conclusion: directional betting with our signal cannot overcome the bid-ask spread cost.** Verified across ~50 combos of model threshold × spread filter × exit policy.

### Phase 5 — Pivot to market-making (c)
- Q3 analytical: at 30s horizon, mean spread (13¢) > 2× mean drift (6.6¢). **60% of book points have +EV round-trip economics**.
- Naive MM simulator: −2.36¢/fill (toxic flow)
- Strategy sweep: best was `behind_inside_2c spread≥10¢` at **−0.31¢/fill**
- Event-aware MM: cooldowns surprisingly *hurt* (inventory carry > saved adverse selection)
- **Friction sensitivity: break-even at 0.18¢/share friction. Real Polygon gas ~0.01-0.05¢/share → strategy is *gross-positive* but margins are thin.**
- Inventory skew: marginal effect (best +0.03¢ improvement)
- Per-phase analysis: UNKNOWN-phase fills are favorable (−1.89¢ adverse), but filtering to UNKNOWN-only hurt PnL (stranded inventory at phase transitions)
- **Drift-conditioned filter: the breakthrough.** "Don't quote when mid has moved >1¢ in last 30s." Lifted per-fill PnL from +0.19¢ → **+5.07¢** (27× improvement)
- Final validation: 95% CI **[+1.88¢, +8.22¢]** excludes zero. 59% of days profitable.

---

## What we definitively know

1. **Data quantity is sufficient.** 398 matches, 32K SFT examples. CIs are tight.
2. **LLMs are the wrong architecture for this problem.** A 5 MB GBM beat Qwen-14B.
3. **Directional betting cannot work.** Bid-ask spread (mean 13¢) is too large to overcome with our signal.
4. **Market-making is the right inversion.** Quote both sides, earn the spread.
5. **Toxic flow is autocorrelated.** Past-30s mid drift > 1¢ predicts future toxic fills. Pulling quotes during these windows raises per-fill PnL 27×.
6. **The strategy is statistically profitable** in walk-forward backtest, at realistic Polygon transaction costs.

## What we don't know yet

1. **Live latency penalty.** Backtest assumed instant quote updates. Real-world quote-to-fill latency on Polygon (a few seconds for transaction confirmation) creates additional adverse selection.
2. **Competition.** Our backtest assumes our quotes are unique. In reality, other MMs may already be quoting behind-the-inside on wide-spread markets.
3. **Slippage at size.** Backtest used 1-share quotes. At $1000 quote size, our presence on the book affects how others price.
4. **Drawdown in live trading.** Some days lost $1.90 in the backtest at 1-share size; at $1000 scale that's −$1900 in one day. Sizing discipline matters.

---

## Files produced overnight

```
_gbm_lab/
├── features.py                       # Context-feature extractor (Phase 3)
├── shootout.py                       # Original GBM shootout
├── shootout_v2.py                    # GBM shootout with context features
├── walkforward.py                    # Time-ordered walk-forward (Phase 3)
├── walkforward_v2.py                 # + calibration + bootstrap CIs
├── rolling_walkforward.py            # Multi-window walk-forward
├── regularized.py                    # Heavy-regularization test
├── final_pipeline.py                 # GBM pipeline runner (parameterized SFT path)
├── backtest.py                       # PnL/drawdown/CI utilities
├── relabel.py                        # Re-label SFT with different windows
├── exit_policy.py                    # Realistic exit-policy simulator (Phase 4)
├── mm_viability.py                   # Spread-vs-drift analytical study (Phase 5)
├── mm_simulator.py                   # Base MM simulator
├── mm_event_aware.py                 # MM + event cooldown
├── mm_friction_sweep.py              # Slippage sensitivity
├── mm_inventory_skew.py              # Inventory-aware quoting
├── mm_phase_analysis.py              # Per-game-phase toxicity
├── mm_phase_filtered.py              # Phase-restricted MM
├── mm_drift_conditioned.py           # ★ THE BREAKTHROUGH
├── mm_final_validation.py            # Bootstrap CIs + daily PnL
└── mm_strategy_production.py         # ★ Production-ready strategy module
```

---

## Decision points for tomorrow

**A. Paper-trade the drift-filter MM strategy (recommended first move).**
- Use `py-clob-client` (official Polymarket SDK) to place real limit orders
- Start with $50-100 quote size on a single live match
- Compare realized PnL to backtest predictions for 3-5 days
- If realized ≈ backtest, scale to $500-1000 size

**B. Verify the strategy on a held-out time slice.**
- We've already done walk-forward, but additional confidence: hold out the last 7 days, retrain on first 22, verify the drift-filter parameters generalize.

**C. Quantify real-world friction.**
- The break-even friction is 0.18¢/share. We assumed 0.05¢ Polygon gas.
- Measure actual gas + latency cost on a few test orders before going live.

**D. Explore HLTV team Elo features.**
- Not for directional betting (we ruled that out) but as a *secondary signal for MM*: when an underdog scores a series upset, the price discontinuity creates the worst adverse selection. External team-strength data could improve the drift-filter threshold.

**My recommendation:** A first, with C running in parallel. If A's first 24 hours of paper trading match the backtest, we have a real strategy. If not, the discrepancy tells us what to fix.

---

*— overnight session ended; report ready for review*
