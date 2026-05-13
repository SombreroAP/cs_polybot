"""
Realistic backtest with:
  - position sizing (flat % of bankroll, or fractional Kelly)
  - extra slippage buffer (since hindsight_pnl is already net of spread, we
    just add a tiny gas/slippage buffer)
  - drawdown tracking
  - per-day P&L curve
  - threshold sweep with bootstrap CIs on avg_pnl

Inputs: a list of dicts with {decision_ts, label, pred_proba, pnl}
"""
import json, random, math
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np


SLIPPAGE_BUFFER = 0.003   # 0.3% per bet to cover gas + small slippage
TRADING_FEE = 0.0          # Polymarket V3 outcome markets


def realized_pnl(hindsight_pnl_pct: float) -> float:
    """Take hindsight_pnl% (already net of spread), subtract our extra buffers."""
    gross = hindsight_pnl_pct / 100.0
    return gross - SLIPPAGE_BUFFER - TRADING_FEE


def threshold_report(rows: list[dict], tau: float):
    """Apply threshold τ. Return dict with metrics + per-bet pnl list (ordered)."""
    rows_sorted = sorted(rows, key=lambda r: r["decision_ts"])
    bets = []
    for r in rows_sorted:
        if r["pred_proba"] >= tau:
            bets.append((r["decision_ts"], r["label"], realized_pnl(r["pnl"])))
    return bets


def equity_curve(bets, start_bankroll=10000.0, frac_per_bet=0.01):
    """Flat-fractional sizing: bet `frac_per_bet` of current bankroll each time.
    Returns list of (ts, bankroll) ordered by ts."""
    bankroll = start_bankroll
    curve = [(0, bankroll)]
    for ts, label, pnl in bets:
        stake = bankroll * frac_per_bet
        bankroll += stake * pnl
        curve.append((ts, bankroll))
    return curve


def drawdown_max(curve):
    """Compute max drawdown of an equity curve."""
    peak = -math.inf
    max_dd = 0.0
    for _, b in curve:
        peak = max(peak, b)
        if peak > 0:
            dd = (peak - b) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def daily_pnl(bets):
    """Aggregate raw pnl by UTC day. Returns dict day->sum_pnl."""
    daily = defaultdict(float)
    for ts, _, pnl in bets:
        day = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        daily[day] += pnl
    return dict(sorted(daily.items()))


def bootstrap_ci(values, n_boot=2000, alpha=0.05, rng=None):
    if not values:
        return None, None, None
    rng = rng or random.Random(42)
    n = len(values)
    mean = sum(values) / n
    boots = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boots.append(sum(sample) / n)
    boots.sort()
    lo = boots[int(n_boot * (alpha / 2))]
    hi = boots[int(n_boot * (1 - alpha / 2))]
    return mean, lo, hi


def report(rows, taus=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9), label=""):
    print(f"\n{'='*78}\nBACKTEST {label}")
    print(f"slippage_buffer={SLIPPAGE_BUFFER*100:.2f}%  trading_fee={TRADING_FEE*100:.2f}%")
    print(f"{'='*78}")
    print(f"{'τ':>5} {'bets':>5} {'prec':>6} {'avg_pnl':>9} "
          f"{'CI_lo':>8} {'CI_hi':>8} {'final_$':>9} {'max_DD':>7}")
    for tau in taus:
        bets = threshold_report(rows, tau)
        if not bets:
            print(f"{tau:>5} {0:>5}      -        -        -        -        -       -")
            continue
        pnls = [b[2] for b in bets]
        n = len(pnls)
        wins = sum(1 for p in pnls if p > 0)
        prec = wins / n
        mean, lo, hi = bootstrap_ci(pnls)
        curve = equity_curve(bets)
        final = curve[-1][1]
        dd = drawdown_max(curve)
        print(f"{tau:>5} {n:>5} {prec:>.3f} {mean*100:>+7.2f}% "
              f"{lo*100:>+7.2f}% {hi*100:>+7.2f}% ${final:>7.0f} {dd*100:>5.1f}%")
    return rows
