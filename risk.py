"""
Risk engine — code-owned (NOT prompt-owned) bankroll protection.

All bet sizing, stop rules, cooldowns, daily/hard-stops live here so they're
deterministic, unit-testable, and not at the mercy of a prompt change.

The LLM decides {action, confidence, reason}. Python owns EVERYTHING ELSE:

    - bet_size         (pct of bankroll × confidence)
    - tp_pct / sl_pct  (fixed per market type)
    - position_cap     (no more than N concurrent)
    - daily_loss_stop  (halt trading for the day at -X%)
    - hard_stop        (permanent kill at -50% bankroll — writes lock file)
    - cooldown         (pause after N consecutive losses)
    - tail_risk_guard  (cap bet on high-ask favorites)

Read from .env; unit-tested against synthetic trade sequences.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Config — loaded from env at startup, frozen for the session
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class RiskConfig:
    per_trade_base_pct:   float = 0.04     # 4% of balance for standard trade
    per_trade_strong_pct: float = 0.06     # 6% for high-confidence trade
    per_trade_max_pct:    float = 0.08     # hard cap, never exceed
    per_trade_min_usd:    float = 5.0      # below this, skip the trade
    training_wheels_pct:  float = 0.0      # override max to this during week 1
                                            # (0 = disabled)

    confidence_floor:     float = 0.70     # skip any LLM call below this
    confidence_strong:    float = 0.85     # use per_trade_strong_pct above this

    tp_pct:               float = 0.10     # take profit at +10%
    sl_pct:               float = 0.15     # stop loss at -15%
    position_timeout_s:   float = 600.0    # force-exit after 10 minutes

    max_concurrent_positions: int = 3
    daily_loss_stop_pct:  float = 0.10     # stop trading at -10% bankroll for the day
    hard_stop_pct:        float = 0.50     # permanent kill at -50% bankroll

    # Tail-risk cap — big favorites can collapse spectacularly
    tail_risk_ask_threshold:     float = 0.70    # ask > 0.70 = big favorite
    tail_risk_max_bet_usd:       float = 25.0    # cap position size for these
    tail_risk_require_map_win:   bool = True     # require map_win trigger above 0.85

    # Cooldown
    cooldown_after_n_losses: int = 3
    cooldown_duration_s:     float = 1800   # 30 minutes

    # Market floor — don't touch illiquid books
    min_liquidity_usd:    float = 1000.0
    max_spread_pct:       float = 0.10     # skip if spread > 10%

    # Lock file for hard stop
    hard_stop_lock_path:  str = "HARD_STOP.lock"


def load_config_from_env() -> RiskConfig:
    """Load risk config from environment variables (.env)."""
    def _f(name: str, default: float) -> float:
        v = os.environ.get(name)
        try:
            return float(v) if v is not None else default
        except ValueError:
            return default

    def _i(name: str, default: int) -> int:
        v = os.environ.get(name)
        try:
            return int(v) if v is not None else default
        except ValueError:
            return default

    return RiskConfig(
        per_trade_base_pct   = _f("PER_TRADE_BASE_PCT",   0.04),
        per_trade_strong_pct = _f("PER_TRADE_STRONG_PCT", 0.06),
        per_trade_max_pct    = _f("PER_TRADE_MAX_PCT",    0.08),
        training_wheels_pct  = _f("TRAINING_WHEELS_MAX_PCT", 0.0),
        confidence_floor     = _f("CONFIDENCE_FLOOR", 0.70),
        tp_pct               = _f("TAKE_PROFIT_PCT", 0.10),
        sl_pct               = _f("STOP_LOSS_PCT", 0.15),
        position_timeout_s   = _f("EXIT_TIMEOUT_SECONDS", 600),
        max_concurrent_positions = _i("MAX_CONCURRENT_POSITIONS", 3),
        daily_loss_stop_pct  = _f("DAILY_LOSS_STOP_PCT", 0.10),
        hard_stop_pct        = _f("HARD_STOP_PCT", 0.50),
        cooldown_after_n_losses = _i("COOLDOWN_AFTER_N_LOSSES", 3),
        min_liquidity_usd    = _f("MIN_LIQUIDITY_USD", 1000.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Session state — mutable, tracks what happened today
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskState:
    starting_balance:      float
    current_balance:       float
    open_positions_count:  int = 0
    today_realized_pnl:    float = 0.0
    today_start_ts:        float = 0.0
    consecutive_losses:    int = 0
    cooldown_until_ts:     float = 0.0
    last_reset_date:       str = ""
    trades_today:          int = 0

    def reset_daily_if_needed(self, now_ts: float) -> None:
        """Reset daily counters at 00:00 UTC."""
        today = time.strftime("%Y-%m-%d", time.gmtime(now_ts))
        if today != self.last_reset_date:
            self.today_realized_pnl = 0.0
            self.trades_today = 0
            self.today_start_ts = now_ts
            self.last_reset_date = today


# ─────────────────────────────────────────────────────────────────────────────
# Decision — what the engine returns
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RiskDecision:
    allowed:          bool
    bet_size_usd:     float = 0.0
    tp_pct:           float = 0.0
    sl_pct:           float = 0.0
    timeout_s:        float = 0.0
    reason:           str = ""
    warnings:         list[str] = field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────────────
# Core — decide whether / how much to bet on a proposed trade
# ─────────────────────────────────────────────────────────────────────────────

def check_hard_stop(state: RiskState, cfg: RiskConfig) -> Optional[str]:
    """Return a reason string if hard-stop triggered, else None."""
    drawdown = (state.starting_balance - state.current_balance) / state.starting_balance
    if drawdown >= cfg.hard_stop_pct:
        # Write the lock file so bot refuses to restart
        try:
            Path(cfg.hard_stop_lock_path).write_text(
                f"HARD_STOP triggered at {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}\n"
                f"bankroll: ${state.current_balance:.2f} "
                f"(-{drawdown*100:.1f}% from ${state.starting_balance:.2f})\n"
                f"delete this file manually to re-enable trading."
            )
        except Exception:
            pass
        return f"HARD_STOP: bankroll at -{drawdown*100:.1f}% (threshold {cfg.hard_stop_pct*100}%)"

    # Also check if the lock file exists from a prior run
    if Path(cfg.hard_stop_lock_path).exists():
        return f"HARD_STOP: lock file {cfg.hard_stop_lock_path} exists — delete to re-enable"
    return None


def check_daily_stop(state: RiskState, cfg: RiskConfig) -> Optional[str]:
    """Return reason if daily-loss stop triggered."""
    if state.starting_balance <= 0:
        return None
    daily_loss_pct = -state.today_realized_pnl / state.starting_balance
    if daily_loss_pct >= cfg.daily_loss_stop_pct:
        return (f"DAILY_STOP: today's loss ${state.today_realized_pnl:.2f} "
                f"({daily_loss_pct*100:.1f}%) exceeds {cfg.daily_loss_stop_pct*100}% limit")
    return None


def check_cooldown(state: RiskState, cfg: RiskConfig, now_ts: float) -> Optional[str]:
    """Return reason if cooldown still active."""
    if now_ts < state.cooldown_until_ts:
        remaining_min = (state.cooldown_until_ts - now_ts) / 60
        return f"COOLDOWN: {remaining_min:.0f}m remaining after {state.consecutive_losses} losses"
    return None


def compute_bet_size(confidence: float, balance: float, ask: float,
                     trigger_type: str, cfg: RiskConfig) -> tuple[float, list[str]]:
    """Determine the USD bet size. Returns (size, warnings)."""
    warnings: list[str] = []

    # Pick the base pct from confidence
    if confidence >= cfg.confidence_strong:
        pct = cfg.per_trade_strong_pct
    else:
        pct = cfg.per_trade_base_pct

    # Apply max cap
    pct = min(pct, cfg.per_trade_max_pct)

    # Training-wheels override
    if cfg.training_wheels_pct > 0 and cfg.training_wheels_pct < pct:
        pct = cfg.training_wheels_pct
        warnings.append(f"training_wheels active: capped at {pct*100:.1f}%")

    size = balance * pct

    # Tail-risk guard
    if ask > cfg.tail_risk_ask_threshold:
        if ask > 0.85 and cfg.tail_risk_require_map_win and trigger_type != "map_win":
            # Refuse this trade
            return (0.0, warnings + ["tail_risk: ask > 85¢ without map_win trigger"])
        if size > cfg.tail_risk_max_bet_usd:
            size = cfg.tail_risk_max_bet_usd
            warnings.append(f"tail_risk: capped at ${size:.0f} (ask {ask*100:.0f}¢)")

    # Min check
    if size < cfg.per_trade_min_usd:
        return (0.0, warnings + [f"size ${size:.2f} below min ${cfg.per_trade_min_usd}"])

    # Final absolute cap: never exceed current balance × max_pct
    hard_max = balance * cfg.per_trade_max_pct
    if size > hard_max:
        size = hard_max
        warnings.append(f"capped at hard max {cfg.per_trade_max_pct*100:.0f}% of balance")

    return (round(size, 2), warnings)


def evaluate_trade(
    confidence: float,
    ask: float,
    spread_pct: float,
    liquidity_usd: float,
    trigger_type: str,
    state: RiskState,
    cfg: RiskConfig,
    now_ts: Optional[float] = None,
) -> RiskDecision:
    """Decide whether to enter a proposed trade and with what size.

    Args:
        confidence:    LLM's confidence 0.0-1.0
        ask:           current ask price (0.0-1.0)
        spread_pct:    (ask - bid) / ask
        liquidity_usd: orderbook depth in USD
        trigger_type:  "round_end" / "map_win" / "kill_streak" / etc
        state:         session risk state (mutated)
        cfg:           risk config (frozen)
        now_ts:        defaults to time.time()
    """
    if now_ts is None:
        now_ts = time.time()
    state.reset_daily_if_needed(now_ts)

    warnings: list[str] = []

    # Stops in order of severity
    hs = check_hard_stop(state, cfg)
    if hs:
        return RiskDecision(allowed=False, reason=hs)

    ds = check_daily_stop(state, cfg)
    if ds:
        return RiskDecision(allowed=False, reason=ds)

    cd = check_cooldown(state, cfg, now_ts)
    if cd:
        return RiskDecision(allowed=False, reason=cd)

    # Market-level filters
    if liquidity_usd < cfg.min_liquidity_usd:
        return RiskDecision(allowed=False,
                            reason=f"liquidity ${liquidity_usd:.0f} < min ${cfg.min_liquidity_usd:.0f}")
    if spread_pct > cfg.max_spread_pct:
        return RiskDecision(allowed=False,
                            reason=f"spread {spread_pct*100:.1f}% > max {cfg.max_spread_pct*100:.0f}%")

    # Confidence floor
    if confidence < cfg.confidence_floor:
        return RiskDecision(allowed=False,
                            reason=f"conf {confidence:.2f} < floor {cfg.confidence_floor}")

    # Position cap
    if state.open_positions_count >= cfg.max_concurrent_positions:
        return RiskDecision(allowed=False,
                            reason=f"already at {state.open_positions_count}/{cfg.max_concurrent_positions} positions")

    # Size the bet
    size, size_warnings = compute_bet_size(confidence, state.current_balance, ask,
                                           trigger_type, cfg)
    warnings.extend(size_warnings)
    if size <= 0:
        return RiskDecision(allowed=False,
                            reason=size_warnings[-1] if size_warnings else "bet size zero")

    # Ensure bet doesn't push us below a sensible minimum balance
    if state.current_balance - size < 20.0:
        return RiskDecision(allowed=False,
                            reason=f"bet ${size:.0f} would leave < $20 bankroll")

    return RiskDecision(
        allowed=True,
        bet_size_usd=size,
        tp_pct=cfg.tp_pct,
        sl_pct=cfg.sl_pct,
        timeout_s=cfg.position_timeout_s,
        reason=f"OK — {trigger_type} @ ask={ask*100:.0f}¢ conf={confidence:.2f} size=${size:.0f}",
        warnings=warnings,
    )


def on_trade_closed(state: RiskState, pnl_usd: float,
                    cfg: RiskConfig, now_ts: Optional[float] = None) -> None:
    """Update risk state after a trade closes. Enforces cooldown."""
    if now_ts is None:
        now_ts = time.time()
    state.reset_daily_if_needed(now_ts)
    state.current_balance += pnl_usd
    state.today_realized_pnl += pnl_usd
    state.trades_today += 1
    state.open_positions_count = max(0, state.open_positions_count - 1)
    if pnl_usd < 0:
        state.consecutive_losses += 1
        if state.consecutive_losses >= cfg.cooldown_after_n_losses:
            state.cooldown_until_ts = now_ts + cfg.cooldown_duration_s
            state.consecutive_losses = 0  # reset; cooldown fires once
    else:
        state.consecutive_losses = 0


def on_trade_opened(state: RiskState) -> None:
    state.open_positions_count += 1


# ─────────────────────────────────────────────────────────────────────────────
# Serialization — save/restore risk state across restarts
# ─────────────────────────────────────────────────────────────────────────────

def save_state(state: RiskState, path: str = "data/risk_state.json") -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump({
            "starting_balance":     state.starting_balance,
            "current_balance":      state.current_balance,
            "open_positions_count": state.open_positions_count,
            "today_realized_pnl":   state.today_realized_pnl,
            "today_start_ts":       state.today_start_ts,
            "consecutive_losses":   state.consecutive_losses,
            "cooldown_until_ts":    state.cooldown_until_ts,
            "last_reset_date":      state.last_reset_date,
            "trades_today":         state.trades_today,
        }, f, indent=2)


def load_state(path: str = "data/risk_state.json",
               default_starting: float = 1000.0) -> RiskState:
    p = Path(path)
    if not p.exists():
        return RiskState(starting_balance=default_starting, current_balance=default_starting)
    with open(p) as f:
        d = json.load(f)
    return RiskState(**d)
