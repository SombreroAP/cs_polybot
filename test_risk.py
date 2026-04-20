"""Unit tests for risk.py — run with `python3 -m unittest test_risk.py`."""
from __future__ import annotations
import os
import tempfile
import time
import unittest
from pathlib import Path

import risk
from risk import RiskConfig, RiskState, evaluate_trade, on_trade_closed, on_trade_opened


def fresh_state(balance: float = 1000.0) -> RiskState:
    return RiskState(starting_balance=balance, current_balance=balance)


def base_cfg(**kwargs) -> RiskConfig:
    defaults = dict(
        hard_stop_lock_path=os.path.join(tempfile.gettempdir(), f"HARD_STOP_test_{time.time_ns()}.lock"),
    )
    defaults.update(kwargs)
    return RiskConfig(**defaults)


class TestHardStop(unittest.TestCase):
    def test_triggers_at_50pct_drawdown(self):
        cfg = base_cfg()
        state = fresh_state(1000)
        state.current_balance = 500  # -50%
        d = evaluate_trade(0.8, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertFalse(d.allowed)
        self.assertIn("HARD_STOP", d.reason)
        self.assertTrue(Path(cfg.hard_stop_lock_path).exists())
        Path(cfg.hard_stop_lock_path).unlink(missing_ok=True)

    def test_does_not_trigger_at_49pct(self):
        cfg = base_cfg()
        state = fresh_state(1000)
        state.current_balance = 510  # -49% — juuust under the line
        d = evaluate_trade(0.8, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertTrue(d.allowed)
        Path(cfg.hard_stop_lock_path).unlink(missing_ok=True)


class TestDailyStop(unittest.TestCase):
    def test_triggers_at_minus_10pct_today(self):
        cfg = base_cfg(daily_loss_stop_pct=0.10)
        state = fresh_state(1000)
        state.current_balance = 900
        state.today_realized_pnl = -100  # exactly -10%
        # Must set last_reset_date to today so reset_daily_if_needed doesn't clear our PnL
        state.last_reset_date = time.strftime("%Y-%m-%d", time.gmtime())
        d = evaluate_trade(0.8, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertFalse(d.allowed)
        self.assertIn("DAILY_STOP", d.reason)


class TestCooldown(unittest.TestCase):
    def test_cooldown_after_3_consecutive_losses(self):
        cfg = base_cfg(cooldown_after_n_losses=3, cooldown_duration_s=1800)
        state = fresh_state(1000)
        t = time.time()
        on_trade_closed(state, -20, cfg, now_ts=t)
        on_trade_closed(state, -15, cfg, now_ts=t + 60)
        on_trade_closed(state, -10, cfg, now_ts=t + 120)
        # Now in cooldown
        d = evaluate_trade(0.85, 0.50, 0.03, 2000, "round_end", state, cfg,
                           now_ts=t + 180)
        self.assertFalse(d.allowed)
        self.assertIn("COOLDOWN", d.reason)

    def test_cooldown_lifts_after_duration(self):
        cfg = base_cfg(cooldown_after_n_losses=3, cooldown_duration_s=1800)
        state = fresh_state(1000)
        t = time.time()
        on_trade_closed(state, -20, cfg, now_ts=t)
        on_trade_closed(state, -15, cfg, now_ts=t)
        on_trade_closed(state, -10, cfg, now_ts=t)
        # Skip ahead past cooldown
        d = evaluate_trade(0.85, 0.50, 0.03, 2000, "round_end", state, cfg,
                           now_ts=t + 1801)
        self.assertTrue(d.allowed)


class TestBetSizing(unittest.TestCase):
    def test_base_size_4pct_at_confidence_70(self):
        cfg = base_cfg(per_trade_base_pct=0.04, confidence_floor=0.70,
                       confidence_strong=0.85)
        state = fresh_state(1000)
        d = evaluate_trade(0.70, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertTrue(d.allowed)
        self.assertAlmostEqual(d.bet_size_usd, 40.0, delta=0.01)

    def test_strong_size_6pct_at_high_confidence(self):
        cfg = base_cfg(per_trade_strong_pct=0.06, confidence_strong=0.85)
        state = fresh_state(1000)
        d = evaluate_trade(0.90, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertTrue(d.allowed)
        self.assertAlmostEqual(d.bet_size_usd, 60.0, delta=0.01)

    def test_never_exceeds_max_pct(self):
        cfg = base_cfg(per_trade_max_pct=0.08, per_trade_strong_pct=0.15)
        state = fresh_state(1000)
        d = evaluate_trade(0.95, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertLessEqual(d.bet_size_usd, 80.0)

    def test_training_wheels_caps_hard(self):
        cfg = base_cfg(training_wheels_pct=0.01)
        state = fresh_state(1000)
        d = evaluate_trade(0.95, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertAlmostEqual(d.bet_size_usd, 10.0, delta=0.01)


class TestTailRisk(unittest.TestCase):
    def test_cap_for_high_ask_favorite(self):
        cfg = base_cfg(tail_risk_ask_threshold=0.70, tail_risk_max_bet_usd=25.0)
        state = fresh_state(1000)
        d = evaluate_trade(0.90, 0.75, 0.03, 2000, "round_end", state, cfg)
        self.assertTrue(d.allowed)
        self.assertLessEqual(d.bet_size_usd, 25.0)

    def test_refuses_big_fav_without_map_win(self):
        cfg = base_cfg(tail_risk_require_map_win=True)
        state = fresh_state(1000)
        d = evaluate_trade(0.90, 0.90, 0.03, 2000, "kill_streak", state, cfg)
        self.assertFalse(d.allowed)

    def test_allows_big_fav_on_map_win(self):
        cfg = base_cfg(tail_risk_require_map_win=True)
        state = fresh_state(1000)
        d = evaluate_trade(0.90, 0.90, 0.03, 2000, "map_win", state, cfg)
        self.assertTrue(d.allowed)


class TestMarketFilters(unittest.TestCase):
    def test_rejects_thin_liquidity(self):
        cfg = base_cfg(min_liquidity_usd=1000)
        state = fresh_state(1000)
        d = evaluate_trade(0.85, 0.55, 0.03, 500, "round_end", state, cfg)
        self.assertFalse(d.allowed)
        self.assertIn("liquidity", d.reason)

    def test_rejects_wide_spread(self):
        cfg = base_cfg(max_spread_pct=0.10)
        state = fresh_state(1000)
        d = evaluate_trade(0.85, 0.55, 0.15, 5000, "round_end", state, cfg)
        self.assertFalse(d.allowed)
        self.assertIn("spread", d.reason)

    def test_rejects_low_confidence(self):
        cfg = base_cfg(confidence_floor=0.70)
        state = fresh_state(1000)
        d = evaluate_trade(0.65, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertFalse(d.allowed)
        self.assertIn("conf", d.reason)


class TestPositionCap(unittest.TestCase):
    def test_refuses_when_at_cap(self):
        cfg = base_cfg(max_concurrent_positions=3)
        state = fresh_state(1000)
        state.open_positions_count = 3
        d = evaluate_trade(0.85, 0.55, 0.03, 2000, "round_end", state, cfg)
        self.assertFalse(d.allowed)


class TestSerialization(unittest.TestCase):
    def test_roundtrip(self):
        state = fresh_state(1000)
        state.current_balance = 1123.45
        state.today_realized_pnl = -45
        state.consecutive_losses = 2
        path = os.path.join(tempfile.gettempdir(), f"risk_state_{time.time_ns()}.json")
        risk.save_state(state, path)
        loaded = risk.load_state(path)
        self.assertEqual(loaded.current_balance, 1123.45)
        self.assertEqual(loaded.today_realized_pnl, -45)
        self.assertEqual(loaded.consecutive_losses, 2)
        Path(path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main(verbosity=2)
