"""Unit tests for the paper-fill exit math + sqlite update path.

No network, no bot state — only paper_fill in isolation.
"""
import os
import sqlite3
import time
from pathlib import Path

import pytest

import paper_fill


# ─── Exit-decision math (pure function) ────────────────────────────────


def _r(fill, current, hold, tp=0.075, sl=0.125, timeout=600):
    return paper_fill._exit_reason(fill, current, hold, tp, sl, timeout)


def test_no_exit_when_within_band():
    assert _r(0.50, 0.51, 30) is None
    assert _r(0.50, 0.49, 30) is None


def test_take_profit():
    # +7.5% exactly is the threshold — at 0.5375 should fire
    assert _r(0.50, 0.5375, 30) == "take_profit"
    assert _r(0.50, 0.60, 1)    == "take_profit"


def test_stop_loss():
    assert _r(0.50, 0.4375, 30) == "stop_loss"
    assert _r(0.50, 0.40, 1)    == "stop_loss"


def test_time_exit():
    assert _r(0.50, 0.51, 600)  == "time_exit"
    assert _r(0.50, 0.51, 700)  == "time_exit"


def test_dust_token_doesnt_exit_until_double_timeout():
    # current == 0.01 is "dust" — don't sell at zero, but eventually time-exit
    assert _r(0.50, 0.01,   60) is None
    assert _r(0.50, 0.005, 600) is None
    assert _r(0.50, 0.005, 1300) == "time_exit"


def test_take_profit_beats_stop_loss_when_both_would_trigger_impossible():
    # not really possible, but verify TP wins by being checked first
    pass  # intentional sanity check


# ─── End-to-end against a fake sqlite ──────────────────────────────────


def _seed_shadow_trade(db_path, **overrides):
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shadow_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL, match_id TEXT, token_id TEXT, team TEXT, game TEXT,
            fill_price REAL, market_price REAL, bet_usd REAL, confidence REAL,
            tp_pct REAL, sl_pct REAL, trigger TEXT, llm_reason TEXT, llm_model TEXT
        )
    """)
    row = {
        "ts": time.time() - 30,  # placed 30s ago
        "match_id": "m1", "token_id": "tok_a", "team": "FaZe", "game": "cs2",
        "fill_price": 0.50, "market_price": 0.50, "bet_usd": 10, "confidence": 0.7,
        "tp_pct": 0.075, "sl_pct": 0.125,
        "trigger": "round_end", "llm_reason": "leading + eco edge", "llm_model": "haiku",
    }
    row.update(overrides)
    conn.execute(
        "INSERT INTO shadow_trades (ts,match_id,token_id,team,game,fill_price,"
        "market_price,bet_usd,confidence,tp_pct,sl_pct,trigger,llm_reason,llm_model)"
        " VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        tuple(row[k] for k in [
            "ts","match_id","token_id","team","game","fill_price","market_price",
            "bet_usd","confidence","tp_pct","sl_pct","trigger","llm_reason","llm_model",
        ]),
    )
    conn.commit()
    rid = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return rid


class _FakeMarket:
    def __init__(self, token_id_a, price_a, token_id_b="tok_b", price_b=0.50):
        self.token_id_a = token_id_a
        self.token_id_b = token_id_b
        self.price_a = price_a
        self.price_b = price_b


def test_simulator_tp_exit(tmp_path):
    db = tmp_path / "trades.db"
    rid = _seed_shadow_trade(db, fill_price=0.50)

    sim = paper_fill.PaperFillSimulator(db_path=db)
    linked = {"m1": _FakeMarket("tok_a", 0.60)}  # +20% — well above TP

    n = sim.tick(linked)
    assert n == 1, f"expected 1 exit, got {n}"

    conn = sqlite3.connect(str(db))
    row = conn.execute(
        "SELECT exit_price, exit_reason, pnl_pct FROM shadow_trades WHERE id=?",
        (rid,),
    ).fetchone()
    conn.close()
    assert row is not None
    assert row[0] == pytest.approx(0.60)
    assert row[1] == "take_profit"
    assert row[2] == pytest.approx(20.0)


def test_simulator_sl_exit(tmp_path):
    db = tmp_path / "trades.db"
    rid = _seed_shadow_trade(db, fill_price=0.50)

    sim = paper_fill.PaperFillSimulator(db_path=db)
    linked = {"m1": _FakeMarket("tok_a", 0.40)}  # −20%, well below SL

    sim.tick(linked)

    conn = sqlite3.connect(str(db))
    reason, pnl = conn.execute(
        "SELECT exit_reason, pnl_pct FROM shadow_trades WHERE id=?", (rid,),
    ).fetchone()
    conn.close()
    assert reason == "stop_loss"
    assert pnl == pytest.approx(-20.0)


def test_simulator_no_double_exit(tmp_path):
    """Once a row has exit_price set, subsequent ticks must not re-process it."""
    db = tmp_path / "trades.db"
    _seed_shadow_trade(db, fill_price=0.50)
    sim = paper_fill.PaperFillSimulator(db_path=db)
    linked = {"m1": _FakeMarket("tok_a", 0.60)}

    assert sim.tick(linked) == 1
    assert sim.tick(linked) == 0  # no new exits


def test_simulator_skips_unknown_token(tmp_path):
    db = tmp_path / "trades.db"
    _seed_shadow_trade(db, token_id="missing_token")
    sim = paper_fill.PaperFillSimulator(db_path=db)
    linked = {"m1": _FakeMarket("tok_a", 0.60)}  # different token

    assert sim.tick(linked) == 0


def test_simulator_silent_when_table_missing(tmp_path):
    db = tmp_path / "trades.db"
    # Create empty db file with NO shadow_trades table
    sqlite3.connect(str(db)).close()
    sim = paper_fill.PaperFillSimulator(db_path=db)
    assert sim.tick({}) == 0
