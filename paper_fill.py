"""
Paper-fill simulator — turns shadow_trades rows into realized P&L by simulating
TP/SL/timeout exits against live bid/ask ticks.

Runs in a daemon thread alongside the live position monitor. Reads shadow rows
with no exit yet, looks up the current price for that token via the latency
analyzer's `_match_to_market` dict, and on exit-condition match, persists:
    exit_price, exit_reason, exit_ts, hold_seconds, pnl_pct

Telegram /pnl and /positions now reflect realized shadow P&L automatically
(they already read from shadow_trades).

Why a separate module: keeps bot.py uncluttered, allows unit-testing the
exit-decision math against synthetic ticks.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import config

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).resolve().parent / "data" / "trades.db"
EXIT_COLUMNS = {
    "exit_price": "REAL",
    "exit_reason": "TEXT",   # 'take_profit' | 'stop_loss' | 'time_exit'
    "exit_ts": "REAL",
    "hold_seconds": "REAL",
    "pnl_pct": "REAL",
}


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Add exit columns to shadow_trades if missing. Idempotent."""
    cur = conn.execute("PRAGMA table_info(shadow_trades)")
    have = {row[1] for row in cur.fetchall()}
    if not have:
        # Table doesn't exist yet (no shadow trades logged) — bot.py creates it
        # on first BUY decision. Nothing to do until then.
        return
    for col, kind in EXIT_COLUMNS.items():
        if col not in have:
            conn.execute(f"ALTER TABLE shadow_trades ADD COLUMN {col} {kind}")
            logger.info(f"[PAPER-FILL] schema: added column {col} {kind}")
    conn.commit()


def _current_price_for_token(linked_markets: dict, token_id: str) -> Optional[float]:
    """Look up the current ask (what we'd pay to enter — same field used by
    the live executor for fill-price simulation) for a token across linked markets."""
    if not token_id:
        return None
    for mid, mkt in linked_markets.items():
        if token_id == getattr(mkt, "token_id_a", None):
            return float(getattr(mkt, "price_a", 0) or 0)
        if token_id == getattr(mkt, "token_id_b", None):
            return float(getattr(mkt, "price_b", 0) or 0)
    return None


def _exit_reason(fill: float, current: float, hold_s: float,
                 tp_pct: float, sl_pct: float, timeout_s: float) -> Optional[str]:
    """Replicate executor.check_exit_conditions for shadow rows."""
    if current <= 0.01:
        # Token effectively dust — don't exit, let timeout handle it.
        return "time_exit" if hold_s >= timeout_s * 2 else None
    if current >= fill * (1 + tp_pct):
        return "take_profit"
    if current <= fill * (1 - sl_pct):
        return "stop_loss"
    if hold_s >= timeout_s:
        return "time_exit"
    return None


class PaperFillSimulator:
    """Polls shadow_trades and writes simulated exits.

    Designed to be called from a thread in bot.py via `tick()`. Doesn't open
    long-lived connections — sqlite over a fresh connection per tick is fine
    at this volume (typically <50 open shadow trades).
    """

    def __init__(self, db_path: Path = DB_PATH) -> None:
        self.db_path = db_path
        self._schema_checked = False
        self.exits_recorded = 0
        self._last_log = 0.0

    def tick(self, linked_markets: dict) -> int:
        """Process all open shadow rows. Returns count of new exits."""
        if not self.db_path.exists():
            return 0
        try:
            conn = sqlite3.connect(str(self.db_path))
        except sqlite3.OperationalError as e:
            logger.warning(f"[PAPER-FILL] db open failed: {e}")
            return 0
        try:
            if not self._schema_checked:
                _ensure_schema(conn)
                self._schema_checked = True

            cur = conn.execute(
                "SELECT id, ts, token_id, fill_price, tp_pct, sl_pct "
                "FROM shadow_trades WHERE exit_price IS NULL"
            )
            rows = cur.fetchall()
            if not rows:
                return 0

            now = time.time()
            timeout_s = float(getattr(config, "EXIT_TIMEOUT_SECONDS", 600))
            tp_default = float(getattr(config, "TAKE_PROFIT_PCT", 0.075))
            sl_default = float(getattr(config, "STOP_LOSS_PCT", 0.125))

            updated = 0
            for row_id, ts, token_id, fill, row_tp, row_sl in rows:
                fill = float(fill or 0)
                if fill <= 0:
                    continue
                current = _current_price_for_token(linked_markets, token_id)
                if current is None:
                    continue
                hold_s = now - float(ts)
                tp = float(row_tp) if row_tp else tp_default
                sl = float(row_sl) if row_sl else sl_default
                reason = _exit_reason(fill, current, hold_s, tp, sl, timeout_s)
                if not reason:
                    continue
                pnl_pct = (current - fill) / fill * 100.0
                conn.execute(
                    "UPDATE shadow_trades SET exit_price=?, exit_reason=?, "
                    "exit_ts=?, hold_seconds=?, pnl_pct=? WHERE id=?",
                    (current, reason, now, hold_s, pnl_pct, row_id),
                )
                updated += 1
                self.exits_recorded += 1
                logger.info(
                    f"[PAPER-FILL] id={row_id} {reason.upper()} "
                    f"fill={fill:.3f} exit={current:.3f} hold={hold_s:.0f}s pnl={pnl_pct:+.1f}%"
                )
            if updated:
                conn.commit()

            # Heartbeat every 60s
            if now - self._last_log > 60:
                self._last_log = now
                open_n = len(rows)
                logger.info(
                    f"[PAPER-FILL] tick: {open_n} open shadow rows, "
                    f"{self.exits_recorded} total simulated exits"
                )
            return updated
        finally:
            conn.close()
