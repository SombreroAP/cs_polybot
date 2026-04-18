"""
SQLite persistence layer for trade positions and bot state.
Stores data in data/trades.db so it survives bot restarts.
"""
import json
import os
import sqlite3
import logging
import threading
from typing import Optional

logger = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_DB_PATH = os.path.join(_PROJECT_ROOT, "data", "trades.db")


class TradeDB:
    """SQLite-backed storage for positions and bot state.

    Concurrency model: SQLite + WAL allows many concurrent readers but only ONE
    writer at a time. Multiple bot threads (position monitor, edge analyst,
    market scanner, executor) all share this single connection. Without a
    write mutex, two threads calling .execute() simultaneously can race —
    the second one gets `database is locked` even though we've set busy_timeout.
    The lock here serializes ALL writes so the timeout never trips.
    """

    def __init__(self, db_path: str = _DEFAULT_DB_PATH):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=60)
        self._conn.execute("PRAGMA busy_timeout = 60000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")  # WAL-safe + ~2x faster writes
        self._write_lock = threading.Lock()
        self._init_schema()
        logger.info(f"Trade DB opened at {db_path}")

    def _init_schema(self):
        c = self._conn
        c.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                order_id TEXT NOT NULL UNIQUE,
                market_id TEXT NOT NULL,
                token_id TEXT NOT NULL,
                team TEXT NOT NULL,
                game TEXT NOT NULL,
                direction TEXT NOT NULL,
                amount REAL NOT NULL,
                fill_price REAL NOT NULL,
                shares REAL NOT NULL,
                timestamp REAL NOT NULL,
                resolved INTEGER NOT NULL DEFAULT 0,
                won INTEGER,
                pnl REAL NOT NULL DEFAULT 0.0,
                signal_confidence REAL NOT NULL DEFAULT 0.0,
                latency_edge_ms REAL NOT NULL DEFAULT 0.0,
                market_price_at_entry REAL NOT NULL DEFAULT 0.0,
                our_price_at_entry REAL NOT NULL DEFAULT 0.0,
                strategy TEXT NOT NULL DEFAULT 'claude'
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS bot_state (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                balance REAL NOT NULL,
                starting_balance REAL NOT NULL,
                total_fees_paid REAL NOT NULL DEFAULT 0.0,
                trade_count INTEGER NOT NULL DEFAULT 0
            )
        """)
        c.execute("""
            CREATE TABLE IF NOT EXISTS team_cache (
                team_name TEXT NOT NULL,
                game TEXT NOT NULL,
                team_id TEXT DEFAULT '',
                ranking INTEGER DEFAULT 0,
                rating REAL DEFAULT 0.0,
                recent_form REAL DEFAULT 0.5,
                recent_matches INTEGER DEFAULT 0,
                raw_data TEXT DEFAULT '',
                last_updated REAL NOT NULL,
                UNIQUE(team_name, game)
            )
        """)
        # ─── Claude Decisions (ALL decisions, not just recent) ─────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS claude_decisions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                action TEXT NOT NULL,
                confidence REAL NOT NULL,
                reason TEXT NOT NULL,
                team TEXT NOT NULL,
                game TEXT NOT NULL,
                match_id TEXT DEFAULT '',
                market_type TEXT DEFAULT '',
                market_price REAL DEFAULT 0,
                buy_price REAL DEFAULT 0,
                spread REAL DEFAULT 0,
                gold_lead REAL DEFAULT 0,
                kills_a INTEGER DEFAULT 0,
                kills_b INTEGER DEFAULT 0,
                game_minutes REAL DEFAULT 0,
                events_analyzed INTEGER DEFAULT 0,
                event_window TEXT DEFAULT '',
                game_state TEXT DEFAULT '',
                market_state TEXT DEFAULT '',
                cost REAL DEFAULT 0,
                trade_opened INTEGER DEFAULT 0
            )
        """)

        # ─── Loss Analysis (auto-categorized on every stop loss) ──────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS loss_analysis (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                order_id TEXT NOT NULL,
                team TEXT NOT NULL,
                game TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                pnl_pct REAL NOT NULL,
                age_seconds REAL DEFAULT 0,
                spread_at_entry REAL DEFAULT 0,
                category TEXT NOT NULL,
                details TEXT NOT NULL,
                market_type TEXT DEFAULT '',
                market_volume REAL DEFAULT 0,
                our_price REAL DEFAULT 0,
                market_price REAL DEFAULT 0
            )
        """)

        # ─── Match Events (game events for post-mortem analysis) ──────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS match_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                match_id TEXT NOT NULL,
                game TEXT NOT NULL,
                event_type TEXT NOT NULL,
                team TEXT NOT NULL,
                description TEXT NOT NULL,
                team_a TEXT DEFAULT '',
                team_b TEXT DEFAULT '',
                score_a INTEGER DEFAULT 0,
                score_b INTEGER DEFAULT 0,
                round_a INTEGER DEFAULT 0,
                round_b INTEGER DEFAULT 0,
                kills_a INTEGER DEFAULT 0,
                kills_b INTEGER DEFAULT 0,
                gold_lead REAL DEFAULT 0,
                economy_a REAL DEFAULT 0,
                economy_b REAL DEFAULT 0,
                market_price REAL DEFAULT 0,
                had_market INTEGER DEFAULT 0
            )
        """)

        # ─── Lessons Learned (bugs, rules, insights) ─────────────────────
        c.execute("""
            CREATE TABLE IF NOT EXISTS lessons_learned (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                category TEXT NOT NULL,
                title TEXT NOT NULL,
                description TEXT NOT NULL,
                impact_dollars REAL DEFAULT 0,
                fix_applied TEXT DEFAULT '',
                active INTEGER DEFAULT 1
            )
        """)

        # Migration: add analysis column if missing
        try:
            c.execute("SELECT analysis FROM positions LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE positions ADD COLUMN analysis TEXT DEFAULT ''")
            logger.info("Migrated DB: added analysis column to positions")
        # Migration: add exit_reason column if missing
        try:
            c.execute("SELECT exit_reason FROM positions LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE positions ADD COLUMN exit_reason TEXT DEFAULT ''")
            logger.info("Migrated DB: added exit_reason column to positions")
        try:
            c.execute("SELECT trigger_event FROM positions LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE positions ADD COLUMN trigger_event TEXT DEFAULT ''")
            logger.info("Migrated DB: added trigger_event column to positions")
        try:
            c.execute("SELECT sell_price FROM positions LIMIT 1")
        except sqlite3.OperationalError:
            c.execute("ALTER TABLE positions ADD COLUMN sell_price REAL DEFAULT 0")
            logger.info("Migrated DB: added sell_price column to positions")
        c.commit()

    # ─── Positions ───────────────────────────────────────────────────────────

    def save_position(self, pos) -> None:
        """Insert a new position. `pos` is an executor.Position dataclass."""
        analysis_json = ''
        if hasattr(pos, 'analysis') and pos.analysis:
            try:
                analysis_json = json.dumps(pos.analysis)
            except (TypeError, ValueError):
                analysis_json = ''
        with self._write_lock:
            self._conn.execute("""
                INSERT OR IGNORE INTO positions
                    (order_id, market_id, token_id, team, game, direction,
                     amount, fill_price, shares, timestamp, resolved, won, pnl,
                     signal_confidence, latency_edge_ms, market_price_at_entry, our_price_at_entry, strategy, analysis,
                     trigger_event, sell_price)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pos.order_id, pos.market_id, pos.token_id, pos.team, pos.game,
                pos.direction, pos.amount, pos.fill_price, pos.shares, pos.timestamp,
                int(pos.resolved), pos.won if pos.won is not None else None, pos.pnl,
                pos.signal_confidence, pos.latency_edge_ms,
                pos.market_price_at_entry, pos.our_price_at_entry,
                getattr(pos, 'strategy', 'claude'),
                analysis_json,
                getattr(pos, 'trigger_event', ''),
                getattr(pos, 'sell_price', 0),
            ))
            self._conn.commit()

    def update_position(self, pos) -> None:
        """Update a position after resolution."""
        with self._write_lock:
            self._conn.execute("""
                UPDATE positions
                SET resolved = ?, won = ?, pnl = ?, exit_reason = ?, trigger_event = ?, sell_price = ?
                WHERE order_id = ?
            """, (int(pos.resolved), 1 if pos.won else 0, pos.pnl,
                  getattr(pos, 'exit_reason', ''), getattr(pos, 'trigger_event', ''),
                  getattr(pos, 'sell_price', 0), pos.order_id))
            self._conn.commit()

    def load_positions(self):
        """Load all positions from DB. Returns list of dicts."""
        cur = self._conn.execute("SELECT * FROM positions ORDER BY timestamp")
        cols = [d[0] for d in cur.description]
        rows = []
        for row in cur.fetchall():
            d = dict(zip(cols, row))
            # Convert SQLite integers back to booleans
            d["resolved"] = bool(d["resolved"])
            d["won"] = bool(d["won"]) if d["won"] is not None else None
            d["exit_reason"] = d.get("exit_reason", "") or ""
            d["trigger_event"] = d.get("trigger_event", "") or ""
            d["sell_price"] = d.get("sell_price", 0) or 0
            # Deserialize analysis JSON
            raw = d.get("analysis", "")
            if raw:
                try:
                    d["analysis"] = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    d["analysis"] = {}
            else:
                d["analysis"] = {}
            rows.append(d)
        return rows

    # ─── Bot State ───────────────────────────────────────────────────────────

    def save_state(self, balance: float, starting_balance: float,
                   total_fees: float, trade_count: int) -> None:
        """Upsert the single bot_state row."""
        with self._write_lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO bot_state (id, balance, starting_balance, total_fees_paid, trade_count)
                VALUES (1, ?, ?, ?, ?)
            """, (balance, starting_balance, total_fees, trade_count))
            self._conn.commit()

    def load_state(self) -> Optional[dict]:
        """Load bot state. Returns dict or None if first run."""
        cur = self._conn.execute("SELECT * FROM bot_state WHERE id = 1")
        row = cur.fetchone()
        if not row:
            return None
        cols = [d[0] for d in cur.description]
        return dict(zip(cols, row))

    # ─── Claude Decisions ────────────────────────────────────────────────────

    def save_claude_decision(self, decision: dict) -> None:
        """Persist every Claude edge decision for historical analysis."""
        try:
            with self._write_lock:
                self._conn.execute("""
                    INSERT INTO claude_decisions
                        (timestamp, action, confidence, reason, team, game, match_id,
                         market_type, market_price, buy_price, spread, gold_lead,
                         kills_a, kills_b, game_minutes, events_analyzed,
                         event_window, game_state, market_state, cost, trade_opened)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    decision.get("timestamp", 0),
                    decision.get("action", ""),
                    decision.get("confidence", 0),
                    decision.get("reason", ""),
                    decision.get("team", ""),
                    decision.get("game", ""),
                    decision.get("match_id", ""),
                    decision.get("market_type", ""),
                    decision.get("market_price", 0),
                    decision.get("buy_price", 0),
                    decision.get("spread", 0),
                    decision.get("gold_lead", 0),
                    decision.get("kills_a", 0),
                    decision.get("kills_b", 0),
                    decision.get("game_minutes", 0),
                    decision.get("events_analyzed", 0),
                    json.dumps(decision.get("event_window", [])),
                    json.dumps(decision.get("game_state", {})),
                    json.dumps(decision.get("market_state", {})),
                    decision.get("cost", 0),
                    1 if decision.get("trade_opened") else 0,
                ))
                self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save Claude decision: {e}")

    # ─── Loss Analysis ───────────────────────────────────────────────────────

    def save_loss_analysis(self, analysis: dict) -> None:
        """Persist loss root cause analysis for every stop loss."""
        try:
            with self._write_lock:
                self._conn.execute("""
                    INSERT INTO loss_analysis
                        (timestamp, order_id, team, game, entry_price, exit_price,
                         pnl, pnl_pct, age_seconds, spread_at_entry, category,
                         details, market_type, market_volume, our_price, market_price)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    analysis.get("timestamp", 0),
                    analysis.get("order_id", ""),
                    analysis.get("team", ""),
                    analysis.get("game", ""),
                    analysis.get("entry_price", 0),
                    analysis.get("exit_price", 0),
                    analysis.get("pnl", 0),
                    analysis.get("pnl_pct", 0),
                    analysis.get("age_seconds", 0),
                    analysis.get("spread_at_entry", 0),
                    analysis.get("category", "unknown"),
                    analysis.get("details", ""),
                    analysis.get("market_type", ""),
                    analysis.get("market_volume", 0),
                    analysis.get("our_price", 0),
                    analysis.get("market_price", 0),
                ))
                self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save loss analysis: {e}")

    # ─── Match Events ────────────────────────────────────────────────────────

    def save_match_event(self, event: dict) -> None:
        """Persist game events for matches with Polymarket markets."""
        try:
            with self._write_lock:
                self._conn.execute("""
                    INSERT INTO match_events
                        (timestamp, match_id, game, event_type, team, description,
                         team_a, team_b, score_a, score_b, round_a, round_b,
                         kills_a, kills_b, gold_lead, economy_a, economy_b,
                         market_price, had_market)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    event.get("timestamp", 0),
                    event.get("match_id", ""),
                    event.get("game", ""),
                    event.get("event_type", ""),
                    event.get("team", ""),
                    event.get("description", "")[:200],
                    event.get("team_a", ""),
                    event.get("team_b", ""),
                    event.get("score_a", 0),
                    event.get("score_b", 0),
                    event.get("round_a", 0),
                    event.get("round_b", 0),
                    event.get("kills_a", 0),
                    event.get("kills_b", 0),
                    event.get("gold_lead", 0),
                    event.get("economy_a", 0),
                    event.get("economy_b", 0),
                    event.get("market_price", 0),
                    1 if event.get("had_market") else 0,
                ))
                # Batch commit — only every 200 events to reduce lock contention
                if not hasattr(self, '_event_count'):
                    self._event_count = 0
                self._event_count += 1
                if self._event_count % 200 == 0:
                    self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save match event: {e}")

    def flush_events(self) -> None:
        """Force commit any pending match events."""
        try:
            with self._write_lock:
                self._conn.commit()
        except Exception:
            pass

    # ─── Lessons Learned ─────────────────────────────────────────────────────

    def save_lesson(self, category: str, title: str, description: str,
                    impact_dollars: float = 0, fix_applied: str = "") -> None:
        """Record a bug, insight, or trading rule discovery."""
        try:
            with self._write_lock:
                self._conn.execute("""
                    INSERT INTO lessons_learned
                        (timestamp, category, title, description, impact_dollars, fix_applied, active)
                    VALUES (?, ?, ?, ?, ?, ?, 1)
                """, (
                    __import__('time').time(),
                    category, title, description, impact_dollars, fix_applied,
                ))
                self._conn.commit()
        except Exception as e:
            logger.error(f"Failed to save lesson: {e}")

    def get_lessons(self) -> list:
        """Get all active lessons for Claude's context."""
        cur = self._conn.execute(
            "SELECT category, title, description, fix_applied FROM lessons_learned WHERE active=1 ORDER BY timestamp"
        )
        return [{"category": r[0], "title": r[1], "description": r[2], "fix": r[3]} for r in cur.fetchall()]

    # ─── Analytics Queries ───────────────────────────────────────────────────

    def get_loss_stats(self) -> dict:
        """Get loss category breakdown."""
        cur = self._conn.execute("""
            SELECT category, COUNT(*) as cnt, SUM(pnl) as total_loss, AVG(pnl_pct) as avg_pct
            FROM loss_analysis GROUP BY category ORDER BY total_loss
        """)
        return {r[0]: {"count": r[1], "total_loss": r[2], "avg_pct": r[3]} for r in cur.fetchall()}

    def get_decision_stats(self) -> dict:
        """Get Claude decision win rate by game."""
        cur = self._conn.execute("""
            SELECT game, action, COUNT(*) as cnt, AVG(confidence) as avg_conf
            FROM claude_decisions GROUP BY game, action
        """)
        stats = {}
        for r in cur.fetchall():
            game = r[0]
            if game not in stats:
                stats[game] = {}
            stats[game][r[1]] = {"count": r[2], "avg_conf": r[3]}
        return stats

    # ─── Lifecycle ───────────────────────────────────────────────────────────

    def close(self):
        if self._conn:
            try:
                self._conn.commit()  # flush any pending events
            except Exception:
                pass
            self._conn.close()
            logger.info("Trade DB closed")
