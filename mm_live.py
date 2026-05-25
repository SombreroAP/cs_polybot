"""
Live market-making runner — shadow mode (no real orders).

Sits between the bot's book-update flow and the MMStrategy module. For each
observed book update on a token, it:
  1. Detects simulated fills against our previous logged quote
  2. Calls MMStrategy.on_book_update() to decide the new quote
  3. Logs the decision + simulated fill to SQLite (mm_decisions, mm_fills, mm_state)

Critical: this writes NO real orders. It records what the strategy WOULD have
done so we can validate against the backtest before going live.

Schema (auto-created on first call):
  mm_decisions(ts, token_id, match_id, action, bid_price, ask_price,
               book_bid, book_ask, recent_drift_cents, reason)
  mm_fills    (ts, token_id, match_id, side, price, sim_inv_after, sim_cash_after)
  mm_state    (token_id, last_update_ts, inventory, cash, n_fills,
               n_bid_fills, n_ask_fills)
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Dict, Optional

from mm_strategy import MMConfig, MMStrategy

log = logging.getLogger("mm_live")

# Default DB path — separate file from trades.db to avoid lock contention
# with the bot's other writers (claude_decisions, shadow_trades).
DB_PATH = Path(__file__).resolve().parent / "data" / "mm.db"


# ────────────────────────────────────────────────────────────────────────
# Schema
# ────────────────────────────────────────────────────────────────────────
_SCHEMA = """
CREATE TABLE IF NOT EXISTS mm_decisions (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL    NOT NULL,
    token_id     TEXT    NOT NULL,
    match_id     TEXT,
    action       TEXT    NOT NULL,   -- "quote" | "cancel_all" | "noop"
    bid_price    REAL,
    ask_price    REAL,
    book_bid     REAL,
    book_ask     REAL,
    recent_drift_cents REAL,
    reason       TEXT
);
CREATE INDEX IF NOT EXISTS ix_mm_decisions_token_ts ON mm_decisions(token_id, ts);
CREATE INDEX IF NOT EXISTS ix_mm_decisions_ts ON mm_decisions(ts);

CREATE TABLE IF NOT EXISTS mm_fills (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,
    token_id        TEXT    NOT NULL,
    match_id        TEXT,
    side            TEXT    NOT NULL,   -- "bid_fill" | "ask_fill"
    price           REAL    NOT NULL,
    sim_inv_after   REAL,
    sim_cash_after  REAL
);
CREATE INDEX IF NOT EXISTS ix_mm_fills_token_ts ON mm_fills(token_id, ts);
CREATE INDEX IF NOT EXISTS ix_mm_fills_ts ON mm_fills(ts);

CREATE TABLE IF NOT EXISTS mm_state (
    token_id        TEXT    PRIMARY KEY,
    last_update_ts  REAL,
    inventory       REAL,
    cash            REAL,
    n_fills         INTEGER,
    n_bid_fills     INTEGER,
    n_ask_fills     INTEGER
);
"""


# ────────────────────────────────────────────────────────────────────────
# Runner
# ────────────────────────────────────────────────────────────────────────
class LiveMMRunner:
    """Per-token MM strategies, fill detection, SQLite logging. Shadow mode."""

    def __init__(self, db_path: Optional[Path] = None,
                 cfg: Optional[MMConfig] = None):
        import queue as _queue_mod
        self.db_path = Path(db_path) if db_path else DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg or MMConfig()
        self._lock = threading.Lock()
        self._strategies: Dict[str, MMStrategy] = {}
        # Per-token last-quote we logged, used for fill detection on next update
        self._last_quote: Dict[str, dict] = {}
        # Last book state, used to detect transitions
        self._last_book: Dict[str, dict] = {}
        self._init_db()
        self._load_state()
        # Background worker for non-blocking submits.
        # Bounded queue so a flood from a hot loop can never accumulate RAM.
        # When full, oldest items are dropped (most recent book is fresher).
        self._queue: _queue_mod.Queue = _queue_mod.Queue(maxsize=20000)
        self._dropped = 0
        # Producer-side dedup: skip a submit if (bid, ask) is byte-identical to
        # the last one we enqueued for that token. An unchanged book carries no
        # new information for the strategy — this collapses the bulk of the
        # flood (REST poller + WS callback + unlinked feeder all re-report the
        # same book) before it ever touches the queue.
        self._last_submitted: Dict[str, tuple] = {}
        # Persistent connection owned by the worker thread (set in _run_worker).
        # All on_book writes happen on that one thread, so a single long-lived
        # connection replaces the open-connection-per-write hot path that was
        # the real bottleneck (3 PRAGMA round-trips * thousands of writes/sec).
        self._worker_conn: Optional[sqlite3.Connection] = None
        # Throttle state persistence — it's an overwritable checkpoint, no need
        # to write it on every single book tick. {token_id: last_persist_ts}
        self._last_state_persist: Dict[str, float] = {}
        # Game label per token (set by bot.py feeders so the strategy can pick
        # the per-game toxic-flow model). Default "cs2" for any unregistered
        # token to preserve back-compat with the CS2-trained model.
        self._token_game: Dict[str, str] = {}
        self._stop_worker = threading.Event()
        self._worker = threading.Thread(target=self._run_worker, daemon=True,
                                         name="mm-worker")
        self._worker.start()
        log.info(
            "[MM] LiveMMRunner ready (shadow mode). "
            f"db={self.db_path}  cfg={self.cfg}  worker=running"
        )

    # ────── Worker (drains the queue, never blocks producers) ────────────
    def _run_worker(self):
        # Open the worker's long-lived connection once. Every on_book write
        # reuses it instead of opening a fresh connection (with 3 PRAGMAs)
        # per statement — that per-write open was throttling the worker to
        # well under the producer rate, causing the millions of drops.
        try:
            self._worker_conn = self._connect()
        except Exception as e:
            log.warning(f"[MM-WORKER] could not open persistent conn: {e}")
            self._worker_conn = None
        while not self._stop_worker.is_set():
            try:
                item = self._queue.get(timeout=1.0)
            except Exception:
                # idle tick — good moment to run periodic settlement sweep
                self._maybe_settle()
                continue
            self._maybe_settle()
            try:
                token_id, match_id, bid, ask, ts = item
                # Reuse the existing on_book code path; it holds self._lock
                # internally, but only this worker thread calls it, so there's
                # no producer contention.
                self.on_book(token_id, match_id, bid, ask, ts)
            except Exception as e:
                log.debug(f"[MM-WORKER] error processing: {e}")
            finally:
                self._queue.task_done()

    # ────── Settlement of stale / resolved positions ────────────────────────
    # When a market resolves, Polymarket stops streaming its book — the sim
    # would otherwise carry that inventory forever at a stale (fictional) mark.
    # We periodically close any position whose book hasn't updated in
    # SETTLE_STALE_S by realising it at the last observed mid (≈0 or 1 once a
    # market has resolved). Logged as ordinary fills so the DB-derived PnL
    # stays consistent (each unit closes at the settle price).
    SETTLE_STALE_S = 1800.0   # 30 min with no book update → treat as closed
    _settle_interval = 300.0  # run the sweep at most every 5 min

    def _maybe_settle(self):
        now = time.time()
        if now - getattr(self, "_last_settle_sweep", 0.0) < self._settle_interval:
            return
        self._last_settle_sweep = now
        try:
            with self._lock:
                self._settle_stale_positions_locked(now)
        except Exception as e:
            log.debug(f"[MM] settle sweep error: {e}")

    def _settle_stale_positions_locked(self, now: float):
        settled = 0
        for token_id, strat in list(self._strategies.items()):
            inv = getattr(strat, "inventory", 0.0) or 0.0
            if abs(inv) < 1e-9:
                continue
            book = self._last_book.get(token_id)
            if not book or not book.get("ts"):
                continue
            if now - book["ts"] < self.SETTLE_STALE_S:
                continue  # still fresh — leave the open position alone
            mid = (book.get("bid", 0) + book.get("ask", 0)) / 2.0
            if mid <= 0:
                continue
            units = int(round(abs(inv)))
            side = "ask" if inv > 0 else "bid"  # long→sell, short→buy to flatten
            for _ in range(units):
                strat.on_fill(side, mid, size=self.cfg.quote_size, ts=now)
                self._log_fill(now, token_id, None, side + "_fill", mid,
                               strat.inventory, strat.cash)
            strat.inventory = 0.0  # clear any fractional remainder
            self._last_quote[token_id] = {"bid_price": None, "ask_price": None, "ts": now}
            self._persist_state(strat)
            settled += 1
        if settled:
            log.info(f"[MM] settled {settled} stale positions at last mid "
                     f"(>{self.SETTLE_STALE_S/60:.0f}min idle)")

    def register_token_game(self, token_id: str, game: str) -> None:
        """Tag a token with its game so the strategy picks the right model.

        Idempotent: re-calling with the same (token,game) is a no-op. Called
        by bot.py feeders when they discover a market. Default for an
        unregistered token is "cs2" (back-compat).
        """
        if token_id and game:
            self._token_game[token_id] = game

    def submit_book_update(self, token_id: str, match_id: Optional[str],
                          bid: float, ask: float, ts: Optional[float] = None):
        """Non-blocking enqueue. Drops oldest if queue is full so producers
        (price_updater, WS callback) can never be backpressured.

        Coalesces at the source: an unchanged (bid, ask) for a token is
        dropped before it hits the queue — multiple producers re-reporting
        the same book is the dominant source of load and carries no signal.
        """
        import queue as _queue_mod
        if token_id:
            prev = self._last_submitted.get(token_id)
            if prev is not None and prev[0] == bid and prev[1] == ask:
                return  # identical book — nothing changed, skip
            self._last_submitted[token_id] = (bid, ask)
        try:
            self._queue.put_nowait((token_id, match_id, bid, ask, ts or time.time()))
        except _queue_mod.Full:
            # Drop oldest by getting + discarding, then put new
            try:
                self._queue.get_nowait()
                self._dropped += 1
                self._queue.put_nowait((token_id, match_id, bid, ask, ts or time.time()))
            except Exception:
                pass

    def _execute_with_retry(self, sql: str, params: tuple = (), max_attempts: int = 5):
        """Execute a single statement with backoff on busy/locked DB.

        Uses the worker thread's persistent connection (this is only ever
        called from on_book, which runs exclusively on the worker). Falls
        back to a one-off connection if the persistent one isn't up yet.
        """
        last_err = None
        for attempt in range(max_attempts):
            try:
                conn = self._worker_conn
                if conn is not None:
                    conn.execute(sql, params)
                    conn.commit()
                else:
                    with self._connect() as c:
                        c.execute(sql, params)
                return
            except sqlite3.OperationalError as e:
                last_err = e
                if "locked" in str(e).lower() or "busy" in str(e).lower():
                    time.sleep(0.05 * (2 ** attempt))  # 50ms, 100ms, 200ms, 400ms, 800ms
                    continue
                # Non-lock errors get logged at WARNING so they're visible
                log.warning(f"[MM] write error on {sql[:50]}…: {e}")
                return
            except Exception as e:
                log.warning(f"[MM] unexpected write error on {sql[:50]}…: {e}")
                return
        if last_err:
            log.warning(f"[MM] exhausted retries on: {sql[:60]}… err={last_err}")

    def _connect(self):
        """Open a SQLite connection with WAL mode + lock timeout.
        Critical: trades.db is shared with the bot's other writers, so we MUST
        use WAL (allows concurrent readers + a single writer without blocking)
        and a generous timeout for lock acquisition.
        """
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")  # fast + safe for our writes
            conn.execute("PRAGMA busy_timeout=30000")
        except Exception:
            pass
        return conn

    def _init_db(self):
        # Statement-by-statement so any single CREATE that hits a busy lock
        # gets the per-connection 30s timeout, instead of failing the whole
        # script atomically. Splitting on `;` keeps it simple.
        for stmt in [s.strip() for s in _SCHEMA.split(";") if s.strip()]:
            for attempt in range(3):
                try:
                    with self._connect() as c:
                        c.execute(stmt)
                    break
                except sqlite3.OperationalError as e:
                    if "locked" in str(e).lower() and attempt < 2:
                        log.warning(f"[MM] init_db retry on lock: {stmt[:40]}…")
                        time.sleep(1.0 * (attempt + 1))
                        continue
                    raise

    def _load_state(self):
        """Resume in-memory strategy state from mm_state."""
        try:
            with self._connect() as c:
                rows = c.execute("SELECT * FROM mm_state").fetchall()
            for row in rows:
                tok, last_ts, inv, cash, nf, nb, na = row
                s = MMStrategy(token_id=tok, cfg=self.cfg)
                s.inventory = inv or 0.0
                s.cash = cash or 0.0
                s.n_fills = nf or 0
                s.n_bid_fills = nb or 0
                s.n_ask_fills = na or 0
                self._strategies[tok] = s
            if rows:
                log.info(f"[MM] Resumed state for {len(rows)} tokens.")
        except Exception as e:
            log.warning(f"[MM] state load failed (will start fresh): {e}")

    # ────── Public entry point ───────────────────────────────────────────
    _diag_total = 0
    _diag_skipped = 0

    def on_book(self, token_id: str, match_id: Optional[str],
                bid: float, ask: float, ts: Optional[float] = None):
        """Called by the bot on every observed book change for a token."""
        LiveMMRunner._diag_total += 1
        # Heartbeat every 25 calls so we see firing rate
        if LiveMMRunner._diag_total % 25 == 1:
            log.info(f"[MM] on_book heartbeat: total={LiveMMRunner._diag_total} "
                     f"skipped={LiveMMRunner._diag_skipped} "
                     f"tokens_tracked={len(self._strategies)}")

        if not token_id or bid is None or ask is None:
            LiveMMRunner._diag_skipped += 1
            return
        try:
            bid = float(bid); ask = float(ask)
        except Exception:
            LiveMMRunner._diag_skipped += 1
            return
        if bid <= 0 or ask <= 0 or ask <= bid or bid >= 1 or ask >= 1:
            LiveMMRunner._diag_skipped += 1
            return  # malformed or stale book
        ts = ts or time.time()

        with self._lock:
            self._on_book_locked(token_id, match_id, bid, ask, ts)

    def _on_book_locked(self, token_id, match_id, bid, ask, ts):
        # 1. Detect simulated fills against our PREVIOUSLY logged quote
        last_q = self._last_quote.get(token_id)
        last_book = self._last_book.get(token_id, {})
        prev_bid = last_book.get("bid")
        prev_ask = last_book.get("ask")

        strat = self._strategies.get(token_id)
        if strat is None:
            game = self._token_game.get(token_id, "cs2")
            strat = MMStrategy(token_id=token_id, cfg=self.cfg, game=game)
            self._strategies[token_id] = strat

        fill_happened = False
        if last_q and prev_bid is not None and prev_ask is not None:
            our_bid = last_q.get("bid_price")
            our_ask = last_q.get("ask_price")
            # Bid fills when the NEW best_ask drops to or below our bid
            if our_bid is not None and ask <= our_bid + 1e-9:
                strat.on_fill("bid", our_bid, size=self.cfg.quote_size, ts=ts)
                self._log_fill(ts, token_id, match_id, "bid_fill",
                               our_bid, strat.inventory, strat.cash)
                last_q["bid_price"] = None  # consumed
                fill_happened = True
            # Ask fills when the NEW best_bid rises to or above our ask
            if our_ask is not None and bid >= our_ask - 1e-9:
                strat.on_fill("ask", our_ask, size=self.cfg.quote_size, ts=ts)
                self._log_fill(ts, token_id, match_id, "ask_fill",
                               our_ask, strat.inventory, strat.cash)
                last_q["ask_price"] = None
                fill_happened = True

        # 2. Get strategy's decision for the new book
        decision = strat.on_book_update(bid, ask, ts=ts)

        # 3. Compute recent drift (informational, for the DB)
        drift = strat._recent_drift(ts) if hasattr(strat, "_recent_drift") else 0.0

        # 4. Log decision + remember our new quote
        if decision is not None:
            self._log_decision(ts, token_id, match_id, decision,
                               bid, ask, drift)
            if decision.get("action") == "cancel_and_replace":
                self._last_quote[token_id] = {
                    "bid_price": decision.get("new_bid_price"),
                    "ask_price": decision.get("new_ask_price"),
                    "ts": ts,
                }
                # Fire to live trader (no-op if MM_LIVE_TRADING != true)
                try:
                    from mm_live_trader import get_trader
                    get_trader().quote(
                        token_id,
                        decision.get("new_bid_price"),
                        decision.get("new_ask_price"),
                        decision.get("size", 1.0),
                    )
                except Exception as e:
                    log.debug(f"[MM] live-trader quote err: {e}")
            elif decision.get("action") == "cancel_all":
                self._last_quote[token_id] = {"bid_price": None,
                                              "ask_price": None, "ts": ts}
                try:
                    from mm_live_trader import get_trader
                    get_trader().cancel(token_id)
                except Exception as e:
                    log.debug(f"[MM] live-trader cancel err: {e}")

        # 5. Update last book + persist state.
        # State is an overwritable checkpoint — throttle to once / 5s per token
        # so the hot path isn't doing a DB write on every single book tick.
        # A fill always forces an immediate persist (see below) so PnL-relevant
        # state is never more than one tick stale.
        self._last_book[token_id] = {"bid": bid, "ask": ask, "ts": ts}
        now = time.time()
        last_persist = self._last_state_persist.get(token_id, 0.0)
        if now - last_persist > 5.0 or fill_happened:
            self._persist_state(strat)
            self._last_state_persist[token_id] = now

    # ────── DB writers ──────────────────────────────────────────────────
    def _log_decision(self, ts, token_id, match_id, decision,
                      book_bid, book_ask, drift):
        self._execute_with_retry(
            "INSERT INTO mm_decisions "
            "(ts, token_id, match_id, action, bid_price, ask_price, "
            "book_bid, book_ask, recent_drift_cents, reason) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            (ts, token_id, match_id, decision.get("action"),
             decision.get("new_bid_price"), decision.get("new_ask_price"),
             book_bid, book_ask, drift * 100, decision.get("reason"))
        )

    def _log_fill(self, ts, token_id, match_id, side, price, inv_after, cash_after):
        self._execute_with_retry(
            "INSERT INTO mm_fills "
            "(ts, token_id, match_id, side, price, sim_inv_after, sim_cash_after) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, token_id, match_id, side, price, inv_after, cash_after)
        )
        log.info(
            f"[MM] SIM-FILL {side} on {token_id[:8]}…@{price:.4f}  "
            f"inv={inv_after}  cash=${cash_after:+.4f}"
        )

    def _persist_state(self, strat: MMStrategy):
        # State row is overwritable; quietly skip on retry exhaustion.
        try:
            self._execute_with_retry(
                "INSERT OR REPLACE INTO mm_state "
                "(token_id, last_update_ts, inventory, cash, n_fills, "
                "n_bid_fills, n_ask_fills) VALUES (?,?,?,?,?,?,?)",
                (strat.token_id, time.time(),
                 strat.inventory, strat.cash, strat.n_fills,
                 strat.n_bid_fills, strat.n_ask_fills)
            )
        except Exception:
            pass

    # ────── Stats interface (for dashboard) ─────────────────────────────
    def aggregate_stats(self) -> dict:
        """Quick aggregate stats from in-memory + DB."""
        with self._lock:
            n_tokens = len(self._strategies)
            total_fills = sum(s.n_fills for s in self._strategies.values())
            total_cash = sum(s.cash for s in self._strategies.values())
            active_now = sum(1 for tok, q in self._last_quote.items()
                             if q.get("bid_price") is not None
                             or q.get("ask_price") is not None)
        # Sim PnL per fill
        avg_pf = (total_cash / total_fills * 100) if total_fills > 0 else 0.0
        # Pull recent decisions count + drift-filter activations from DB
        try:
            cutoff = time.time() - 24 * 3600
            with self._connect() as c:
                row = c.execute(
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN reason='drift_filter' THEN 1 ELSE 0 END) "
                    "FROM mm_decisions WHERE ts > ?", (cutoff,)).fetchone()
                decisions_24h = row[0] or 0
                drift_pulls_24h = row[1] or 0
                fills_24h = c.execute(
                    "SELECT COUNT(*) FROM mm_fills WHERE ts > ?",
                    (cutoff,)).fetchone()[0] or 0
                cash_24h = c.execute(
                    "SELECT COALESCE(SUM(CASE WHEN side='ask_fill' THEN price "
                    "ELSE -price END), 0) FROM mm_fills WHERE ts > ?",
                    (cutoff,)).fetchone()[0] or 0.0
        except Exception:
            decisions_24h = drift_pulls_24h = fills_24h = 0
            cash_24h = 0.0
        return {
            "n_tokens_tracked": n_tokens,
            "active_quotes_now": active_now,
            "total_fills_lifetime": total_fills,
            "total_sim_cash_pnl": round(total_cash, 4),
            "avg_pnl_per_fill_cents": round(avg_pf, 2),
            "decisions_24h": decisions_24h,
            "drift_filter_pulls_24h": drift_pulls_24h,
            "fills_24h": fills_24h,
            "cash_24h": round(cash_24h, 4),
            "queue_depth": self._queue.qsize(),
            "queue_dropped_lifetime": self._dropped,
            "diag_total_calls": LiveMMRunner._diag_total,
            "diag_skipped": LiveMMRunner._diag_skipped,
        }

    def top_tokens(self, n: int = 10):
        with self._lock:
            ranked = sorted(self._strategies.values(),
                            key=lambda s: -s.n_fills)[:n]
        return [
            {"token_id": s.token_id, "n_fills": s.n_fills,
             "inventory": s.inventory, "cash": round(s.cash, 4)}
            for s in ranked
        ]


# ────────────────────────────────────────────────────────────────────────
# Singleton accessor (used by bot.py)
# ────────────────────────────────────────────────────────────────────────
_singleton: Optional[LiveMMRunner] = None
_singleton_lock = threading.Lock()


def get_runner() -> LiveMMRunner:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LiveMMRunner()
    return _singleton
