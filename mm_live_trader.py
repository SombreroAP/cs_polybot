"""
Live market-maker order placement / cancellation for Polymarket CLOB.

Gated behind the env var MM_LIVE_TRADING=true. When disabled (default),
all methods are no-ops that just log what they would do — useful for
running the strategy in dual mode (shadow logging continues, live calls
are visible but inert).

When enabled:
  - Wallet credentials come from env (POLYMARKET_PRIVATE_KEY or API key set)
  - Orders are placed via py-clob-client
  - Hard safety limits (max daily loss, max per-token inventory, max quote
    size) prevent runaway exposure
  - Telegram alert on every fill and any error

Env vars required to actually trade:
  MM_LIVE_TRADING=true
  POLYMARKET_PRIVATE_KEY=<hex>           # OR set the API trio:
  POLYMARKET_API_KEY=...                  # POLYMARKET_API_SECRET=...
                                          # POLYMARKET_API_PASSPHRASE=...
  POLYMARKET_PROXY_WALLET=<0x address>    # proxy/EOA address — REQUIRED
  TELEGRAM_BOT_TOKEN, TELEGRAM_USER_ID    # for fill alerts

Safety env vars (have sensible defaults):
  MM_MAX_DAILY_LOSS_USD=10                # kill switch
  MM_MAX_INVENTORY_USD_PER_TOKEN=20       # per-token exposure cap
  MM_MAX_QUOTE_SIZE_USD=5                 # per-order $ cap
"""
from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

log = logging.getLogger("mm_trader")


# ────────────────────────────────────────────────────────────────────────
# Safety limits (env-overridable)
# ────────────────────────────────────────────────────────────────────────
def _envf(key: str, default: float) -> float:
    try:
        return float(os.environ.get(key, default))
    except Exception:
        return default


MM_LIVE_TRADING = os.environ.get("MM_LIVE_TRADING", "false").lower() == "true"
MAX_DAILY_LOSS = _envf("MM_MAX_DAILY_LOSS_USD", 10.0)
MAX_INV_PER_TOKEN = _envf("MM_MAX_INVENTORY_USD_PER_TOKEN", 20.0)
MAX_QUOTE_SIZE = _envf("MM_MAX_QUOTE_SIZE_USD", 5.0)


# ────────────────────────────────────────────────────────────────────────
# Telegram helper (best-effort)
# ────────────────────────────────────────────────────────────────────────
def _tg(text: str):
    tok = os.environ.get("TELEGRAM_BOT_TOKEN")
    uid = os.environ.get("TELEGRAM_USER_ID")
    if not tok or not uid:
        return
    try:
        import urllib.request, urllib.parse
        url = f"https://api.telegram.org/bot{tok}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": uid, "text": text}).encode()
        urllib.request.urlopen(url, data=data, timeout=5)
    except Exception as e:
        log.debug(f"[MM-TRADER] telegram failed: {e}")


# ────────────────────────────────────────────────────────────────────────
# Trader
# ────────────────────────────────────────────────────────────────────────
@dataclass
class LiveTrader:
    """Wraps py-clob-client with safety limits + telemetry.

    All public methods are safe to call when MM_LIVE_TRADING is false —
    they log the intent and return False/None instead of placing orders.
    """
    enabled: bool = MM_LIVE_TRADING
    _client: object = None
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # Tracked active orders: { token_id: { "bid": order_id, "ask": order_id } }
    _active_orders: dict = field(default_factory=dict)
    # Tracked realized PnL today (USD)
    _daily_realized_pnl: float = 0.0
    _daily_started_at: float = field(default_factory=time.time)
    # Per-token inventory in USD (best-effort tracking)
    _inv_usd: dict = field(default_factory=dict)
    # Hard kill switch (set when daily loss exceeded)
    _killed: bool = False
    # Real-fill tracking: trade-ids already recorded (so polling is idempotent)
    _seen_trade_ids: set = field(default_factory=set)
    _real_fills_db: str = ""

    def __post_init__(self):
        if not self.enabled:
            log.info("[MM-TRADER] LIVE trading DISABLED (MM_LIVE_TRADING != true). "
                     "Module loaded in no-op mode.")
            return
        try:
            self._init_client()
            self._init_real_fills_db()
            _tg(f"🟢 MM live-trader activated. Limits: max_daily_loss=${MAX_DAILY_LOSS}, "
                f"max_inv/token=${MAX_INV_PER_TOKEN}, max_quote=${MAX_QUOTE_SIZE}")
        except Exception as e:
            log.error(f"[MM-TRADER] init failed, disabling: {e}")
            self.enabled = False
            _tg(f"🔴 MM live-trader init FAILED: {str(e)[:200]}")

    def _init_real_fills_db(self):
        """Create the mm_real_fills table — captures ACTUAL Polymarket fills so
        we can compare real vs simulated and validate the strategy live."""
        import sqlite3
        from pathlib import Path
        self._real_fills_db = str(Path(__file__).resolve().parent / "data" / "mm.db")
        try:
            with sqlite3.connect(self._real_fills_db, timeout=10) as c:
                c.execute("PRAGMA busy_timeout=10000")
                c.execute("""CREATE TABLE IF NOT EXISTS mm_real_fills (
                    trade_id TEXT PRIMARY KEY, ts REAL, token_id TEXT, side TEXT,
                    price REAL, size REAL, status TEXT, raw TEXT)""")
                c.execute("CREATE INDEX IF NOT EXISTS ix_realfills_ts ON mm_real_fills(ts)")
                # preload seen ids so a restart doesn't double-count
                for (tid,) in c.execute("SELECT trade_id FROM mm_real_fills"):
                    self._seen_trade_ids.add(tid)
            log.info(f"[MM-TRADER] real-fill recording ready ({len(self._seen_trade_ids)} prior)")
        except Exception as e:
            log.warning(f"[MM-TRADER] real-fills db init failed: {e}")

    def poll_real_fills(self) -> int:
        """Poll Polymarket for our recent trades; record any new ones, update
        inventory + daily PnL + kill switch. Returns count of new fills.
        Called by a background thread in bot.py every few seconds when live."""
        if not self.enabled or self._client is None:
            return 0
        try:
            trades = self._client.get_trades()
        except Exception as e:
            log.debug(f"[MM-TRADER] get_trades failed: {e}")
            return 0
        if not isinstance(trades, list):
            trades = (trades or {}).get("data", []) if isinstance(trades, dict) else []
        new = 0
        import sqlite3, json as _json
        for t in trades:
            try:
                tid = str(t.get("id") or t.get("trade_id") or t.get("transaction_hash") or "")
                if not tid or tid in self._seen_trade_ids:
                    continue
                side = (t.get("side") or "").upper()
                price = float(t.get("price") or 0)
                size = float(t.get("size") or t.get("matched_amount") or 0)
                token_id = str(t.get("asset_id") or t.get("token_id") or "")
                status = t.get("status") or ""
                if price <= 0 or size <= 0:
                    continue
                with sqlite3.connect(self._real_fills_db, timeout=10) as c:
                    c.execute("PRAGMA busy_timeout=10000")
                    c.execute("INSERT OR IGNORE INTO mm_real_fills "
                              "(trade_id, ts, token_id, side, price, size, status, raw) "
                              "VALUES (?,?,?,?,?,?,?,?)",
                              (tid, time.time(), token_id, side, price, size, status,
                               _json.dumps(t)[:2000]))
                self._seen_trade_ids.add(tid)
                self.on_fill(token_id, side, price, size)
                new += 1
            except Exception as e:
                log.debug(f"[MM-TRADER] trade parse err: {e}")
        if new:
            log.info(f"[MM-TRADER] recorded {new} REAL fills")
        return new

    def _init_client(self):
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON

        host = os.environ.get("POLYMARKET_CLOB_HOST", "https://clob.polymarket.com")
        priv_key = os.environ.get("POLYMARKET_PRIVATE_KEY")
        proxy = os.environ.get("POLYMARKET_PROXY_WALLET")

        if not priv_key:
            raise RuntimeError("POLYMARKET_PRIVATE_KEY not set")
        if not proxy:
            raise RuntimeError("POLYMARKET_PROXY_WALLET not set")

        # Level 2 (signed order placement) requires creating/deriving API creds
        self._client = ClobClient(
            host=host,
            key=priv_key,
            chain_id=POLYGON,
            signature_type=2,  # email/magic proxy wallet
            funder=proxy,
        )
        # Derive API creds (read-only ok; orders signed with priv key)
        try:
            creds = self._client.create_or_derive_api_creds()
            self._client.set_api_creds(creds)
            log.info("[MM-TRADER] connected; API creds derived")
        except Exception as e:
            log.warning(f"[MM-TRADER] could not derive API creds (orders may still work): {e}")
        log.info(f"[MM-TRADER] address: {self._client.get_address()}")

    # ────────────────────────────────────────────────────────────────────
    # Safety gates
    # ────────────────────────────────────────────────────────────────────
    def _check_kill_switch(self) -> bool:
        """Return True if we should NOT trade further today."""
        if self._killed:
            return True
        if abs(self._daily_realized_pnl) >= MAX_DAILY_LOSS and self._daily_realized_pnl < 0:
            self._killed = True
            msg = (f"🔴 MM KILL SWITCH: daily realized PnL ${self._daily_realized_pnl:.2f} "
                   f"≤ -${MAX_DAILY_LOSS}. NO MORE ORDERS TODAY.")
            log.error(msg)
            _tg(msg)
            try:
                if self.enabled and self._client is not None:
                    self._client.cancel_all()
            except Exception:
                pass
            return True
        # Daily reset at UTC midnight
        if time.time() - self._daily_started_at > 86400:
            self._daily_realized_pnl = 0.0
            self._daily_started_at = time.time()
            self._killed = False
        return False

    def _check_inventory(self, token_id: str, side: str, price: float, size: float) -> bool:
        """True if the order would breach inventory cap for this token."""
        delta = price * size if side.upper() == "BUY" else -price * size
        projected = self._inv_usd.get(token_id, 0.0) + delta
        if abs(projected) > MAX_INV_PER_TOKEN:
            log.warning(f"[MM-TRADER] {token_id[:8]} side={side} would push inv "
                        f"to ${projected:.2f} (cap ${MAX_INV_PER_TOKEN})")
            return False
        return True

    # ────────────────────────────────────────────────────────────────────
    # Public API
    # ────────────────────────────────────────────────────────────────────
    def quote(self, token_id: str, bid_price: Optional[float],
              ask_price: Optional[float], size_shares: float) -> dict:
        """Cancel any existing quote for this token, place new bid+ask.
        Returns dict with intended action + (if live) any order IDs / errors."""
        result = {"token_id": token_id, "live": self.enabled, "actions": []}
        with self._lock:
            # Kill-switch check
            if self._check_kill_switch():
                result["killed"] = True
                return result

            # Cancel existing quote first (always safe to do)
            self.cancel(token_id, _locked=True)

            # Sizing — cap quote size to MAX_QUOTE_SIZE USD
            if bid_price:
                size_usd = bid_price * size_shares
                if size_usd > MAX_QUOTE_SIZE:
                    size_shares = MAX_QUOTE_SIZE / bid_price
                    result["actions"].append(f"bid resized to ${MAX_QUOTE_SIZE:.2f}")

            # Place bid
            if bid_price is not None and bid_price > 0:
                if self._check_inventory(token_id, "BUY", bid_price, size_shares):
                    if self.enabled:
                        try:
                            order_id = self._place(token_id, "BUY", bid_price, size_shares)
                            self._active_orders.setdefault(token_id, {})["bid"] = order_id
                            result["actions"].append(f"BID @{bid_price:.4f} x{size_shares} id={order_id[:10]}")
                            _tg(f"🟢 BID placed: {token_id[:8]} @{bid_price:.4f} x{size_shares}")
                        except Exception as e:
                            log.error(f"[MM-TRADER] BID place failed: {e}")
                            result["actions"].append(f"BID FAIL: {str(e)[:80]}")
                            _tg(f"🔴 BID place failed: {token_id[:8]}: {str(e)[:120]}")
                    else:
                        result["actions"].append(f"WOULD BID @{bid_price:.4f} x{size_shares}")
                else:
                    result["actions"].append("BID skipped (inventory cap)")

            # Place ask (sell)
            if ask_price is not None and ask_price > 0:
                if self._check_inventory(token_id, "SELL", ask_price, size_shares):
                    if self.enabled:
                        try:
                            order_id = self._place(token_id, "SELL", ask_price, size_shares)
                            self._active_orders.setdefault(token_id, {})["ask"] = order_id
                            result["actions"].append(f"ASK @{ask_price:.4f} x{size_shares} id={order_id[:10]}")
                            _tg(f"🔵 ASK placed: {token_id[:8]} @{ask_price:.4f} x{size_shares}")
                        except Exception as e:
                            log.error(f"[MM-TRADER] ASK place failed: {e}")
                            result["actions"].append(f"ASK FAIL: {str(e)[:80]}")
                            _tg(f"🔴 ASK place failed: {token_id[:8]}: {str(e)[:120]}")
                    else:
                        result["actions"].append(f"WOULD ASK @{ask_price:.4f} x{size_shares}")
                else:
                    result["actions"].append("ASK skipped (inventory cap)")

        return result

    def _place(self, token_id: str, side: str, price: float, size: float) -> str:
        """Actually place an order. Returns order_id."""
        from py_clob_client.clob_types import OrderArgs, OrderType
        args = OrderArgs(token_id=token_id, price=round(price, 4),
                         size=round(size, 4), side=side)
        signed = self._client.create_order(args)
        resp = self._client.post_order(signed, OrderType.GTC)
        return resp.get("orderID", "")

    def cancel(self, token_id: str, _locked: bool = False) -> dict:
        """Cancel both sides for this token."""
        result = {"token_id": token_id, "cancelled": []}
        if not _locked:
            self._lock.acquire()
        try:
            ids = self._active_orders.pop(token_id, {})
            if not ids:
                return result
            for side, oid in ids.items():
                if not oid:
                    continue
                if self.enabled:
                    try:
                        self._client.cancel(order_id=oid)
                        result["cancelled"].append(f"{side}={oid[:10]}")
                    except Exception as e:
                        log.warning(f"[MM-TRADER] cancel {side} {oid[:10]} failed: {e}")
                else:
                    result["cancelled"].append(f"WOULD cancel {side}={oid}")
        finally:
            if not _locked:
                self._lock.release()
        return result

    def on_fill(self, token_id: str, side: str, price: float, size: float):
        """Update inventory tracking + realized PnL on a confirmed fill."""
        with self._lock:
            delta = price * size if side.upper() == "BUY" else -price * size
            self._inv_usd[token_id] = self._inv_usd.get(token_id, 0.0) + delta
            # Realized PnL: on a paired round-trip, the cash flow nets out
            # to a positive (spread captured). Track per-fill cash flow:
            cash = -price * size if side.upper() == "BUY" else price * size
            self._daily_realized_pnl += cash
            _tg(f"✅ FILL {side} {token_id[:8]} @{price:.4f} x{size:.2f}  "
                f"inv_usd=${self._inv_usd[token_id]:.2f}  "
                f"day_pnl=${self._daily_realized_pnl:.2f}")
            self._check_kill_switch()

    def status(self) -> dict:
        with self._lock:
            return {
                "enabled": self.enabled,
                "killed": self._killed,
                "daily_realized_pnl": round(self._daily_realized_pnl, 4),
                "max_daily_loss": MAX_DAILY_LOSS,
                "max_inv_per_token": MAX_INV_PER_TOKEN,
                "max_quote_size": MAX_QUOTE_SIZE,
                "active_quotes": len(self._active_orders),
                "tokens_with_inv": {k: round(v, 2) for k, v in self._inv_usd.items() if abs(v) > 0.01},
            }


# Singleton
_singleton: Optional[LiveTrader] = None
_singleton_lock = threading.Lock()


def get_trader() -> LiveTrader:
    global _singleton
    with _singleton_lock:
        if _singleton is None:
            _singleton = LiveTrader()
    return _singleton
