"""
Polymarket WebSocket — real-time orderbook price updates.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market
Subscribes to token IDs and receives price changes multiple times per second.
"""
import asyncio
import json
import logging
import time
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class PolymarketWebSocket:
    """Real-time price feed from Polymarket CLOB WebSocket."""

    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws: Optional[aiohttp.ClientWebSocketResponse] = None
        self._subscribed_tokens: set[str] = set()
        self._callbacks: list[Callable] = []
        self._running = False
        self._connected = False
        self._reconnect_delay = 1

        # Latest prices per token (updated in real-time)
        self.prices: dict[str, dict] = {}  # token_id -> {best_bid, best_ask, last_trade, timestamp}

        # Price history buffer for momentum detection (last 60 seconds)
        self._price_history: dict[str, list] = {}  # token_id -> [{bid, ask, ts}, ...]
        self._history_max_age = 120  # keep 2 minutes of history

        # Stats
        self.message_count = 0
        self.last_message_time = 0

    def on_price_update(self, callback: Callable):
        """Register callback for price updates: callback(token_id, best_bid, best_ask)"""
        self._callbacks.append(callback)

    async def connect(self):
        """Connect to Polymarket WebSocket."""
        self._running = True
        self._session = aiohttp.ClientSession()
        asyncio.create_task(self._connection_loop())
        logger.info("[WS] Polymarket WebSocket starting")

    async def _connection_loop(self):
        """Maintain persistent WebSocket connection with auto-reconnect."""
        while self._running:
            try:
                async with self._session.ws_connect(WS_URL, heartbeat=30) as ws:
                    self._ws = ws
                    self._connected = True
                    self._reconnect_delay = 1
                    logger.info("[WS] Connected to Polymarket WebSocket")

                    # Re-subscribe to all tokens
                    if self._subscribed_tokens:
                        await self._send_subscribe(list(self._subscribed_tokens))

                    # Read messages
                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self._process_message(msg.data)
                            except Exception as e:
                                if self.message_count < 5:
                                    logger.warning(f"[WS] Parse error: {e} | raw: {msg.data[:200]}")
                        elif msg.type == aiohttp.WSMsgType.ERROR:
                            logger.error(f"[WS] Error: {ws.exception()}")
                            break
                        elif msg.type == aiohttp.WSMsgType.CLOSED:
                            break

            except Exception as e:
                logger.warning(f"[WS] Connection error: {e}")

            self._connected = False
            if self._running:
                logger.info(f"[WS] Reconnecting in {self._reconnect_delay}s...")
                await asyncio.sleep(self._reconnect_delay)
                self._reconnect_delay = min(self._reconnect_delay * 2, 30)

    async def subscribe(self, token_ids: list[str]):
        """Subscribe to real-time updates for these token IDs."""
        new_tokens = [t for t in token_ids if t not in self._subscribed_tokens]
        if not new_tokens:
            return

        self._subscribed_tokens.update(new_tokens)

        # Initialize price entries
        for tid in new_tokens:
            if tid not in self.prices:
                self.prices[tid] = {"best_bid": 0, "best_ask": 0, "last_trade": 0, "timestamp": 0}

        if self._connected and self._ws:
            await self._send_subscribe(new_tokens)

    async def _send_subscribe(self, token_ids: list[str]):
        """Send subscription message to WebSocket."""
        msg = {
            "assets_ids": token_ids,
            "type": "market",
            "custom_feature_enabled": True,
        }
        try:
            await self._ws.send_json(msg)
            logger.info(f"[WS] Subscribed to {len(token_ids)} tokens")
        except Exception as e:
            logger.error(f"[WS] Subscribe error: {e}")

    @staticmethod
    def _parse_price(entry) -> float:
        """Parse price from various formats: dict, list, or string."""
        if isinstance(entry, dict):
            return float(entry.get("price", 0))
        elif isinstance(entry, (list, tuple)) and len(entry) >= 1:
            return float(entry[0])
        elif isinstance(entry, (int, float)):
            return float(entry)
        return 0

    def _process_message(self, raw: str):
        """Process incoming WebSocket message."""
        self.message_count += 1
        self.last_message_time = time.time()

        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Messages can be arrays or objects
        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict):
                    self._process_single(item)
        elif isinstance(data, dict):
            self._process_single(data)

    def _process_single(self, data: dict):
        """Process a single WebSocket event."""
        event_type = data.get("event_type", "")

        if event_type == "book":
            asset_id = str(data.get("asset_id", ""))
            if asset_id and asset_id in self.prices:
                bids = data.get("bids", [])
                asks = data.get("asks", [])
                # Bids ASC / asks DESC → best prices at [-1], not [0].
                real_bids = [b for b in bids if self._parse_price(b) > 0.03]
                real_asks = [a for a in asks if self._parse_price(a) < 0.97]
                best_bid = self._parse_price(real_bids[-1]) if real_bids else 0
                best_ask = self._parse_price(real_asks[-1]) if real_asks else 0
                self.prices[asset_id]["best_bid"] = best_bid
                self.prices[asset_id]["best_ask"] = best_ask
                self.prices[asset_id]["timestamp"] = time.time()
                self._notify(asset_id, best_bid, best_ask)

        elif event_type == "price_change":
            changes = data.get("price_changes", [])
            if not isinstance(changes, list):
                changes = [data]  # single change
            for change in changes:
                if not isinstance(change, dict):
                    continue
                asset_id = str(change.get("asset_id", ""))
                if asset_id and asset_id in self.prices:
                    bb = change.get("best_bid")
                    ba = change.get("best_ask")
                    if bb is not None:
                        self.prices[asset_id]["best_bid"] = float(bb)
                    if ba is not None:
                        self.prices[asset_id]["best_ask"] = float(ba)
                    self.prices[asset_id]["timestamp"] = time.time()
                    self._notify(asset_id, self.prices[asset_id]["best_bid"],
                                 self.prices[asset_id]["best_ask"])

        elif event_type == "last_trade_price":
            asset_id = str(data.get("asset_id", ""))
            if asset_id and asset_id in self.prices:
                price = float(data.get("price", 0))
                self.prices[asset_id]["last_trade"] = price
                self.prices[asset_id]["timestamp"] = time.time()

    def _notify(self, token_id: str, best_bid: float, best_ask: float):
        """Notify callbacks of price change and track history."""
        # Record price history for momentum detection
        now = time.time()
        if token_id not in self._price_history:
            self._price_history[token_id] = []
        self._price_history[token_id].append({
            "bid": best_bid, "ask": best_ask, "ts": now
        })
        # Prune old entries
        cutoff = now - self._history_max_age
        self._price_history[token_id] = [
            p for p in self._price_history[token_id] if p["ts"] > cutoff
        ]

        for cb in self._callbacks:
            try:
                cb(token_id, best_bid, best_ask)
            except Exception:
                pass

    async def close(self):
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()

    def get_price(self, token_id: str) -> Optional[dict]:
        """Get latest price for a token. Returns {best_bid, best_ask, last_trade, timestamp}."""
        return self.prices.get(token_id)

    def is_stale(self, token_id: str, max_age: float = 60) -> bool:
        """Check if price data is stale (no updates in max_age seconds)."""
        p = self.prices.get(token_id)
        if not p or p["timestamp"] == 0:
            return True
        return (time.time() - p["timestamp"]) > max_age

    def get_staleness(self, token_id: str) -> float:
        """Get seconds since last price update."""
        p = self.prices.get(token_id)
        if not p or p["timestamp"] == 0:
            return 999
        return time.time() - p["timestamp"]

    def get_momentum(self, token_id: str, window: float = 30) -> dict:
        """Get price momentum over the last N seconds.
        Returns: {direction, bid_change, trades, avg_bid, trend}"""
        history = self._price_history.get(token_id, [])
        if len(history) < 2:
            return {"direction": "unknown", "bid_change": 0, "trades": 0, "avg_bid": 0, "trend": "flat"}

        cutoff = time.time() - window
        recent = [p for p in history if p["ts"] > cutoff]
        if len(recent) < 2:
            return {"direction": "unknown", "bid_change": 0, "trades": len(recent), "avg_bid": 0, "trend": "flat"}

        first_bid = recent[0]["bid"]
        last_bid = recent[-1]["bid"]
        bid_change = last_bid - first_bid
        avg_bid = sum(p["bid"] for p in recent) / len(recent)

        if bid_change > 0.02:
            trend = "strong_up"
        elif bid_change > 0.005:
            trend = "up"
        elif bid_change < -0.02:
            trend = "strong_down"
        elif bid_change < -0.005:
            trend = "down"
        else:
            trend = "flat"

        return {
            "direction": "up" if bid_change > 0 else "down" if bid_change < 0 else "flat",
            "bid_change": round(bid_change, 4),
            "trades": len(recent),
            "avg_bid": round(avg_bid, 4),
            "trend": trend,
            "first_bid": first_bid,
            "last_bid": last_bid,
        }

    def get_midpoint(self, token_id: str) -> float:
        """Get current price for a token. Prefers last trade price (most accurate),
        falls back to bid/ask midpoint."""
        p = self.prices.get(token_id)
        if not p:
            return 0
        # Last trade price is the real market price
        if p["last_trade"] > 0.01:
            return p["last_trade"]
        # Fallback to midpoint
        bid = p["best_bid"]
        ask = p["best_ask"]
        if bid > 0 and ask > 0:
            return (bid + ask) / 2
        return bid or ask or 0
