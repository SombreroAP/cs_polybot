"""
Price Impact Tracker — measures how game events move Polymarket prices.

Records price before/after each game event to find which events
create the biggest, fastest market movements.
"""
import json
import os
import time
import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
_IMPACT_FILE = os.path.join(_DATA_DIR, "price_impacts.json")


@dataclass
class PriceSnapshot:
    """Price at a moment in time."""
    token_id: str
    price: float
    best_bid: float
    best_ask: float
    timestamp: float


@dataclass
class EventImpact:
    """Measured impact of a game event on market price."""
    game: str
    event_type: str
    team: str
    description: str
    match_id: str
    market_question: str
    # Prices
    price_before: float
    price_after_5s: float = 0
    price_after_15s: float = 0
    price_after_30s: float = 0
    price_after_45s: float = 0
    price_after_60s: float = 0
    price_after_90s: float = 0
    # Calculated
    move_5s: float = 0
    move_15s: float = 0
    move_30s: float = 0
    move_45s: float = 0
    move_60s: float = 0
    move_90s: float = 0
    game_state: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class PriceImpactTracker:
    """Tracks price movements around game events."""

    def __init__(self, ws):
        self.ws = ws  # PolymarketWebSocket instance
        self._pending: list[dict] = []  # events waiting for price measurement
        self._impacts: list[EventImpact] = []  # completed measurements
        self._max_impacts = 500

        # Aggregate stats by event type + game
        self.stats: dict[str, dict] = defaultdict(lambda: {
            "count": 0, "total_move_5s": 0, "total_move_15s": 0, "total_move_30s": 0,
            "max_move": 0, "profitable": 0,
        })
        self._last_save = 0

        # Load persisted data
        self._load()

    def _load(self):
        """Load persisted impacts from disk."""
        try:
            if os.path.exists(_IMPACT_FILE):
                with open(_IMPACT_FILE, "r") as f:
                    data = json.load(f)
                self.stats = defaultdict(lambda: {
                    "count": 0, "total_move_5s": 0, "total_move_15s": 0, "total_move_30s": 0,
                    "max_move": 0, "profitable": 0,
                }, data.get("stats", {}))
                for item in data.get("impacts", []):
                    self._impacts.append(EventImpact(**item))
                logger.info(f"[IMPACT] Loaded {len(self._impacts)} impacts, {len(self.stats)} event types from disk")
        except Exception as e:
            logger.warning(f"[IMPACT] Load failed: {e}")

    def _save(self):
        """Persist impacts to disk (called periodically)."""
        try:
            os.makedirs(_DATA_DIR, exist_ok=True)
            data = {
                "stats": dict(self.stats),
                "impacts": [
                    {
                        "game": i.game, "event_type": i.event_type, "team": i.team,
                        "description": i.description, "match_id": i.match_id,
                        "market_question": i.market_question,
                        "price_before": i.price_before,
                        "price_after_5s": i.price_after_5s,
                        "price_after_15s": i.price_after_15s,
                        "price_after_30s": i.price_after_30s,
                        "price_after_45s": getattr(i, 'price_after_45s', 0),
                        "price_after_60s": getattr(i, 'price_after_60s', 0),
                        "price_after_90s": getattr(i, 'price_after_90s', 0),
                        "move_5s": i.move_5s, "move_15s": i.move_15s, "move_30s": i.move_30s,
                        "move_45s": getattr(i, 'move_45s', 0),
                        "move_60s": getattr(i, 'move_60s', 0),
                        "move_90s": getattr(i, 'move_90s', 0),
                        "timestamp": i.timestamp,
                        "game_state": getattr(i, 'game_state', {}),
                    }
                    for i in self._impacts[-500:]
                ],
            }
            tmp = _IMPACT_FILE + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, _IMPACT_FILE)
        except Exception as e:
            logger.warning(f"[IMPACT] Save failed: {e}")

    def record_event(self, event, market, token_id: str):
        """Record a game event and snapshot the current price."""
        price = self.ws.get_midpoint(token_id)
        if price <= 0.01:
            return

        ws_data = self.ws.get_price(token_id)
        # Capture rich game state for future Claude analysis
        game_state = {}
        if event.match_state:
            ms = event.match_state
            game_state = {
                "team_a": ms.team_a,
                "team_b": ms.team_b,
                "score_a": ms.score_a,
                "score_b": ms.score_b,
                "round_a": ms.round_score_a,
                "round_b": ms.round_score_b,
                "win_prob_a": round(ms.win_probability_a, 3),
            }
            if ms.extra:
                for k in ["kills_a", "kills_b", "gold_lead", "game_minutes",
                           "team_a_money", "team_b_money", "towers_a", "towers_b",
                           "dragons_a", "dragons_b", "barons_a", "barons_b"]:
                    if k in ms.extra:
                        game_state[k] = ms.extra[k]

        self._pending.append({
            "game": event.game,
            "event_type": event.event_type.value,
            "team": event.team,
            "description": event.description,
            "match_id": event.match_id,
            "market_question": market.question[:50] if market else "",
            "token_id": token_id,
            "price_before": price,
            "best_bid": ws_data["best_bid"] if ws_data else 0,
            "best_ask": ws_data["best_ask"] if ws_data else 0,
            "game_state": game_state,
            "timestamp": time.time(),
            "checked_5s": False,
            "checked_15s": False,
            "checked_30s": False,
        })

    def tick(self):
        """Check pending events for price changes. Call every 1-2 seconds."""
        now = time.time()
        completed = []

        for p in self._pending:
            age = now - p["timestamp"]
            current = self.ws.get_midpoint(p["token_id"])
            if current <= 0.01:
                continue

            if age >= 5 and not p["checked_5s"]:
                p["price_5s"] = current
                p["checked_5s"] = True

            if age >= 15 and not p["checked_15s"]:
                p["price_15s"] = current
                p["checked_15s"] = True

            if age >= 30 and not p["checked_30s"]:
                p["price_30s"] = current
                p["checked_30s"] = True

            if age >= 45 and not p.get("checked_45s"):
                p["price_45s"] = current
                p["checked_45s"] = True

            if age >= 60 and not p.get("checked_60s"):
                p["price_60s"] = current
                p["checked_60s"] = True

            if age >= 90 and not p.get("checked_90s"):
                p["price_90s"] = current
                p["checked_90s"] = True

                # Complete at 90s — full measurement window
                impact = EventImpact(
                    game=p["game"],
                    event_type=p["event_type"],
                    team=p["team"],
                    description=p["description"],
                    match_id=p["match_id"],
                    market_question=p["market_question"],
                    price_before=p["price_before"],
                    price_after_5s=p.get("price_5s", 0),
                    price_after_15s=p.get("price_15s", 0),
                    price_after_30s=p.get("price_30s", 0),
                    price_after_45s=p.get("price_45s", 0),
                    price_after_60s=p.get("price_60s", 0),
                    price_after_90s=p.get("price_90s", 0),
                    timestamp=p["timestamp"],
                    game_state=p.get("game_state", {}),
                )
                impact.move_5s = impact.price_after_5s - impact.price_before
                impact.move_15s = impact.price_after_15s - impact.price_before
                impact.move_30s = impact.price_after_30s - impact.price_before
                impact.move_45s = impact.price_after_45s - impact.price_before
                impact.move_60s = impact.price_after_60s - impact.price_before
                impact.move_90s = impact.price_after_90s - impact.price_before

                self._impacts.append(impact)
                if len(self._impacts) > self._max_impacts:
                    self._impacts = self._impacts[-self._max_impacts:]

                # Update aggregate stats
                key = f"{p['game']}_{p['event_type']}"
                s = self.stats[key]
                s["count"] += 1
                s["total_move_5s"] += abs(impact.move_5s)
                s["total_move_15s"] += abs(impact.move_15s)
                s["total_move_30s"] += abs(impact.move_30s)
                s.setdefault("total_move_45s", 0)
                s.setdefault("total_move_60s", 0)
                s.setdefault("total_move_90s", 0)
                s["total_move_45s"] += abs(impact.move_45s)
                s["total_move_60s"] += abs(impact.move_60s)
                s["total_move_90s"] += abs(impact.move_90s)
                s["max_move"] = max(s["max_move"], abs(impact.move_90s))
                if abs(impact.move_90s) > 0.02:
                    s["profitable"] += 1

                move_pct = impact.move_90s / impact.price_before * 100 if impact.price_before > 0 else 0
                if abs(move_pct) > 1:
                    logger.info(
                        f"[IMPACT] {p['game']} {p['event_type']} | "
                        f"{impact.price_before:.3f} → {impact.price_after_90s:.3f} "
                        f"({move_pct:+.1f}% @90s) | 30s={impact.move_30s*100:+.1f}% 60s={impact.move_60s*100:+.1f}% | "
                        f"{p['description'][:40]}"
                    )

                completed.append(p)

            # Expire old pending events (>120s)
            if age > 120 and not p.get("checked_90s"):
                completed.append(p)

        for c in completed:
            self._pending.remove(c)

        # Save to disk every 30s
        if completed and time.time() - self._last_save > 30:
            self._last_save = time.time()
            self._save()

    def get_summary(self) -> dict:
        """Return summary stats for dashboard."""
        summary = {}
        for key, s in sorted(self.stats.items(), key=lambda x: -x[1]["total_move_30s"]):
            if s["count"] == 0:
                continue
            game, etype = key.split("_", 1)
            avg_5 = s["total_move_5s"] / s["count"]
            avg_15 = s["total_move_15s"] / s["count"]
            avg_30 = s["total_move_30s"] / s["count"]
            avg_45 = s.get("total_move_45s", 0) / s["count"]
            avg_60 = s.get("total_move_60s", 0) / s["count"]
            avg_90 = s.get("total_move_90s", 0) / s["count"]
            summary[key] = {
                "game": game, "event_type": etype, "count": s["count"],
                "avg_move_5s": round(avg_5 * 100, 2),
                "avg_move_15s": round(avg_15 * 100, 2),
                "avg_move_30s": round(avg_30 * 100, 2),
                "avg_move_45s": round(avg_45 * 100, 2),
                "avg_move_60s": round(avg_60 * 100, 2),
                "avg_move_90s": round(avg_90 * 100, 2),
                "max_move": round(s["max_move"] * 100, 2),
                "profitable_pct": round(s["profitable"] / s["count"] * 100, 0),
            }
        return summary

    def get_recent_impacts(self, n=20) -> list:
        """Return recent price impacts for dashboard."""
        return [
            {
                "game": i.game, "event_type": i.event_type,
                "description": i.description[:50],
                "price_before": round(i.price_before, 3),
                "move_5s": round(i.move_5s * 100, 2),
                "move_15s": round(i.move_15s * 100, 2),
                "move_30s": round(i.move_30s * 100, 2),
                "move_45s": round(getattr(i, 'move_45s', 0) * 100, 2),
                "move_60s": round(getattr(i, 'move_60s', 0) * 100, 2),
                "move_90s": round(getattr(i, 'move_90s', 0) * 100, 2),
                "time": i.timestamp,
            }
            for i in self._impacts[-n:]
        ]
