"""
Match Recorder — records ALL live data in JSONL format for backtesting.

Captures the same format as Test Data/ files:
- Game events (rounds, kills, economy, map wins)
- Polymarket orderbook (bid/ask/last_trade)
- Match metadata

Output: data/recordings/{match_id}_{team1}_vs_{team2}_{date}.jsonl
"""
import json
import logging
import os
import time
from datetime import datetime, timezone
from threading import Lock

logger = logging.getLogger(__name__)

_RECORDINGS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "recordings")


class MatchRecorder:
    """Records live match data for future backtesting."""

    def __init__(self):
        os.makedirs(_RECORDINGS_DIR, exist_ok=True)
        self._files: dict[str, object] = {}  # match_id -> open file handle
        self._meta_written: set[str] = set()
        self._lock = Lock()
        self._event_count = 0
        self._matches_recorded = set()
        logger.info(f"[RECORDER] Recording to {_RECORDINGS_DIR}")

    def _get_file(self, match_id: str, team_a: str = "", team_b: str = ""):
        """Get or create the JSONL file for a match."""
        if match_id in self._files:
            return self._files[match_id]

        # Clean team names for filename
        clean = lambda s: "".join(c if c.isalnum() or c in "._- " else "" for c in s).strip().replace(" ", "_")
        ta = clean(team_a)[:20] if team_a else "unknown"
        tb = clean(team_b)[:20] if team_b else "unknown"
        date = datetime.now().strftime("%Y-%m-%d")
        filename = f"{match_id}_{ta}_vs_{tb}_{date}.jsonl"
        filepath = os.path.join(_RECORDINGS_DIR, filename)

        f = open(filepath, "a", buffering=1)  # line-buffered
        self._files[match_id] = f
        self._matches_recorded.add(match_id)
        logger.info(f"[RECORDER] Recording match {match_id}: {team_a} vs {team_b} → {filename}")
        return f

    def _write(self, match_id: str, message_type: str, raw: dict,
               team_a: str = "", team_b: str = ""):
        """Write a single event line to the JSONL file."""
        now = time.time()
        record = {
            "ts": now,
            "ts_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "match_id": match_id,
            "message_type": message_type,
            "raw": raw,
        }
        with self._lock:
            try:
                f = self._get_file(match_id, team_a, team_b)
                f.write(json.dumps(record) + "\n")
                self._event_count += 1
                # Batch flush every 100 events
                if self._event_count % 100 == 0:
                    for fh in self._files.values():
                        try: fh.flush()
                        except Exception: pass
            except Exception as e:
                logger.error(f"[RECORDER] Write error: {e}")

    def record_meta(self, match_id: str, team_a: str, team_b: str,
                    game: str, market_slug: str = "", token_id_a: str = "",
                    token_id_b: str = "", bo_type: int = 3):
        """Record match metadata (once per match)."""
        if match_id in self._meta_written:
            return
        self._meta_written.add(match_id)
        self._write(match_id, "_META", {
            "match_id": match_id,
            "team1": team_a,
            "team2": team_b,
            "game": game,
            "bo_type": bo_type,
            "pm_slug": market_slug,
            "token_id_a": token_id_a,
            "token_id_b": token_id_b,
            "start_date": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
        }, team_a, team_b)

    def record_orderbook(self, match_id: str, token_id: str,
                         best_bid: float, best_ask: float,
                         team_a: str = "", team_b: str = ""):
        """Record Polymarket orderbook update."""
        self._write(match_id, "_PM_book", {
            "tokenId": token_id,
            "matchId": match_id,
            "bestBid": best_bid,
            "bestAsk": best_ask,
            "asOf": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
        }, team_a, team_b)

    def record_best_bid_ask(self, match_id: str, token_id: str,
                            best_bid: float, best_ask: float,
                            team_a: str = "", team_b: str = ""):
        """Record best bid/ask snapshot."""
        self._write(match_id, "_PM_best_bid_ask", {
            "tokenId": token_id,
            "matchId": match_id,
            "bestBid": best_bid,
            "bestAsk": best_ask,
            "asOf": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
        }, team_a, team_b)

    def record_last_trade(self, match_id: str, token_id: str, price: float,
                          team_a: str = "", team_b: str = ""):
        """Record last trade price."""
        self._write(match_id, "_PM_last_trade_price", {
            "tokenId": token_id,
            "matchId": match_id,
            "price": price,
            "asOf": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
        }, team_a, team_b)

    def record_game_event(self, match_id: str, event_type: str, raw: dict,
                          team_a: str = "", team_b: str = ""):
        """Record a game event — only REAL game events, not heartbeats."""
        # Skip neutral score_updates (PandaScore heartbeats with no real data)
        if event_type == "score_update" and raw.get("team", "") == "neutral":
            return
        # Skip economy_shift (fires alongside round_end, redundant)
        if event_type == "economy_shift":
            return

        # Map our event types to the GRID-style types used in test data
        type_map = {
            "round_end": "GAME_EVENT_MATCH_END_ROUND",
            "kill_streak": "GAME_EVENT_PLAYER_KILL",
            "teamfight_won": "GAME_EVENT_PLAYER_KILL",
            "map_win": "GAME_EVENT_MATCH_END_ROUND",
            "match_end": "GAME_EVENT_MATCH_END_ROUND",
            "score_update": "SNAPSHOT_MATCH_UPDATE",
            "objective_taken": "GAME_EVENT_PLAYER_KILL",
        }
        msg_type = type_map.get(event_type, "SNAPSHOT_MATCH_UPDATE")
        self._write(match_id, msg_type, raw, team_a, team_b)

    def record_snapshot(self, match_id: str, match_state, market=None,
                        team_a: str = "", team_b: str = ""):
        """Record a full match state snapshot with market data."""
        ms = match_state
        extra = ms.extra or {} if ms else {}

        raw = {
            "message_type": "SNAPSHOT_MATCH_UPDATE",
            "match_id": match_id,
            "map_name": extra.get("map_name", ""),
            "game_number": ms.current_map if ms else 1,
            "round_number": (ms.round_score_a or 0) + (ms.round_score_b or 0) if ms else 0,
            "game_ended": False,
            "team_one": {
                "name": ms.team_a if ms else team_a,
                "side": extra.get("team_a_current_side", ""),
                "score": ms.round_score_a if ms else 0,
                "match_score": ms.score_a if ms else 0,
                "equipment_value": extra.get("team_a_money", 0),
                "players_alive": 5,
            },
            "team_two": {
                "name": ms.team_b if ms else team_b,
                "side": "",
                "score": ms.round_score_b if ms else 0,
                "match_score": ms.score_b if ms else 0,
                "equipment_value": extra.get("team_b_money", 0),
                "players_alive": 5,
            },
        }

        # Add Dota2/LoL specific data
        if ms and ms.game in ("dota2", "lol"):
            raw["team_one"]["kills"] = extra.get("kills_a", 0)
            raw["team_two"]["kills"] = extra.get("kills_b", 0)
            raw["gold_lead"] = extra.get("gold_lead", 0)
            raw["game_minutes"] = extra.get("game_minutes", 0)

        self._write(match_id, "SNAPSHOT_MATCH_UPDATE", raw, team_a, team_b)

    def record_price_snapshot(self, match_id: str, market, ws_data_a: dict = None,
                              ws_data_b: dict = None, team_a: str = "", team_b: str = ""):
        """Record combined price snapshot (market + orderbook)."""
        raw = {
            "matchId": match_id,
            "market_id": getattr(market, 'market_id', ''),
            "price_a": market.price_a if market else 0,
            "price_b": market.price_b if market else 0,
            "best_bid_a": ws_data_a.get("best_bid", 0) if ws_data_a else 0,
            "best_ask_a": ws_data_a.get("best_ask", 0) if ws_data_a else 0,
            "last_trade_a": ws_data_a.get("last_trade", 0) if ws_data_a else 0,
            "best_bid_b": ws_data_b.get("best_bid", 0) if ws_data_b else 0,
            "best_ask_b": ws_data_b.get("best_ask", 0) if ws_data_b else 0,
            "last_trade_b": ws_data_b.get("last_trade", 0) if ws_data_b else 0,
            "volume": market.volume if market else 0,
            "asOf": datetime.fromtimestamp(time.time(), tz=timezone.utc).isoformat(),
        }
        self._write(match_id, "_PRICE_SNAPSHOT", raw, team_a, team_b)

    def get_stats(self) -> dict:
        return {
            "recordings_active": len(self._files),
            "matches_recorded": len(self._matches_recorded),
            "total_events_recorded": self._event_count,
        }

    def close(self):
        """Close all open file handles."""
        with self._lock:
            for f in self._files.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._files.clear()
        logger.info(f"[RECORDER] Closed — {self._event_count} events recorded across {len(self._matches_recorded)} matches")
