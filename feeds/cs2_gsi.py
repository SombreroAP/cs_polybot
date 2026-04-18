"""
CS2 Game State Integration (GSI) feed.

Receives real-time game state from CS2 client via HTTP POST.
Requires:
1. GSI config file in CS2's cfg folder
2. CS2 running and spectating a match via GOTV

This gives us the FASTEST possible CS2 data — sub-second latency,
same data that production esports overlays use.

Data: round scores, player economy, bomb status, kills, map name,
player health/armor/weapons — everything.
"""
import asyncio
import json
import logging
import time
from typing import Optional
from aiohttp import web

from feeds.base import GameFeed, GameEvent, MatchState, EventType
from feeds.cs2_model import CS2WinProbabilityModel, classify_buy

logger = logging.getLogger(__name__)

GSI_PORT = 3001  # Local port CS2 sends data to


class CS2GSIFeed(GameFeed):
    """Real-time CS2 data from Game State Integration (spectating via GOTV)."""

    def __init__(self):
        super().__init__("cs2")
        self._model = CS2WinProbabilityModel()
        self._server: Optional[web.AppRunner] = None
        self._current_match_id = "gsi_live"
        self._last_state: dict = {}
        self._initialized_match = False

    async def connect(self):
        """Start HTTP server to receive GSI data from CS2."""
        app = web.Application()
        app.router.add_post("/", self._handle_gsi)

        self._server = web.AppRunner(app)
        await self._server.setup()
        site = web.TCPSite(self._server, "127.0.0.1", GSI_PORT)
        await site.start()

        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="cs2", team="neutral",
            description=f"CS2 GSI listening on port {GSI_PORT} — spectate a match in CS2",
            impact=0.0,
        ))
        logger.info(f"CS2 GSI feed listening on http://127.0.0.1:{GSI_PORT}")
        logger.info("Open CS2 → Watch → select a live match to start receiving data")

    async def disconnect(self):
        self._running = False
        if self._server:
            await self._server.cleanup()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        pass

    async def unsubscribe_match(self, match_id: str):
        pass

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._matches.keys())

    def estimate_win_probability(self, state: MatchState) -> float:
        raw = self._model.calculate(
            maps_a=state.score_a,
            maps_b=state.score_b,
            best_of=state.total_maps,
            round_score_a=state.round_score_a,
            round_score_b=state.round_score_b,
            team_a_side=state.extra.get("team_a_current_side", "ct"),
            team_a_start_side=state.extra.get("team_a_start_side", None),
            team_a_money=state.extra.get("team_a_money", 0),
            team_b_money=state.extra.get("team_b_money", 0),
        )
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    # ─── GSI HTTP Handler ────────────────────────────────────────────────────

    async def _handle_gsi(self, request: web.Request) -> web.Response:
        """Handle incoming GSI POST from CS2 client."""
        try:
            data = await request.json()
            self._process_gsi(data)
        except Exception as e:
            logger.debug(f"GSI parse error: {e}")
        return web.Response(text="OK")

    def _process_gsi(self, data: dict):
        """Process a GSI state update from CS2."""
        now = time.time()

        # Extract key data
        map_data = data.get("map", {})
        if not map_data:
            return  # not in a match

        phase = map_data.get("phase", "")
        if phase not in ("live", "intermission", "freezetime", "over"):
            return

        # Team scores
        ct = map_data.get("team_ct", {})
        t = map_data.get("team_t", {})
        ct_score = ct.get("score", 0)
        t_score = t.get("score", 0)
        ct_name = ct.get("name", "CT")
        t_name = t.get("name", "T")
        map_name = map_data.get("name", "unknown")

        # Determine team_a = CT for now (will be corrected)
        team_a = ct_name
        team_b = t_name
        score_a = ct_score
        score_b = t_score

        # Player economy
        all_players = data.get("allplayers", {})
        ct_money = 0
        t_money = 0
        ct_kills = 0
        t_kills = 0
        for pid, player in all_players.items():
            team = player.get("team", "")
            money = player.get("state", {}).get("money", 0)
            kills = player.get("match_stats", {}).get("kills", 0)
            if team == "CT":
                ct_money += money
                ct_kills += kills
            elif team == "T":
                t_money += money
                t_kills += kills

        # Create/update match state
        mid = self._current_match_id
        if mid not in self._matches:
            state = MatchState(
                match_id=mid, game="cs2",
                team_a=team_a, team_b=team_b,
                is_live=True, started_at=now,
                total_maps=3,
                extra={
                    "team_a_current_side": "ct",
                    "team_a_money": ct_money,
                    "team_b_money": t_money,
                    "round_kills_a": 0,
                    "round_kills_b": 0,
                    "map_name": map_name,
                },
            )
            self._matches[mid] = state
            logger.info(f"[GSI] Match started: {team_a} vs {team_b} on {map_name}")

            self._emit(GameEvent(
                event_type=EventType.MATCH_START,
                match_id=mid, game="cs2", team="neutral",
                description=f"{team_a} vs {team_b} on {map_name} (GSI)",
                impact=0.0, match_state=state,
            ))

        state = self._matches[mid]
        state.team_a = team_a
        state.team_b = team_b
        state.extra["team_a_money"] = ct_money
        state.extra["team_b_money"] = t_money
        state.extra["team_a_current_side"] = "ct"
        state.extra["map_name"] = map_name

        # Detect round score changes
        old_a = state.round_score_a
        old_b = state.round_score_b

        if score_a != old_a or score_b != old_b:
            state.round_score_a = score_a
            state.round_score_b = score_b

            if old_a + old_b > 0:  # skip first update
                winning_team = "a" if score_a > old_a else "b"
                winner_name = team_a if winning_team == "a" else team_b

                old_prob = state.win_probability_a
                state.win_probability_a = self.estimate_win_probability(state)
                impact = state.win_probability_a - old_prob

                logger.info(f"[GSI ROUND] {team_a} {score_a}-{score_b} {team_b} | {map_name}")

                self._emit(GameEvent(
                    event_type=EventType.ROUND_END,
                    match_id=mid, game="cs2", team=winning_team,
                    description=f"Round to {winner_name} ({score_a}-{score_b}) on {map_name}",
                    impact=impact, timestamp=now, match_state=state,
                ))

                # Economy event
                self._emit(GameEvent(
                    event_type=EventType.ECONOMY_SHIFT,
                    match_id=mid, game="cs2", team=winning_team,
                    description=f"Eco: CT=${ct_money:,} ({classify_buy(ct_money)}) vs T=${t_money:,} ({classify_buy(t_money)})",
                    impact=0.02 if winning_team == "a" else -0.02,
                    timestamp=now, match_state=state,
                ))

        # Bomb events
        bomb = data.get("round", {}).get("bomb", "")
        prev_bomb = self._last_state.get("bomb", "")
        if bomb == "planted" and prev_bomb != "planted":
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=mid, game="cs2", team="b",  # T side plants
                description="Bomb planted",
                impact=-0.05, timestamp=now, match_state=state,
            ))
        elif bomb == "defused" and prev_bomb != "defused":
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=mid, game="cs2", team="a",  # CT defuses
                description="Bomb defused",
                impact=0.05, timestamp=now, match_state=state,
            ))

        # Match end
        if phase == "over" and state.is_live:
            state.is_live = False
            winner = "a" if score_a > score_b else "b"
            winner_name = team_a if winner == "a" else team_b
            self._emit(GameEvent(
                event_type=EventType.MAP_WIN,
                match_id=mid, game="cs2", team=winner,
                description=f"Map to {winner_name} ({score_a}-{score_b})",
                impact=0.3 if winner == "a" else -0.3,
                timestamp=now, match_state=state,
            ))

        state.win_probability_a = self.estimate_win_probability(state)
        state.last_event_time = now
        self._last_state = {"bomb": bomb, "score_a": score_a, "score_b": score_b}
