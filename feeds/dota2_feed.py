"""
Dota 2 live match feed.

Data sources (in order of latency):
1. Game State Integration (GSI) — near-zero, local machine only
2. Steam Game Coordinator (GC) — real-time, complex Steam protocol
3. PandaScore WebSocket — ~300ms from broadcast
4. OpenDota / STRATZ API — delayed, post-game
5. Dota2 LiveLeagueGames Steam API — variable, few seconds delay

For production: PandaScore or direct Steam GC integration.
For development: Steam Web API LiveLeagueGames endpoint.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

STEAM_API_URL = "https://api.steampowered.com"


class Dota2Feed(GameFeed):
    """
    Dota 2 live match feed.

    Uses Steam Web API for match discovery and state polling.
    For production latency arb, integrate PandaScore or Steam GC.
    """

    POLL_INTERVAL = 10  # Steam API is rate-limited

    def __init__(self):
        super().__init__("dota2")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._steam_api_key = config.PANDASCORE_API_KEY  # reuse or set STEAM_API_KEY

    async def connect(self):
        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="",
            game="dota2",
            team="neutral",
            description="Connected to Dota 2 feed",
            impact=0.0,
        ))
        logger.info("Dota 2 feed connected (polling mode)")

    async def disconnect(self):
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._session:
            await self._session.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._matches[match_id] = MatchState(
            match_id=match_id,
            game="dota2",
            team_a="",
            team_b="",
            is_live=True,
            started_at=time.time(),
            total_maps=3,
        )

    async def unsubscribe_match(self, match_id: str):
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        """
        Fetch live Dota 2 pro matches.
        Uses DOTA_GetLiveLeagueGames Steam Web API or PandaScore.
        """
        if not self._session:
            return []

        matches = []

        # Try PandaScore first (if API key available)
        if config.PANDASCORE_API_KEY:
            matches = await self._fetch_pandascore_live()

        # Fallback: Steam API (requires Steam API key)
        if not matches:
            matches = await self._fetch_steam_live()

        return matches

    async def _fetch_pandascore_live(self) -> list[MatchState]:
        """Fetch live Dota 2 matches from PandaScore."""
        try:
            async with self._session.get(
                f"{config.PANDASCORE_BASE_URL}/dota2/matches/running",
                headers={"Authorization": f"Bearer {config.PANDASCORE_API_KEY}"},
                params={"per_page": 20},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            matches = []
            for match in data:
                opponents = match.get("opponents", [])
                if len(opponents) < 2:
                    continue

                team_a = opponents[0].get("opponent", {}).get("name", "Team A")
                team_b = opponents[1].get("opponent", {}).get("name", "Team B")
                results = match.get("results", [])

                score_a = results[0].get("score", 0) if results else 0
                score_b = results[1].get("score", 0) if len(results) > 1 else 0
                best_of = match.get("number_of_games", 3)

                state = MatchState(
                    match_id=str(match.get("id", "")),
                    game="dota2",
                    team_a=team_a,
                    team_b=team_b,
                    score_a=score_a,
                    score_b=score_b,
                    total_maps=best_of,
                    is_live=True,
                )
                state.win_probability_a = self.estimate_win_probability(state)
                matches.append(state)

            return matches

        except Exception as e:
            logger.error(f"PandaScore Dota 2 fetch failed: {e}")
            return []

    async def _fetch_steam_live(self) -> list[MatchState]:
        """Fetch live Dota 2 league games from Steam API."""
        # Note: Requires STEAM_API_KEY
        # This endpoint returns basic match info but limited real-time state
        return []

    async def _poll_loop(self):
        while self._running:
            try:
                await self._poll_matches()
                await asyncio.sleep(self.POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dota 2 poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _poll_matches(self):
        """Check for state changes in tracked Dota 2 matches."""
        live_matches = await self.get_live_matches()

        for live in live_matches:
            match_id = live.match_id

            if match_id not in self._matches:
                self._matches[match_id] = live
                self._emit(GameEvent(
                    event_type=EventType.MATCH_START,
                    match_id=match_id,
                    game="dota2",
                    team="neutral",
                    description=f"{live.team_a} vs {live.team_b} is LIVE",
                    impact=0.0,
                    match_state=live,
                ))
                continue

            old_state = self._matches[match_id]
            if live.score_a != old_state.score_a or live.score_b != old_state.score_b:
                now = time.time()
                winner = "a" if live.score_a > old_state.score_a else "b"

                old_prob = old_state.win_probability_a
                live.win_probability_a = self.estimate_win_probability(live)
                impact = live.win_probability_a - old_prob

                self._emit(GameEvent(
                    event_type=EventType.MAP_WIN,
                    match_id=match_id,
                    game="dota2",
                    team=winner,
                    description=(
                        f"Game won by {live.team_a if winner == 'a' else live.team_b} "
                        f"(Series: {live.score_a}-{live.score_b})"
                    ),
                    impact=impact,
                    timestamp=now,
                    match_state=live,
                ))

                self._matches[match_id] = live
