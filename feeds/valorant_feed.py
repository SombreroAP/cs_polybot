"""
Valorant live match feed.

Data sources:
1. VLR.gg scraping — community site, variable latency (5-30s)
2. PandaScore — ~300ms from broadcast (commercial)
3. Riot Valorant esports API — limited, similar to LoL esports API

No official real-time API exists for Valorant esports.
VLR.gg + PandaScore are the main sources.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

# Community VLR.gg API (unofficial)
VLRGG_API_URL = "https://vlrggapi.vercel.app/api/v1"


class ValorantFeed(GameFeed):
    """
    Valorant live match feed.

    Uses VLR.gg community API for match discovery.
    For production latency arb, integrate PandaScore.
    """

    POLL_INTERVAL = 10

    def __init__(self):
        super().__init__("valorant")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None

    async def connect(self):
        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="",
            game="valorant",
            team="neutral",
            description="Connected to Valorant feed (VLR.gg)",
            impact=0.0,
        ))
        logger.info("Valorant feed connected (polling mode)")

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
            game="valorant",
            team_a="",
            team_b="",
            is_live=True,
            started_at=time.time(),
            total_maps=3,
        )

    async def unsubscribe_match(self, match_id: str):
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        """Fetch live Valorant matches from VLR.gg API."""
        if not self._session:
            return []

        # Try PandaScore first
        if config.PANDASCORE_API_KEY:
            matches = await self._fetch_pandascore_live()
            if matches:
                return matches

        # Fallback to VLR.gg
        return await self._fetch_vlrgg_live()

    async def _fetch_pandascore_live(self) -> list[MatchState]:
        """Fetch live Valorant matches from PandaScore."""
        try:
            async with self._session.get(
                f"{config.PANDASCORE_BASE_URL}/valorant/matches/running",
                headers={"Authorization": f"Bearer {config.PANDASCORE_API_KEY}"},
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

                state = MatchState(
                    match_id=str(match.get("id", "")),
                    game="valorant",
                    team_a=team_a,
                    team_b=team_b,
                    score_a=results[0].get("score", 0) if results else 0,
                    score_b=results[1].get("score", 0) if len(results) > 1 else 0,
                    total_maps=match.get("number_of_games", 3),
                    is_live=True,
                )
                state.win_probability_a = self.estimate_win_probability(state)
                matches.append(state)

            return matches

        except Exception as e:
            logger.error(f"PandaScore Valorant fetch failed: {e}")
            return []

    async def _fetch_vlrgg_live(self) -> list[MatchState]:
        """Fetch live Valorant matches from VLR.gg community API."""
        try:
            async with self._session.get(
                f"{VLRGG_API_URL}/match/results",
                params={"page": "1"},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            # VLR.gg API structure varies — parse what we can
            matches = []
            for item in data.get("data", {}).get("segments", []):
                if item.get("match_status", "").lower() == "live":
                    state = MatchState(
                        match_id=item.get("match_page", "").split("/")[-1] if item.get("match_page") else "",
                        game="valorant",
                        team_a=item.get("team1", "Team A"),
                        team_b=item.get("team2", "Team B"),
                        score_a=int(item.get("score1", 0) or 0),
                        score_b=int(item.get("score2", 0) or 0),
                        is_live=True,
                    )
                    state.win_probability_a = self.estimate_win_probability(state)
                    matches.append(state)

            return matches

        except Exception as e:
            logger.error(f"VLR.gg fetch failed: {e}")
            return []

    async def _poll_loop(self):
        while self._running:
            try:
                await self._poll_matches()
                await asyncio.sleep(self.POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Valorant poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _poll_matches(self):
        """Check for state changes in tracked Valorant matches."""
        live_matches = await self.get_live_matches()

        for live in live_matches:
            match_id = live.match_id
            if not match_id:
                continue

            if match_id not in self._matches:
                self._matches[match_id] = live
                self._emit(GameEvent(
                    event_type=EventType.MATCH_START,
                    match_id=match_id,
                    game="valorant",
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
                    game="valorant",
                    team=winner,
                    description=(
                        f"Map won by {live.team_a if winner == 'a' else live.team_b} "
                        f"(Series: {live.score_a}-{live.score_b})"
                    ),
                    impact=impact,
                    timestamp=now,
                    match_state=live,
                ))

                self._matches[match_id] = live
