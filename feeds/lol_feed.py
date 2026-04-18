"""
League of Legends live match feed.

Data sources (in order of latency):
1. Riot Live Client Data API (localhost:2999) — near-zero latency, but only on the game machine
2. PandaScore WebSocket — ~300ms from broadcast stream (commercial, $2k-10k/month)
3. Riot REST API (spectator endpoint) — 3-5 minute delay for pro matches
4. Community scrapers (lolesports.com) — variable delay

For production latency arb, PandaScore is the recommended source.
For development/testing, we use the Riot REST API + lolesports scraping.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

# Riot API endpoints
LOLESPORTS_LIVE_URL = "https://esports-api.lolesports.com/persisted/gw/getLive"
LOLESPORTS_EVENT_DETAILS = "https://esports-api.lolesports.com/persisted/gw/getEventDetails"


class LoLFeed(GameFeed):
    """
    League of Legends live match feed.

    Currently implements polling-based approach via lolesports API.
    For production, replace with PandaScore WebSocket for ~300ms latency.
    """

    POLL_INTERVAL = 5  # seconds between polls

    def __init__(self):
        super().__init__("lol")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._last_states: dict[str, dict] = {}  # match_id -> last known state

    async def connect(self):
        """Initialize HTTP session for API polling."""
        self._session = aiohttp.ClientSession(
            headers={
                "x-api-key": "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z",  # public lolesports key
                "User-Agent": "Mozilla/5.0",
            }
        )
        self._connected = True
        self._connect_time = time.time()

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="",
            game="lol",
            team="neutral",
            description="Connected to LoL Esports API",
            impact=0.0,
        ))

        # Start polling loop
        self._running = True
        self._poll_task = asyncio.create_task(self._poll_loop())
        logger.info("LoL feed connected (polling mode)")

    async def disconnect(self):
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._session:
            await self._session.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        """Track a specific LoL match."""
        self._matches[match_id] = MatchState(
            match_id=match_id,
            game="lol",
            team_a="",
            team_b="",
            is_live=True,
            started_at=time.time(),
            total_maps=3,  # Bo3 default, Bo5 for finals
        )
        logger.info(f"Subscribed to LoL match {match_id}")

    async def unsubscribe_match(self, match_id: str):
        self._matches.pop(match_id, None)
        self._last_states.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        """Fetch currently live LoL matches from lolesports."""
        if not self._session:
            return []

        try:
            async with self._session.get(
                LOLESPORTS_LIVE_URL,
                params={"hl": "en-US"},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()

            matches = []
            events = data.get("data", {}).get("schedule", {}).get("events", [])

            for event in events:
                if event.get("state") != "inProgress":
                    continue

                match = event.get("match", {})
                teams = match.get("teams", [])
                if len(teams) < 2:
                    continue

                match_id = event.get("id", "")
                strategy = match.get("strategy", {})
                best_of = strategy.get("count", 3)

                state = MatchState(
                    match_id=match_id,
                    game="lol",
                    team_a=teams[0].get("name", "Team A"),
                    team_b=teams[1].get("name", "Team B"),
                    score_a=teams[0].get("result", {}).get("gameWins", 0),
                    score_b=teams[1].get("result", {}).get("gameWins", 0),
                    total_maps=best_of,
                    is_live=True,
                )
                state.win_probability_a = self.estimate_win_probability(state)
                matches.append(state)

            return matches

        except Exception as e:
            logger.error(f"Failed to fetch live LoL matches: {e}")
            return []

    async def _poll_loop(self):
        """Poll for match state changes."""
        while self._running:
            try:
                await self._poll_matches()
                await asyncio.sleep(self.POLL_INTERVAL)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LoL poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _poll_matches(self):
        """Check for state changes in tracked matches."""
        live_matches = await self.get_live_matches()

        for live in live_matches:
            match_id = live.match_id

            # Auto-subscribe to new live matches
            if match_id not in self._matches:
                self._matches[match_id] = live
                self._emit(GameEvent(
                    event_type=EventType.MATCH_START,
                    match_id=match_id,
                    game="lol",
                    team="neutral",
                    description=f"{live.team_a} vs {live.team_b} is LIVE",
                    impact=0.0,
                    match_state=live,
                ))
                continue

            # Check for score changes
            old_state = self._matches[match_id]
            if live.score_a != old_state.score_a or live.score_b != old_state.score_b:
                now = time.time()

                # Determine who won the game
                if live.score_a > old_state.score_a:
                    winner = "a"
                else:
                    winner = "b"

                old_prob = old_state.win_probability_a
                live.win_probability_a = self.estimate_win_probability(live)
                impact = live.win_probability_a - old_prob

                self._emit(GameEvent(
                    event_type=EventType.MAP_WIN,
                    match_id=match_id,
                    game="lol",
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

    def estimate_win_probability(self, state: MatchState) -> float:
        """
        LoL-specific win probability model.
        LoL series are typically Bo3 or Bo5.
        Gold leads, dragon/baron counts could improve this with richer data.
        """
        return super().estimate_win_probability(state)
