"""
League of Legends live match feed via lolesports API.

Uses the unofficial but public lolesports.com API with known API key.
Provides game-level scores (wins in a series), updated in near real-time.

Coverage: LCK, LPL, LEC, LCS, CBLOL, PCS, and other regional leagues.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

LOLESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"
LOLESPORTS_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"


class LoLLolesportsFeed(GameFeed):
    """League of Legends live match data from lolesports API."""

    POLL_INTERVAL = 15  # seconds — lolesports updates game-level, not tick-level

    def __init__(self):
        super().__init__("lol")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        self._subscribed_matches: set[str] = set()
        self._initialized: set[str] = set()

    async def connect(self):
        self._session = aiohttp.ClientSession(headers={
            "x-api-key": LOLESPORTS_KEY,
            "User-Agent": "Mozilla/5.0",
        })
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="lol", team="neutral",
            description="Connected to LoL Esports feed", impact=0.0,
        ))
        logger.info("LoL lolesports feed connected")
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "lol_poll"))

    async def _safe_run(self, coro, name: str):
        try:
            await coro
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"Task {name} CRASHED: {e}", exc_info=True)

    async def disconnect(self):
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()
        if self._session:
            await self._session.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._subscribed_matches.add(match_id)
        if match_id not in self._matches:
            self._matches[match_id] = MatchState(
                match_id=match_id, game="lol",
                team_a="", team_b="",
                is_live=True, started_at=time.time(),
                extra={},
            )

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._subscribed_matches)

    def estimate_win_probability(self, state: MatchState) -> float:
        """LoL uses Bo1/Bo3/Bo5 — same series model as other games."""
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01

        # Simple ratio model (same as base class)
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        prob = b_needs / (a_needs + b_needs)
        raw = max(0.01, min(0.99, prob))
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    async def _poll_loop(self):
        """Poll lolesports for live match state changes."""
        logger.info("LoL poll loop started")
        while self._running:
            try:
                events = await self._fetch_live()
                if events is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                live_ids = set()
                for event in events:
                    if event.get("state") != "inProgress":
                        continue

                    match = event.get("match", {})
                    teams = match.get("teams", [])
                    if len(teams) < 2:
                        continue

                    eid = str(event.get("id", ""))
                    live_ids.add(eid)
                    t1_name = teams[0].get("name", "Team 1")
                    t2_name = teams[1].get("name", "Team 2")
                    s1 = teams[0].get("result", {}).get("gameWins", 0)
                    s2 = teams[1].get("result", {}).get("gameWins", 0)
                    strategy = match.get("strategy", {})
                    best_of = strategy.get("count", 1)
                    league = event.get("league", {}).get("name", "")

                    # Auto-subscribe
                    if eid not in self._subscribed_matches:
                        await self.subscribe_match(eid)
                        state = self._matches[eid]
                        state.team_a = t1_name
                        state.team_b = t2_name
                        state.total_maps = best_of
                        state.score_a = s1
                        state.score_b = s2

                        logger.info(f"Subscribed: {t1_name} vs {t2_name} (Bo{best_of}) [{league}]")
                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=eid, game="lol", team="neutral",
                            description=f"{t1_name} vs {t2_name} (Bo{best_of}) LIVE [{league}]",
                            impact=0.0, match_state=state,
                        ))
                        self._initialized.add(eid)
                        continue

                    # Check for game score changes
                    state = self._matches.get(eid)
                    if not state:
                        continue

                    old_a = state.score_a
                    old_b = state.score_b

                    if s1 != old_a or s2 != old_b:
                        winner = "a" if s1 > old_a else "b"
                        state.score_a = s1
                        state.score_b = s2

                        old_prob = state.win_probability_a
                        state.win_probability_a = self.estimate_win_probability(state)
                        impact = state.win_probability_a - old_prob

                        winner_name = state.team_a if winner == "a" else state.team_b
                        logger.info(f"[GAME] {winner_name} wins! {t1_name} {s1}-{s2} {t2_name}")

                        self._emit(GameEvent(
                            event_type=EventType.MAP_WIN,
                            match_id=eid, game="lol", team=winner,
                            description=f"Game to {winner_name} ({s1}-{s2})",
                            impact=impact, match_state=state,
                        ))

                        # Check match end
                        maps_needed = (state.total_maps // 2) + 1
                        if s1 >= maps_needed or s2 >= maps_needed:
                            state.is_live = False
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=eid, game="lol", team=winner,
                                description=f"Match won by {winner_name} ({s1}-{s2})",
                                impact=1.0 if winner == "a" else -1.0,
                                match_state=state,
                            ))

                    state.last_event_time = time.time()

                # Mark ended matches
                for mid in list(self._subscribed_matches):
                    if mid not in live_ids and mid in self._matches:
                        self._matches[mid].is_live = False

                live_count = sum(1 for m in self._matches.values() if m.is_live)
                logger.info(f"LoL: {len(live_ids)} live, {live_count} tracked")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"LoL poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _fetch_live(self) -> Optional[list]:
        try:
            async with self._session.get(
                f"{LOLESPORTS_API}/getLive", params={"hl": "en-US"},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json()
                return data.get("data", {}).get("schedule", {}).get("events", [])
        except Exception as e:
            logger.error(f"lolesports fetch failed: {e}")
            return None
