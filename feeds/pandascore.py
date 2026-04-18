"""
PandaScore multi-game feed (free tier).

Provides reliable match status and map scores for ALL esports games.
Free tier: 1000 req/hr — schedules, results, scores, rosters, streams.

This fills the gap for LoL and Valorant where our other feeds lack data.
Also provides backup data for CS2 and Dota2.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

# PandaScore game slugs → our game names
GAME_MAP = {
    "cs-go": "cs2",
    "dota-2": "dota2",
    "lol": "lol",
    "valorant": "valorant",
}


class PandaScoreFeed(GameFeed):
    """Multi-game feed from PandaScore free tier — map scores + rosters + streams."""

    POLL_INTERVAL = 8  # seconds — faster for LoL/Val event detection

    def __init__(self, game: str):
        super().__init__(game)
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        self._subscribed: set[str] = set()
        self._initialized: set[str] = set()
        self._last_scores: dict[str, tuple] = {}  # match_id -> (score_a, score_b)

    async def connect(self):
        if not config.PANDASCORE_TOKEN:
            logger.warning("PANDASCORE_TOKEN not set — PandaScore feed disabled")
            return

        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game=self.game, team="neutral",
            description=f"PandaScore {self.game} feed connected", impact=0.0,
        ))
        logger.info(f"PandaScore {self.game} feed connected")
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), f"panda_{self.game}"))

    async def _safe_run(self, coro, name):
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
        self._subscribed.add(match_id)

    async def unsubscribe_match(self, match_id: str):
        self._subscribed.discard(match_id)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._subscribed)

    def estimate_win_probability(self, state: MatchState) -> float:
        """Series score based probability."""
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        raw = max(0.01, min(0.99, b_needs / (a_needs + b_needs)))
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    # ─── Polling ─────────────────────────────────────────────────────────────

    async def _poll_loop(self):
        """Poll PandaScore for running matches."""
        logger.info(f"PandaScore {self.game} poll loop started")

        # Map our game name to PandaScore slug
        ps_slug = {v: k for k, v in GAME_MAP.items()}.get(self.game, self.game)

        poll_count = 0
        while self._running:
            try:
                poll_count += 1
                matches = await self._fetch_running(ps_slug)
                if matches is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                live_ids = set()
                for m in matches:
                    mid = f"ps_{m.get('id', '')}"
                    live_ids.add(mid)

                    opponents = m.get("opponents", [])
                    if len(opponents) < 2:
                        continue

                    t1 = opponents[0].get("opponent", {}).get("name", "?")
                    t2 = opponents[1].get("opponent", {}).get("name", "?")
                    results = m.get("results", [])
                    score_a = results[0].get("score", 0) if results else 0
                    score_b = results[1].get("score", 0) if len(results) > 1 else 0
                    bo = m.get("number_of_games", 3) or 3

                    # Stream URL
                    streams = m.get("streams_list", [])
                    stream_url = ""
                    for s in streams:
                        raw = s.get("raw_url", "")
                        if raw and ("twitch" in raw or "youtube" in raw or "kick.com" in raw):
                            stream_url = raw
                            break

                    # New match
                    if mid not in self._subscribed:
                        await self.subscribe_match(mid)
                        state = MatchState(
                            match_id=mid, game=self.game,
                            team_a=t1, team_b=t2,
                            score_a=score_a, score_b=score_b,
                            is_live=True, started_at=time.time(),
                            total_maps=bo,
                            extra={},
                        )
                        if stream_url:
                            state.extra["stream_url"] = stream_url
                        self._matches[mid] = state
                        self._last_scores[mid] = (score_a, score_b)

                        logger.info(f"[PANDA] {self.game}: {t1} vs {t2} (Bo{bo}) | {score_a}-{score_b}")

                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=mid, game=self.game, team="neutral",
                            description=f"{t1} vs {t2} (Bo{bo}) LIVE",
                            impact=0.0, match_state=state,
                        ))
                        continue

                    # Update existing
                    state = self._matches.get(mid)
                    if not state:
                        continue

                    # Update stream if we didn't have one
                    if stream_url and not state.extra.get("stream_url"):
                        state.extra["stream_url"] = stream_url

                    # Update scores and recalc probability every poll
                    state.score_a = score_a
                    state.score_b = score_b
                    state.win_probability_a = self.estimate_win_probability(state)

                    # Emit periodic SCORE_UPDATE so latency analyzer can call Claude
                    # even when map score hasn't changed (e.g. mid-map state)
                    last_heartbeat = state.extra.get("_last_heartbeat", 0)
                    if time.time() - last_heartbeat >= 60:
                        state.extra["_last_heartbeat"] = time.time()
                        self._emit(GameEvent(
                            event_type=EventType.SCORE_UPDATE,
                            match_id=mid, game=self.game, team="neutral",
                            description=f"Series: {t1} {score_a}-{score_b} {t2}",
                            impact=0.0, match_state=state,
                        ))

                    old_a, old_b = self._last_scores.get(mid, (0, 0))
                    if score_a != old_a or score_b != old_b:
                        state.score_a = score_a
                        state.score_b = score_b
                        self._last_scores[mid] = (score_a, score_b)

                        winner = "a" if score_a > old_a else "b"
                        winner_name = t1 if winner == "a" else t2

                        old_prob = state.win_probability_a
                        state.win_probability_a = self.estimate_win_probability(state)
                        impact = state.win_probability_a - old_prob

                        logger.info(f"[PANDA MAP] {self.game}: {t1} {score_a}-{score_b} {t2}")

                        self._emit(GameEvent(
                            event_type=EventType.MAP_WIN,
                            match_id=mid, game=self.game, team=winner,
                            description=f"Map to {winner_name} (Series: {score_a}-{score_b})",
                            impact=impact, match_state=state,
                        ))

                        # Check series end
                        maps_needed = (bo // 2) + 1
                        if score_a >= maps_needed or score_b >= maps_needed:
                            state.is_live = False
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=mid, game=self.game, team=winner,
                                description=f"Match won by {winner_name} ({score_a}-{score_b})",
                                impact=1.0 if winner == "a" else -1.0,
                                match_state=state,
                            ))

                # Detect ended matches
                for mid in list(self._subscribed):
                    if mid not in live_ids and mid in self._matches:
                        state = self._matches[mid]
                        if state.is_live:
                            state.is_live = False
                            winner = "a" if state.score_a > state.score_b else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b
                            logger.info(f"[PANDA END] {self.game}: {winner_name} wins {state.score_a}-{state.score_b}")
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=mid, game=self.game, team=winner,
                                description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                                impact=1.0 if winner == "a" else -1.0,
                                match_state=state,
                            ))

                if poll_count % 4 == 1:
                    logger.info(f"PandaScore {self.game}: {len(matches)} live, {len(self._subscribed)} tracked")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"PandaScore {self.game} poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _fetch_running(self, ps_slug: str) -> Optional[list]:
        """Fetch running matches from PandaScore."""
        try:
            async with self._session.get(
                f"{config.PANDASCORE_BASE_URL}/{ps_slug}/matches/running",
                params={"token": config.PANDASCORE_TOKEN, "per_page": 50},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    if resp.status == 403:
                        logger.warning("PandaScore token invalid or rate limited")
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"PandaScore fetch failed: {e}")
            return None
