"""
Dota 2 live match feed via OpenDota API.

OpenDota /api/live returns all live pro/league matches with:
- Team names, kill scores, game time
- Match ID, series ID, league ID
- Spectator count, delay info

Free, no auth, no Cloudflare. Rate limit: 60 req/min (2000/day free).
Updates every ~60 seconds on their end.

We poll every 10 seconds and detect score changes between polls.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType
from feeds.dota2_model import Dota2WinProbabilityModel

logger = logging.getLogger(__name__)

OPENDOTA_LIVE_URL = "https://api.opendota.com/api/live"


class Dota2OpenDotaFeed(GameFeed):
    """Real-time Dota 2 match data from OpenDota."""

    POLL_INTERVAL = 5  # OpenDota updates ~every 60s, poll every 5s to catch changes ASAP

    def __init__(self):
        super().__init__("dota2")
        self._session: Optional[aiohttp.ClientSession] = None
        self._model = Dota2WinProbabilityModel()
        self._poll_task = None
        self._last_states: dict[str, dict] = {}  # match_id -> last known state
        self._subscribed_matches: set[str] = set()
        self._initialized: set[str] = set()

    async def connect(self):
        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="dota2", team="neutral",
            description="Connected to OpenDota Dota 2 feed",
            impact=0.0,
        ))
        logger.info("Dota 2 OpenDota feed connected")

        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "dota2_poll"))

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
                match_id=match_id,
                game="dota2",
                team_a="", team_b="",
                is_live=True,
                started_at=time.time(),
                extra={"kills_a": 0, "kills_b": 0, "game_minutes": 0},
            )

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)
        self._last_states.pop(match_id, None)
        self._initialized.discard(match_id)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._subscribed_matches)

    def estimate_win_probability(self, state: MatchState) -> float:
        raw = self._model.calculate(
            maps_a=state.score_a,
            maps_b=state.score_b,
            best_of=state.total_maps,
            kills_a=state.extra.get("kills_a", 0),
            kills_b=state.extra.get("kills_b", 0),
            game_minutes=state.extra.get("game_minutes", 0),
        )
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    async def _poll_loop(self):
        """Poll OpenDota live endpoint and detect match state changes."""
        logger.info("Dota 2 poll loop started")
        poll_count = 0
        while self._running:
            try:
                poll_count += 1
                games = await self._fetch_live()

                if games is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                # Filter to pro matches with team names
                pro_games = {
                    str(g["match_id"]): g
                    for g in games
                    if g.get("team_name_radiant") and g.get("team_name_dire")
                }

                # Auto-subscribe to new pro matches
                for mid, g in pro_games.items():
                    if mid not in self._subscribed_matches:
                        await self.subscribe_match(mid)
                        state = self._matches[mid]
                        state.team_a = g.get("team_name_radiant", "Radiant")
                        state.team_b = g.get("team_name_dire", "Dire")
                        state.total_maps = 3  # default Bo3, will be corrected by market data

                        # Watch links — only set if we know the specific tournament stream
                        league = g.get("league_name", "").lower()
                        if "esl" in league:
                            state.extra["stream_url"] = "https://twitch.tv/esl_dota2"
                        elif "premier" in league:
                            state.extra["stream_url"] = "https://twitch.tv/beyondthesummit"
                        elif "blast" in league:
                            state.extra["stream_url"] = "https://twitch.tv/blasttv_dota"
                        elif "pgl" in league:
                            state.extra["stream_url"] = "https://twitch.tv/pgl_dota2"
                        # No generic fallback — let Polymarket page be the watch link instead
                        # DotaTV link — opens Dota2 and watches the match
                        server_id = g.get("server_steam_id", "")
                        if server_id:
                            state.extra["watch_url"] = f"steam://run/570//+watch_server%20{server_id}"
                        else:
                            state.extra["watch_url"] = f"steam://run/570//+watch_server%20{mid}"

                        logger.info(f"Subscribed: {state.team_a} vs {state.team_b} (dota2) [{mid}]")
                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=mid, game="dota2", team="neutral",
                            description=f"{state.team_a} vs {state.team_b} LIVE",
                            impact=0.0, match_state=state,
                        ))

                # Process state changes for tracked matches
                for mid in list(self._subscribed_matches):
                    if mid in pro_games:
                        self._process_game(mid, pro_games[mid])
                    else:
                        # Match no longer in live feed — has ended
                        state = self._matches.get(mid)
                        if state and state.is_live and mid in self._initialized:
                            state.is_live = False
                            ka = state.extra.get("kills_a", 0)
                            kb = state.extra.get("kills_b", 0)
                            winner = "a" if ka > kb else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b
                            logger.info(f"[GAME END] {winner_name} wins game ({ka}-{kb})")

                            # Update series score
                            if winner == "a":
                                state.score_a += 1
                            else:
                                state.score_b += 1

                            # Emit MAP_WIN (game win in Dota2 = map win in series)
                            old_prob = state.win_probability_a
                            state.win_probability_a = self.estimate_win_probability(state)
                            self._emit(GameEvent(
                                event_type=EventType.MAP_WIN,
                                match_id=mid, game="dota2", team=winner,
                                description=f"Game to {winner_name} (Series: {state.score_a}-{state.score_b})",
                                impact=state.win_probability_a - old_prob,
                                match_state=state,
                            ))

                            # Check if series is over
                            maps_needed = (state.total_maps // 2) + 1
                            if state.score_a >= maps_needed or state.score_b >= maps_needed:
                                logger.info(f"[MATCH END] {winner_name} wins series {state.score_a}-{state.score_b}")
                                self._emit(GameEvent(
                                    event_type=EventType.MATCH_END,
                                    match_id=mid, game="dota2", team=winner,
                                    description=f"Series won by {winner_name} ({state.score_a}-{state.score_b})",
                                    impact=1.0 if winner == "a" else -1.0,
                                    match_state=state,
                                ))

                # Log progress
                if poll_count % 6 == 1:
                    logger.info(f"Dota 2: {len(pro_games)} live pro, {len(self._subscribed_matches)} tracked")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dota 2 poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _fetch_live(self) -> Optional[list]:
        """Fetch live matches from OpenDota."""
        try:
            async with self._session.get(OPENDOTA_LIVE_URL, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                if resp.status != 200:
                    logger.warning(f"OpenDota returned {resp.status}")
                    return None
                return await resp.json()
        except Exception as e:
            logger.error(f"OpenDota fetch failed: {e}")
            return None

    def _process_game(self, match_id: str, game: dict):
        """Process a live game update from OpenDota."""
        state = self._matches.get(match_id)
        if not state:
            return

        now = time.time()
        new_kills_a = game.get("radiant_score", 0) or 0
        new_kills_b = game.get("dire_score", 0) or 0
        game_time = game.get("game_time", 0) or 0
        game_minutes = game_time / 60.0
        gold_lead = game.get("radiant_lead", 0) or 0  # positive = radiant leads
        building_state = game.get("building_state", 0)

        # Extract hero picks from player data
        players = game.get("players", [])
        heroes_a = [p.get("hero_id", 0) for p in players if p.get("team") == 0]  # radiant
        heroes_b = [p.get("hero_id", 0) for p in players if p.get("team") == 1]  # dire

        # Count towers from building_state bitmask
        # Bits 0-10 = radiant buildings, 11-21 = dire buildings
        towers_a = bin(building_state & 0x7FF).count("1") if building_state else 0
        towers_b = bin((building_state >> 11) & 0x7FF).count("1") if building_state else 0

        # Initialize on first seen
        if match_id not in self._initialized:
            self._initialized.add(match_id)
            state.extra["kills_a"] = new_kills_a
            state.extra["kills_b"] = new_kills_b
            state.extra["game_minutes"] = game_minutes
            state.extra["gold_lead"] = gold_lead
            state.extra["towers_a"] = towers_a
            state.extra["towers_b"] = towers_b
            state.extra["heroes_a"] = heroes_a
            state.extra["heroes_b"] = heroes_b
            state.win_probability_a = self.estimate_win_probability(state)
            state.last_event_time = now
            gold_str = f" gold_lead={gold_lead:+,}" if gold_lead else ""
            logger.info(
                f"[INIT] {match_id} {state.team_a} vs {state.team_b} | "
                f"kills={new_kills_a}-{new_kills_b} {game_minutes:.0f}min{gold_str} | "
                f"P(A)={state.win_probability_a:.3f}"
            )
            return

        old_kills_a = state.extra.get("kills_a", 0)
        old_kills_b = state.extra.get("kills_b", 0)

        # Update state
        state.extra["kills_a"] = new_kills_a
        state.extra["kills_b"] = new_kills_b
        state.extra["game_minutes"] = game_minutes
        state.extra["gold_lead"] = gold_lead
        state.extra["towers_a"] = towers_a
        state.extra["towers_b"] = towers_b
        if heroes_a:
            state.extra["heroes_a"] = heroes_a
            state.extra["heroes_b"] = heroes_b

        # Detect kill score changes
        kills_changed = (new_kills_a != old_kills_a or new_kills_b != old_kills_b)
        if not kills_changed:
            return

        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)
        impact = state.win_probability_a - old_prob

        # Determine which team got kills
        diff_a = new_kills_a - old_kills_a
        diff_b = new_kills_b - old_kills_b
        if diff_a > diff_b:
            team = "a"
        elif diff_b > diff_a:
            team = "b"
        else:
            team = "neutral"

        logger.info(
            f"[KILLS] {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} | "
            f"{game_minutes:.0f}min | P(A)={state.win_probability_a:.3f} (was {old_prob:.3f})"
        )

        # Emit as SCORE_UPDATE (kills are the "score" in Dota 2)
        self._emit(GameEvent(
            event_type=EventType.SCORE_UPDATE,
            match_id=match_id, game="dota2", team=team,
            description=f"Kills: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} ({game_minutes:.0f}min)",
            impact=impact, timestamp=now, match_state=state,
        ))

        # Big kill swings (3+ kills in one update = teamfight)
        total_new_kills = diff_a + diff_b
        if total_new_kills >= 3:
            self._emit(GameEvent(
                event_type=EventType.TEAMFIGHT_WON,
                match_id=match_id, game="dota2", team=team,
                description=f"Teamfight! {diff_a}-{diff_b} kills ({state.team_a} favor)" if diff_a > diff_b else f"Teamfight! {diff_b}-{diff_a} kills ({state.team_b} favor)",
                impact=impact * 1.5, timestamp=now, match_state=state,
            ))

        state.last_event_time = now
