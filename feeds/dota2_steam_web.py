"""
Dota 2 live match feed via Steam Web API (GetLiveLeagueGames).

Uses Valve's official API with a free Steam API key.
Provides per-player gold, hero picks, tower state — much richer than OpenDota.

Updates every ~5 seconds. Covers all league/tournament matches.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

STEAM_API_URL = "https://api.steampowered.com/IDOTA2Match_570/GetLiveLeagueGames/v1/"

# Hero ID → name mapping (loaded on first use)
_HERO_NAMES: dict = {}


class Dota2SteamWebFeed(GameFeed):
    """Rich Dota2 data from Steam Web API — gold, heroes, towers per player."""

    POLL_INTERVAL = 5  # seconds

    def __init__(self):
        super().__init__("dota2")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        self._subscribed: set[str] = set()
        self._initialized: set[str] = set()

    async def connect(self):
        if not config.STEAM_API_KEY:
            logger.warning("STEAM_API_KEY not set — Steam Web feed disabled")
            return

        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        # Load hero names
        await self._load_hero_names()

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="dota2", team="neutral",
            description="Connected to Dota2 Steam Web API", impact=0.0,
        ))
        logger.info("Dota2 Steam Web API feed connected")
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "dota2_steam_web"))

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
        """Gold-based probability — the strongest Dota2 predictor."""
        import math
        extra = state.extra
        gold_lead = extra.get("gold_lead", 0)
        game_minutes = extra.get("game_minutes", 0)
        kills_a = extra.get("kills_a", 0)
        kills_b = extra.get("kills_b", 0)

        if gold_lead != 0 and game_minutes > 2:
            time_factor = max(0.5, min(2.0, game_minutes / 25))
            game_prob = 1.0 / (1.0 + math.exp(-gold_lead / (5000 / time_factor)))
        elif kills_a + kills_b > 0:
            kill_diff = kills_a - kills_b
            time_factor = max(0.5, min(1.5, game_minutes / 20)) if game_minutes > 0 else 1.0
            game_prob = 1.0 / (1.0 + math.exp(-0.05 * kill_diff * time_factor))
        else:
            game_prob = 0.5

        # Series context
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        series_prob = b_needs / (a_needs + b_needs)

        raw = max(0.01, min(0.99, 0.5 * series_prob + 0.5 * game_prob))

        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    async def _load_hero_names(self):
        """Load hero ID → name mapping from OpenDota."""
        global _HERO_NAMES
        if _HERO_NAMES:
            return
        try:
            async with self._session.get(
                "https://api.opendota.com/api/heroes",
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 200:
                    heroes = await resp.json()
                    _HERO_NAMES = {h["id"]: h["localized_name"] for h in heroes}
                    logger.info(f"Loaded {len(_HERO_NAMES)} Dota2 hero names")
        except Exception:
            pass

    # ─── Polling Loop ────────────────────────────────────────────────────────

    async def _poll_loop(self):
        """Poll Steam Web API for live league games."""
        logger.info("Dota2 Steam Web API poll loop started")
        poll_count = 0
        while self._running:
            try:
                poll_count += 1
                games = await self._fetch_live()
                if games is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                # Filter to pro matches with team names
                pro_games = {}
                for g in games:
                    rt = g.get("radiant_team", {})
                    dt = g.get("dire_team", {})
                    if rt.get("team_name") and dt.get("team_name"):
                        mid = str(g.get("match_id", ""))
                        if mid:
                            pro_games[mid] = g

                # Auto-subscribe new matches
                for mid, g in pro_games.items():
                    if mid not in self._subscribed:
                        await self.subscribe_match(mid)
                        rt = g.get("radiant_team", {})
                        dt = g.get("dire_team", {})

                        # Get server ID for DotaTV watch link
                        server_id = g.get("server_steam_id", "")
                        league = g.get("league_id", 0)

                        state = MatchState(
                            match_id=mid, game="dota2",
                            team_a=rt.get("team_name", "Radiant"),
                            team_b=dt.get("team_name", "Dire"),
                            is_live=True, started_at=time.time(),
                            total_maps=3,
                            extra={
                                "kills_a": 0, "kills_b": 0, "gold_lead": 0, "game_minutes": 0,
                                "stream_url": "https://twitch.tv/dota2",
                                "watch_url": f"steam://run/570//+watch_server%20{server_id}" if server_id else f"steam://run/570//+watch_server%20{mid}",
                            },
                        )
                        self._matches[mid] = state
                        logger.info(f"Subscribed: {state.team_a} vs {state.team_b} (Steam Web) [{mid}]")

                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=mid, game="dota2", team="neutral",
                            description=f"{state.team_a} vs {state.team_b} LIVE (Steam Web)",
                            impact=0.0, match_state=state,
                        ))

                # Process state changes
                for mid in list(self._subscribed):
                    if mid in pro_games:
                        self._process_game(mid, pro_games[mid])
                    else:
                        # Match ended
                        state = self._matches.get(mid)
                        if state and state.is_live and mid in self._initialized:
                            state.is_live = False
                            ka = state.extra.get("kills_a", 0)
                            kb = state.extra.get("kills_b", 0)
                            winner = "a" if ka > kb else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b
                            logger.info(f"[GAME END] {winner_name} wins ({ka}-{kb})")

                            state.score_a += 1 if winner == "a" else 0
                            state.score_b += 1 if winner == "b" else 0

                            self._emit(GameEvent(
                                event_type=EventType.MAP_WIN,
                                match_id=mid, game="dota2", team=winner,
                                description=f"Game to {winner_name} (Series: {state.score_a}-{state.score_b})",
                                impact=0.3 if winner == "a" else -0.3,
                                match_state=state,
                            ))

                            maps_needed = (state.total_maps // 2) + 1
                            if state.score_a >= maps_needed or state.score_b >= maps_needed:
                                self._emit(GameEvent(
                                    event_type=EventType.MATCH_END,
                                    match_id=mid, game="dota2", team=winner,
                                    description=f"Series won by {winner_name} ({state.score_a}-{state.score_b})",
                                    impact=1.0 if winner == "a" else -1.0,
                                    match_state=state,
                                ))

                if poll_count % 12 == 1:
                    logger.info(f"Dota2 Steam Web: {len(pro_games)} pro live, {len(self._subscribed)} tracked")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Dota2 Steam Web poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _fetch_live(self) -> Optional[list]:
        """Fetch live league games from Steam Web API."""
        try:
            async with self._session.get(
                STEAM_API_URL,
                params={"key": config.STEAM_API_KEY},
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"Steam Web API returned {resp.status}")
                    return None
                # Handle non-UTF8 team names (Chinese characters etc)
                raw = await resp.read()
                import json
                data = json.loads(raw.decode("utf-8", errors="replace"))
                return data.get("result", {}).get("games", [])
        except Exception as e:
            logger.error(f"Steam Web API fetch failed: {e}")
            return None

    def _process_game(self, match_id: str, game: dict):
        """Process a live game update with rich data."""
        state = self._matches.get(match_id)
        if not state:
            return

        now = time.time()
        new_kills_a = game.get("radiant_score", 0) or 0
        new_kills_b = game.get("dire_score", 0) or 0
        game_time = game.get("game_time", 0) or 0
        game_minutes = game_time / 60.0

        # Calculate gold from per-player data
        players = game.get("players", [])
        radiant_gold = sum(p.get("gold", 0) for p in players if p.get("team", -1) == 0)
        dire_gold = sum(p.get("gold", 0) for p in players if p.get("team", -1) == 1)
        gold_lead = radiant_gold - dire_gold

        # Extract hero picks
        heroes_a = [_HERO_NAMES.get(p.get("hero_id", 0), f"Hero#{p.get('hero_id',0)}")
                     for p in players if p.get("team", -1) == 0 and p.get("hero_id", 0) > 0]
        heroes_b = [_HERO_NAMES.get(p.get("hero_id", 0), f"Hero#{p.get('hero_id',0)}")
                     for p in players if p.get("team", -1) == 1 and p.get("hero_id", 0) > 0]

        # Tower state from scoreboard
        tower_state = game.get("scoreboard", {})
        towers_a = tower_state.get("radiant", {}).get("tower_state", 0)
        towers_b = tower_state.get("dire", {}).get("tower_state", 0)
        # Count standing towers from bitmask
        t_a = bin(towers_a).count("1") if towers_a else 0
        t_b = bin(towers_b).count("1") if towers_b else 0

        # Initialize
        if match_id not in self._initialized:
            self._initialized.add(match_id)
            state.extra.update({
                "kills_a": new_kills_a, "kills_b": new_kills_b,
                "gold_lead": gold_lead, "game_minutes": game_minutes,
                "towers_a": t_a, "towers_b": t_b,
                "heroes_a": heroes_a, "heroes_b": heroes_b,
                "radiant_gold": radiant_gold, "dire_gold": dire_gold,
            })
            state.win_probability_a = self.estimate_win_probability(state)
            state.last_event_time = now

            hero_str = ", ".join(heroes_a[:3]) if heroes_a else "drafting"
            logger.info(
                f"[INIT] Steam Web {state.team_a} vs {state.team_b} | "
                f"kills={new_kills_a}-{new_kills_b} gold={gold_lead:+,} {game_minutes:.0f}min | "
                f"heroes: {hero_str}"
            )
            return

        old_kills_a = state.extra.get("kills_a", 0)
        old_kills_b = state.extra.get("kills_b", 0)

        # Update state
        state.extra.update({
            "kills_a": new_kills_a, "kills_b": new_kills_b,
            "gold_lead": gold_lead, "game_minutes": game_minutes,
            "towers_a": t_a, "towers_b": t_b,
            "radiant_gold": radiant_gold, "dire_gold": dire_gold,
        })
        if heroes_a:
            state.extra["heroes_a"] = heroes_a
            state.extra["heroes_b"] = heroes_b

        # Detect changes
        kills_changed = (new_kills_a != old_kills_a or new_kills_b != old_kills_b)
        if not kills_changed:
            return

        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)
        impact = state.win_probability_a - old_prob
        state.last_event_time = now

        # Determine event type and which team came out ahead
        kill_diff = abs((new_kills_a - new_kills_b) - (old_kills_a - old_kills_b))
        kills_gained_a = new_kills_a - old_kills_a
        kills_gained_b = new_kills_b - old_kills_b
        team = "a" if kills_gained_a > kills_gained_b else "b" if kills_gained_b > kills_gained_a else "neutral"

        if kill_diff >= 2:
            event_type = EventType.TEAMFIGHT_WON
            desc = f"Teamfight! {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} | Gold: {gold_lead:+,}"
        else:
            event_type = EventType.SCORE_UPDATE
            desc = f"Kills: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} | Gold: {gold_lead:+,}"

        logger.info(f"[KILLS] Steam Web {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} | gold={gold_lead:+,} | {game_minutes:.0f}min")

        self._emit(GameEvent(
            event_type=event_type,
            match_id=match_id, game="dota2", team=team,
            description=desc, impact=impact, match_state=state,
        ))
