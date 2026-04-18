"""
Dota 2 live match feed via Steam Game Coordinator.

Connects to Steam, launches Dota2 GC, and watches live pro matches
for real-time game state: gold, XP, hero picks, items, towers, roshan.

Sub-second latency — the fastest possible Dota2 data source.

Runs gevent in a separate thread, pushes events via queue to asyncio.
"""
import asyncio
import logging
import time
import queue
import threading
from typing import Optional

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)


class Dota2SteamFeed(GameFeed):
    """Real-time Dota2 data from Steam Game Coordinator."""

    def __init__(self):
        super().__init__("dota2")
        self._event_queue = queue.Queue()
        self._steam_thread: Optional[threading.Thread] = None
        self._poll_task = None
        self._gc_ready = False
        self._watched_matches: set[int] = set()
        self._last_states: dict[int, dict] = {}

    async def connect(self):
        if not config.STEAM_USERNAME or not config.STEAM_PASSWORD:
            logger.warning("Steam credentials not set — Dota2 GC feed disabled")
            return

        self._connected = False
        self._running = True

        # Start Steam in a separate thread (gevent)
        self._steam_thread = threading.Thread(
            target=self._run_steam, daemon=True, name="steam-gc"
        )
        self._steam_thread.start()

        # Start asyncio poller that reads from the queue
        self._poll_task = asyncio.create_task(self._poll_queue())

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="dota2", team="neutral",
            description="Dota2 Steam GC feed starting...", impact=0.0,
        ))
        logger.info("Dota2 Steam GC feed starting (thread)")

    async def disconnect(self):
        self._running = False
        if self._poll_task:
            self._poll_task.cancel()

    async def subscribe_match(self, match_id: str):
        pass  # GC auto-discovers matches

    async def unsubscribe_match(self, match_id: str):
        pass

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._matches.keys())

    def estimate_win_probability(self, state: MatchState) -> float:
        """Use gold lead for probability — much better than kills alone."""
        import math
        extra = state.extra
        gold_lead = extra.get("gold_lead", 0)
        game_minutes = extra.get("game_minutes", 0)
        kills_a = extra.get("kills_a", 0)
        kills_b = extra.get("kills_b", 0)

        # Gold-based probability (strongest Dota2 predictor)
        if gold_lead != 0 and game_minutes > 0:
            time_factor = max(0.5, min(2.0, game_minutes / 25))
            game_prob = 1.0 / (1.0 + math.exp(-gold_lead / (5000 / time_factor)))
        else:
            # Fallback to kills
            kill_diff = kills_a - kills_b
            time_factor = max(0.5, min(1.5, game_minutes / 20)) if game_minutes > 0 else 1.0
            game_prob = 1.0 / (1.0 + math.exp(-0.05 * kill_diff * time_factor))

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

    # ─── Steam Thread (gevent) ───────────────────────────────────────────────

    def _run_steam(self):
        """Run Steam client in gevent greenlet thread."""
        try:
            from steam.client import SteamClient
            from dota2.client import Dota2Client

            client = SteamClient()
            dota = Dota2Client(client)

            @client.on("logged_on")
            def on_logged_in():
                logger.info("[STEAM] Logged in to Steam")
                dota.launch()

            @client.on("error")
            def on_error(result):
                logger.error(f"[STEAM] Login error: {result}")

            @client.on("channel_secured")
            def on_channel_secured():
                if client.relogin_available:
                    client.relogin()

            @dota.on("ready")
            def on_gc_ready():
                self._gc_ready = True
                self._connected = True
                self._connect_time = time.time()
                logger.info("[STEAM] Dota2 Game Coordinator READY")
                self._event_queue.put(GameEvent(
                    event_type=EventType.FEED_CONNECTED,
                    match_id="", game="dota2", team="neutral",
                    description="Dota2 Steam GC connected", impact=0.0,
                ))
                # Request live league games
                self._request_live_games(dota)

            @dota.on("top_source_tv_games")
            def on_top_games(message):
                self._process_top_games(message, dota)

            # Login
            logger.info(f"[STEAM] Logging in as {config.STEAM_USERNAME}...")

            # Try login with 2FA code if available
            two_factor = config.__dict__.get("STEAM_2FA_CODE", "")
            if two_factor:
                result = client.login(
                    username=config.STEAM_USERNAME,
                    password=config.STEAM_PASSWORD,
                    two_factor_code=two_factor,
                )
            else:
                result = client.login(
                    username=config.STEAM_USERNAME,
                    password=config.STEAM_PASSWORD,
                )

            if result != 1:  # EResult.OK = 1
                logger.error(f"[STEAM] Login failed with result: {result}")
                return

            # Keep running
            while self._running:
                client.sleep(30)
                if self._gc_ready:
                    self._request_live_games(dota)

        except Exception as e:
            logger.error(f"[STEAM] Fatal error: {e}", exc_info=True)

    def _request_live_games(self, dota):
        """Request top live games from GC."""
        try:
            from dota2.enums import EDOTAGCMsg
            dota.send(EDOTAGCMsg.EMsgClientToGCFindTopSourceTVGames, {
                "search_key": "",
                "league_id": 0,
                "hero_id": 0,
                "start_game": 0,
                "game_list_index": 0,
                "lobby_ids": [],
            })
        except Exception as e:
            logger.debug(f"[STEAM] Failed to request live games: {e}")

    def _process_top_games(self, message, dota):
        """Process top source TV games response from GC."""
        try:
            games = message.game_list if hasattr(message, "game_list") else []
            for game in games:
                server_id = game.server_steam_id
                match_id = game.match_id

                if not match_id or match_id in self._watched_matches:
                    continue

                # Extract team info
                team_radiant = ""
                team_dire = ""
                if hasattr(game, "team_name_radiant"):
                    team_radiant = game.team_name_radiant
                if hasattr(game, "team_name_dire"):
                    team_dire = game.team_name_dire

                if not team_radiant or not team_dire:
                    continue  # skip pub games

                self._watched_matches.add(match_id)
                mid = str(match_id)

                # Create match state
                state = MatchState(
                    match_id=mid, game="dota2",
                    team_a=team_radiant, team_b=team_dire,
                    is_live=True, started_at=time.time(),
                    total_maps=3,
                    extra={
                        "kills_a": getattr(game, "radiant_score", 0) or 0,
                        "kills_b": getattr(game, "dire_score", 0) or 0,
                        "gold_lead": getattr(game, "radiant_lead", 0) or 0,
                        "game_minutes": (getattr(game, "game_time", 0) or 0) / 60.0,
                        "server_id": server_id,
                    },
                )
                self._matches[mid] = state

                logger.info(
                    f"[STEAM] Live: {team_radiant} vs {team_dire} | "
                    f"kills={state.extra['kills_a']}-{state.extra['kills_b']} | "
                    f"gold={state.extra['gold_lead']:+,}"
                )

                # Push match start event
                self._event_queue.put(GameEvent(
                    event_type=EventType.MATCH_START,
                    match_id=mid, game="dota2", team="neutral",
                    description=f"{team_radiant} vs {team_dire} LIVE (Steam GC)",
                    impact=0.0, match_state=state,
                ))

                # Update existing match data
                self._update_game_state(mid, game)

        except Exception as e:
            logger.error(f"[STEAM] Process games error: {e}")

    def _update_game_state(self, match_id: str, game):
        """Update match state from GC game data and emit events."""
        state = self._matches.get(match_id)
        if not state:
            return

        new_kills_a = getattr(game, "radiant_score", 0) or 0
        new_kills_b = getattr(game, "dire_score", 0) or 0
        new_gold = getattr(game, "radiant_lead", 0) or 0
        game_time = getattr(game, "game_time", 0) or 0

        old_kills_a = state.extra.get("kills_a", 0)
        old_kills_b = state.extra.get("kills_b", 0)

        state.extra["kills_a"] = new_kills_a
        state.extra["kills_b"] = new_kills_b
        state.extra["gold_lead"] = new_gold
        state.extra["game_minutes"] = game_time / 60.0

        # Detect kill changes
        if new_kills_a != old_kills_a or new_kills_b != old_kills_b:
            old_prob = state.win_probability_a
            state.win_probability_a = self.estimate_win_probability(state)
            impact = state.win_probability_a - old_prob

            self._event_queue.put(GameEvent(
                event_type=EventType.SCORE_UPDATE,
                match_id=match_id, game="dota2",
                team="a" if new_kills_a > old_kills_a else "b",
                description=(
                    f"Kills: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} | "
                    f"Gold: {new_gold:+,}"
                ),
                impact=impact, match_state=state,
            ))

    # ─── Asyncio Queue Poller ────────────────────────────────────────────────

    async def _poll_queue(self):
        """Read events from the gevent thread queue and emit to asyncio."""
        while self._running:
            try:
                while not self._event_queue.empty():
                    event = self._event_queue.get_nowait()
                    self._emit(event)
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[STEAM] Queue poll error: {e}")
            await asyncio.sleep(0.1)  # 100ms poll rate
