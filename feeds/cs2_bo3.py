"""
CS2 live match feed via Bo3.gg API.

Bo3.gg provides real-time match snapshots with rich data:
- Round scores, round phase, bomb status
- Team sides (CT/T), series scores
- Per-player economy (balance, equipment_value)
- Per-player stats (kills, deaths, is_alive, health)
- Map name, game number

Data reportedly arrives ~30 seconds before live broadcast.
Free, no auth required. Uses the cs2api Python wrapper.

Polling-based: fetches snapshots every POLL_INTERVAL seconds.
State changes are detected by comparing consecutive snapshots.
"""
import asyncio
import logging
import time
from typing import Optional

import cs2api

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType
from feeds.cs2_model import CS2WinProbabilityModel, classify_buy

logger = logging.getLogger(__name__)


class CS2Bo3Feed(GameFeed):
    """Real-time CS2 match data from Bo3.gg."""

    POLL_INTERVAL = 1  # seconds between snapshot polls — faster = more edge

    def __init__(self):
        super().__init__("cs2")
        self._client: Optional[cs2api.CS2] = None
        self._model = CS2WinProbabilityModel()
        self._poll_task = None
        self._discovery_task = None
        self._last_snapshots: dict[str, dict] = {}  # str match_id -> last snapshot
        self._subscribed_matches: set[str] = set()
        self._initialized_matches: set[str] = set()  # matches that had first snapshot processed
        self._ws_tracked_matches: set[str] = set()  # matches tracked by WS — suppress polling events
        # Per-match per-player tracking for granular event detection
        # Structure: { match_id: { player_id: {"kills": int, "deaths": int, "hp": int,
        #                                      "alive": bool, "weapon": str, "side": str, "team": "a"|"b"} } }
        self._player_prev: dict[str, dict] = {}
        # Matches where we've already emitted OPEN_KILL for the current round
        # (match_id, round_key) pairs — reset when round advances
        self._open_kill_fired: dict[str, str] = {}  # match_id -> round_key that got open_kill

    async def connect(self):
        self._client = cs2api.CS2()
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="",
            game="cs2",
            team="neutral",
            description="Connected to Bo3.gg CS2 feed",
            impact=0.0,
        ))
        logger.info("Bo3.gg CS2 feed connected")

        # Run discovery FIRST to populate matches, THEN start polling
        self._discovery_task = asyncio.create_task(self._safe_run(self._discovery_loop(), "discovery"))
        # Small delay so discovery populates matches before polling starts
        await asyncio.sleep(2)
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "poll"))

    async def _safe_run(self, coro, name: str):
        """Run a coroutine and log any unhandled exceptions."""
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
        if self._discovery_task:
            self._discovery_task.cancel()
        if self._client:
            await self._client.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._subscribed_matches.add(match_id)
        if match_id not in self._matches:
            self._matches[match_id] = MatchState(
                match_id=match_id,
                game="cs2",
                team_a="",
                team_b="",
                is_live=True,
                started_at=time.time(),
                extra={
                    "team_a_start_side": "",
                    "team_a_current_side": "",
                    "side_determined": False,
                    "team_a_money": 0,
                    "team_b_money": 0,
                    "round_kills_a": 0,
                    "round_kills_b": 0,
                    "alive_a": 5, "alive_b": 5,
                    "avg_hp_a": 100, "avg_hp_b": 100,
                    "total_hp_a": 500, "total_hp_b": 500,
                    "damage_a": 0, "damage_b": 0,
                    "has_awp_a": False, "has_awp_b": False,
                    "has_defuse_kit_a": False,
                    "bomb_carrier": "",
                    "buy_type_a": "", "buy_type_b": "",
                },
            )

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)
        self._last_snapshots.pop(match_id, None)
        self._initialized_matches.discard(match_id)
        self._player_prev.pop(match_id, None)
        self._open_kill_fired.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        if not self._client:
            return []
        try:
            resp = await self._client.get_live_matches()
            return [str(m["id"]) for m in resp.get("results", []) if m.get("status") == "current"]
        except Exception as e:
            logger.error(f"Bo3.gg discovery failed: {e}")
            return []

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

    # ─── Discovery Loop ──────────────────────────────────────────────────────

    async def _discovery_loop(self):
        """Periodically discover and auto-subscribe to live matches."""
        while self._running:
            try:
                resp = await self._client.get_live_matches()
                results = resp.get("results", [])
                live_count = 0

                for m in results:
                    if m.get("status") != "current":
                        continue
                    live_count += 1

                    mid = str(m["id"])

                    # For already-subscribed matches: check map score changes
                    # This catches PGL matches without snapshot data
                    if mid in self._subscribed_matches:
                        state = self._matches.get(mid)
                        if state and state.is_live:
                            new_map_a = m.get("team1_score", 0) or 0
                            new_map_b = m.get("team2_score", 0) or 0
                            if new_map_a != state.score_a or new_map_b != state.score_b:
                                old_a, old_b = state.score_a, state.score_b
                                state.score_a = new_map_a
                                state.score_b = new_map_b
                                state.current_map = new_map_a + new_map_b + 1
                                winner = "a" if new_map_a > old_a else "b"
                                winner_name = state.team_a if winner == "a" else state.team_b

                                old_prob = state.win_probability_a
                                state.win_probability_a = self.estimate_win_probability(state)
                                impact = state.win_probability_a - old_prob

                                logger.info(f"[MAP] {winner_name} wins map! Series: {new_map_a}-{new_map_b}")
                                self._emit(GameEvent(
                                    event_type=EventType.MAP_WIN,
                                    match_id=mid, game="cs2", team=winner,
                                    description=f"Map to {winner_name} (Series: {new_map_a}-{new_map_b})",
                                    impact=impact, timestamp=time.time(), match_state=state,
                                ))

                                # Check if series over
                                maps_needed = (state.total_maps // 2) + 1
                                if state.score_a >= maps_needed or state.score_b >= maps_needed:
                                    state.is_live = False
                                    logger.info(f"[MATCH END] {winner_name} wins series {state.score_a}-{state.score_b}")
                                    self._emit(GameEvent(
                                        event_type=EventType.MATCH_END,
                                        match_id=mid, game="cs2", team=winner,
                                        description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                                        impact=1.0 if winner == "a" else -1.0,
                                        timestamp=time.time(), match_state=state,
                                    ))
                        continue

                    team1 = m.get("team1", {})
                    team2 = m.get("team2", {})
                    t1_name = team1.get("name", "") if isinstance(team1, dict) else ""
                    t2_name = team2.get("name", "") if isinstance(team2, dict) else ""
                    bo_type = m.get("bo_type", 1)

                    # Skip matches that have already been decided — bo3.gg sometimes
                    # keeps status=current for a few minutes after series end, which
                    # re-emits a phantom MATCH_END on restart and sends qwen stale
                    # post-match events. (Bug: Spirit vs G2 2-0 re-emitted at 22:55:45
                    # after restart even though series ended ~6 min earlier.)
                    _s_a = m.get("team1_score", 0) or 0
                    _s_b = m.get("team2_score", 0) or 0
                    _needed = (bo_type // 2) + 1
                    if _s_a >= _needed or _s_b >= _needed:
                        logger.info(f"[SKIP-ENDED] {t1_name} vs {t2_name} already {_s_a}-{_s_b} (Bo{bo_type}) — not subscribing")
                        continue

                    await self.subscribe_match(mid)
                    state = self._matches[mid]
                    state.team_a = t1_name
                    state.team_b = t2_name
                    state.total_maps = bo_type
                    state.score_a = m.get("team1_score", 0) or 0
                    state.score_b = m.get("team2_score", 0) or 0
                    state.current_map = state.score_a + state.score_b + 1

                    # Watch links: launch CS2 Watch tab + Bo3.gg link
                    slug = m.get("slug", "")
                    if slug:
                        state.extra["watch_url"] = "steam://run/730"  # launches CS2
                        state.extra["bo3gg_url"] = f"https://bo3.gg/matches/{slug}"
                    streams = m.get("streams", [])
                    for s in streams:
                        preview = s.get("preview_image_url", "")
                        name = s.get("name", "")
                        # Extract YouTube video ID from thumbnail
                        if "ytimg.com" in preview:
                            import re as _re
                            yt_match = _re.search(r"/vi/([^/]+)/", preview)
                            if yt_match:
                                state.extra["stream_url"] = f"https://youtube.com/watch?v={yt_match.group(1)}"
                                break
                        # Twitch streams have jtvnw.net thumbnails
                        elif "jtvnw.net" in preview:
                            twitch_match = _re.search(r"live_user_(\w+)", preview) if "_re" in dir() else None
                            if not twitch_match:
                                import re as _re
                                twitch_match = _re.search(r"live_user_(\w+)", preview)
                            if twitch_match:
                                state.extra["stream_url"] = f"https://twitch.tv/{twitch_match.group(1)}"
                                break

                    logger.info(f"Subscribed: {t1_name} vs {t2_name} (Bo{bo_type}) [{mid}]")

                    self._emit(GameEvent(
                        event_type=EventType.MATCH_START,
                        match_id=mid,
                        game="cs2",
                        team="neutral",
                        description=f"{t1_name} vs {t2_name} (Bo{bo_type}) LIVE",
                        impact=0.0,
                        match_state=state,
                    ))

                # Detect ended matches and emit MATCH_END
                live_ids = {str(m["id"]) for m in results if m.get("status") == "current"}
                for mid in list(self._subscribed_matches):
                    if mid not in live_ids and mid in self._matches:
                        state = self._matches[mid]
                        if state.is_live:
                            state.is_live = False
                            winner = "a" if state.score_a > state.score_b else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b
                            logger.info(f"[MATCH END] {mid} {winner_name} wins {state.score_a}-{state.score_b}")
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=mid, game="cs2", team=winner,
                                description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                                impact=1.0 if winner == "a" else -1.0,
                                timestamp=time.time(), match_state=state,
                            ))

                logger.info(f"Bo3.gg: {live_count} live, {len(self._subscribed_matches)} tracked")

            except Exception as e:
                logger.error(f"Discovery error: {e}")

            await asyncio.sleep(config.CS2_DISCOVERY_INTERVAL)

    # ─── Polling Loop ─────────────────────────────────────────────────────────

    async def _poll_loop(self):
        """Poll live match snapshots and detect state changes."""
        logger.info(f"Poll loop started with {len(self._subscribed_matches)} matches")
        snap_count = 0
        event_count = 0
        while self._running:
            try:
                for match_id in list(self._subscribed_matches):
                    if not self._running:
                        break

                    state = self._matches.get(match_id)
                    if not state or not state.is_live:
                        continue

                    bo3_id = int(match_id) if match_id.isdigit() else 0
                    if not bo3_id:
                        continue

                    try:
                        snap = await self._client.get_live_match_snapshot(bo3_id)
                        if snap:
                            snap_count += 1
                            events = self._process_snapshot(match_id, snap)
                            event_count += events
                    except Exception as e:
                        if "404" not in str(e):
                            logger.error(f"Snapshot error {match_id}: {e}")

                # Log progress periodically
                if snap_count > 0 and snap_count % 50 == 0:
                    logger.info(f"Polled {snap_count} snapshots, emitted {event_count} events")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Poll loop error: {e}", exc_info=True)
                await asyncio.sleep(self.POLL_INTERVAL)

    # ─── Snapshot Processing ──────────────────────────────────────────────────

    def _process_snapshot(self, match_id: str, snap: dict) -> int:
        """
        Compare snapshot to previous state and emit events for changes.
        Returns number of events emitted.
        """
        state = self._matches.get(match_id)
        if not state:
            return 0

        # Emit the FULL bo3.gg payload for recording BEFORE state-diff logic.
        # This is what the backtester's trigger engine + training pipeline
        # need: player_states, round_phase, is_bomb_planted, HP, equipment,
        # round_time, sides, etc. Without this the recorder only sees our
        # synthesized 8-field MatchState which is useless for backtesting.
        # (The WS feed at cs2_bo3_ws.py has the same emit.)
        try:
            self._emit_raw_snapshot(match_id, snap)
        except Exception:
            pass

        now = time.time()
        prev = self._last_snapshots.get(match_id)
        self._last_snapshots[match_id] = snap
        events_emitted = 0

        team_one = snap.get("team_one") or {}
        team_two = snap.get("team_two") or {}

        # Update team names
        if not state.team_a and team_one:
            state.team_a = team_one.get("fixture", {}).get("team_name", "") or team_one.get("name", "")
        if not state.team_b and team_two:
            state.team_b = team_two.get("fixture", {}).get("team_name", "") or team_two.get("name", "")

        # Parse snapshot data
        new_score_a = team_one.get("score", 0) or 0
        new_score_b = team_two.get("score", 0) or 0
        new_map_a = team_one.get("match_score", 0) or 0
        new_map_b = team_two.get("match_score", 0) or 0
        is_bomb_planted = snap.get("is_bomb_planted", False)
        game_ended = snap.get("game_ended", False)
        match_status = snap.get("match_status", "")

        # Extract sides
        side_a = (team_one.get("side", "") or "").upper()
        if side_a in ("CT", "TERRORIST"):
            state.extra["team_a_current_side"] = "ct" if side_a == "CT" else "t"
            if not state.extra.get("team_a_start_side"):
                state.extra["team_a_start_side"] = state.extra["team_a_current_side"]

        # Extract economy
        state.extra["team_a_money"] = self._sum_team_money(team_one)
        state.extra["team_b_money"] = self._sum_team_money(team_two)

        # Extract kills this round
        kills_a = self._sum_round_kills(team_one)
        kills_b = self._sum_round_kills(team_two)
        state.extra["round_kills_a"] = kills_a
        state.extra["round_kills_b"] = kills_b

        # ─── RICH PER-PLAYER ROLLUP (alive counts, HP, damage, weapons) ──────
        # Gives qwen a live picture of the round in progress instead of just
        # "round ended 1-0". Bo3.gg sends this every 1s.
        stats_a = self._team_rollup(team_one)
        stats_b = self._team_rollup(team_two)
        state.extra["alive_a"] = stats_a["alive"]
        state.extra["alive_b"] = stats_b["alive"]
        state.extra["avg_hp_a"] = stats_a["avg_hp"]
        state.extra["avg_hp_b"] = stats_b["avg_hp"]
        state.extra["total_hp_a"] = stats_a["total_hp"]
        state.extra["total_hp_b"] = stats_b["total_hp"]
        state.extra["damage_a"] = stats_a["damage"]
        state.extra["damage_b"] = stats_b["damage"]
        state.extra["has_awp_a"] = stats_a["has_awp"]
        state.extra["has_awp_b"] = stats_b["has_awp"]
        state.extra["has_defuse_kit_a"] = stats_a["has_defuse_kit"]
        state.extra["bomb_carrier"] = (
            "a" if stats_a["has_bomb"] else "b" if stats_b["has_bomb"] else ""
        )

        # Exact buy-type classification from equipment_value_round_start
        buy_type_a = self._classify_buy(team_one)
        buy_type_b = self._classify_buy(team_two)
        if buy_type_a:
            state.extra["buy_type_a"] = buy_type_a
        if buy_type_b:
            state.extra["buy_type_b"] = buy_type_b

        # ─── First snapshot: initialize state, don't emit change events ───
        if match_id not in self._initialized_matches:
            self._initialized_matches.add(match_id)
            # STALE-ROUND-SCORE GUARD: when bo3.gg returns a match mid-series, the
            # snapshot sometimes carries the PREVIOUS (finished) map's final round
            # score (e.g. 13-6) while the current map is actually just starting.
            # Feeding that stale round score to qwen → hallucinated "team is 1 round
            # from winning" when they're actually losing. (Bug: Vitality vs Falcons
            # rounds=13-6 on map 3 init was map 2's final; Falcons actually won map 3.)
            # If series is in-progress AND round score looks map-final, zero it.
            series_in_progress = (new_map_a + new_map_b) > 0
            looks_map_final = (new_score_a >= 13 or new_score_b >= 13 or (new_score_a + new_score_b) >= 16)
            if series_in_progress and looks_map_final:
                logger.warning(
                    f"[INIT-STALE] {match_id} map={new_map_a+new_map_b+1} but rounds={new_score_a}-{new_score_b} "
                    f"looks like a finished map — zeroing round score + economy until fresh data arrives"
                )
                new_score_a = 0
                new_score_b = 0
                # Invalidate stale per-round context so qwen sees "unknown" not "massive advantage"
                state.extra["team_a_money"] = 0
                state.extra["team_b_money"] = 0
                state.extra["round_kills_a"] = 0
                state.extra["round_kills_b"] = 0
                state.extra["stale_init"] = True
            state.round_score_a = new_score_a
            state.round_score_b = new_score_b
            state.score_a = new_map_a
            state.score_b = new_map_b
            state.current_map = new_map_a + new_map_b + 1  # derive from series score
            state.win_probability_a = self.estimate_win_probability(state)
            state.last_event_time = now
            logger.info(
                f"[INIT] {match_id} {state.team_a} vs {state.team_b} | "
                f"maps={new_map_a}-{new_map_b} rounds={new_score_a}-{new_score_b} | "
                f"P(A)={state.win_probability_a:.3f}"
            )
            return 0

        # ─── Detect changes ──────────────────────────────────────────────

        old_score_a = state.round_score_a
        old_score_b = state.round_score_b
        old_map_a = state.score_a
        old_map_b = state.score_b

        # Update round scores
        state.round_score_a = new_score_a
        state.round_score_b = new_score_b

        # Round end (score changed)
        if new_score_a != old_score_a or new_score_b != old_score_b:
            # Fresh round data has arrived — it's safe to trust state now.
            if state.extra.pop("stale_init", False):
                logger.info(f"[FRESH] {match_id} stale_init cleared — round score updated to {new_score_a}-{new_score_b}")
            winning_team = "a" if new_score_a > old_score_a else "b"

            old_prob = state.win_probability_a
            state.win_probability_a = self.estimate_win_probability(state)
            impact = state.win_probability_a - old_prob

            winner_name = state.team_a if winning_team == "a" else state.team_b
            logger.info(
                f"[ROUND] {state.team_a} {new_score_a}-{new_score_b} {state.team_b} | "
                f"P(A)={state.win_probability_a:.3f} (was {old_prob:.3f})"
            )

            # Suppress round events if WS feed is tracking this match
            # (WS has correct team ordering and is faster)
            if match_id not in self._ws_tracked_matches:
                self._emit(GameEvent(
                    event_type=EventType.ROUND_END,
                    match_id=match_id, game="cs2", team=winning_team,
                    description=f"Round to {winner_name} ({new_score_a}-{new_score_b})",
                    impact=impact, timestamp=now, match_state=state,
                ))
                events_emitted += 1

            self._emit(GameEvent(
                event_type=EventType.ECONOMY_SHIFT,
                match_id=match_id, game="cs2", team=winning_team,
                description=f"Eco: {state.team_a}=${state.extra['team_a_money']:,} vs {state.team_b}=${state.extra['team_b_money']:,}",
                impact=0.02 if winning_team == "a" else -0.02,
                timestamp=now, match_state=state,
            ))
            events_emitted += 1

        # Map end (series score changed)
        if new_map_a != old_map_a or new_map_b != old_map_b:
            state.score_a = new_map_a
            state.score_b = new_map_b
            winning_team = "a" if new_map_a > old_map_a else "b"

            old_prob = state.win_probability_a
            state.win_probability_a = self.estimate_win_probability(state)
            impact = state.win_probability_a - old_prob

            winner_name = state.team_a if winning_team == "a" else state.team_b
            logger.info(f"[MAP] {winner_name} wins map! Series: {new_map_a}-{new_map_b}")

            self._emit(GameEvent(
                event_type=EventType.MAP_WIN,
                match_id=match_id, game="cs2", team=winning_team,
                description=f"Map to {winner_name} (Series: {new_map_a}-{new_map_b})",
                impact=impact, timestamp=now, match_state=state,
            ))
            events_emitted += 1

            # Reset for new map — clear ALL per-map state so qwen doesn't get
            # stale economy/kills/sides from the just-ended map applied to the next.
            state.round_score_a = 0
            state.round_score_b = 0
            state.extra["team_a_start_side"] = ""
            state.extra["side_determined"] = False
            state.extra["team_a_money"] = 0
            state.extra["team_b_money"] = 0
            state.extra["round_kills_a"] = 0
            state.extra["round_kills_b"] = 0
            state.extra["team_a_current_side"] = ""
            state.extra["map_transition_ts"] = now
            state.current_map = snap.get("game_number", state.current_map)
            # Clear per-tick rollup + per-player tracking so new map starts fresh
            for k in ("alive_a","alive_b","avg_hp_a","avg_hp_b","total_hp_a","total_hp_b",
                     "damage_a","damage_b","has_awp_a","has_awp_b","has_defuse_kit_a",
                     "bomb_carrier","buy_type_a","buy_type_b","clutch_side"):
                state.extra.pop(k, None)
            self._player_prev.pop(match_id, None)
            self._open_kill_fired.pop(match_id, None)

        # Bomb planted
        if is_bomb_planted and prev and not prev.get("is_bomb_planted", False):
            t_side = state.extra.get("team_a_current_side", "ct")
            team = "b" if t_side == "ct" else "a"
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="cs2", team=team,
                description="Bomb planted",
                impact=-0.05 if t_side == "ct" else 0.05,
                timestamp=now, match_state=state,
            ))
            events_emitted += 1

        # Kill streaks (3k+)
        if prev:
            prev_kills_a = self._sum_round_kills(prev.get("team_one") or {})
            prev_kills_b = self._sum_round_kills(prev.get("team_two") or {})

            if kills_a >= 3 and kills_a > prev_kills_a:
                self._emit(GameEvent(
                    event_type=EventType.KILL_STREAK,
                    match_id=match_id, game="cs2", team="a",
                    description=f"{kills_a}k by {state.team_a}",
                    impact=0.02 * kills_a, timestamp=now, match_state=state,
                ))
                events_emitted += 1

            if kills_b >= 3 and kills_b > prev_kills_b:
                self._emit(GameEvent(
                    event_type=EventType.KILL_STREAK,
                    match_id=match_id, game="cs2", team="b",
                    description=f"{kills_b}k by {state.team_b}",
                    impact=0.02 * kills_b, timestamp=now, match_state=state,
                ))
                events_emitted += 1

        # ─── GRANULAR PER-PLAYER EVENTS (KILL / OPEN_KILL / HP_DAMAGE) ──────
        # Compare prev player_state to current per-player state; emit fine-grained
        # events when kills increment, is_alive flips, or HP drops hard.
        prev_players = self._player_prev.get(match_id, {})
        cur_players: dict[str, dict] = {}
        # Reset per-round open-kill tracker when round score changes OR we see new kills
        round_key = f"{state.score_a}_{state.score_b}_{state.round_score_a}_{state.round_score_b}"
        if self._open_kill_fired.get(match_id) != round_key:
            # New round — allow OPEN_KILL once this round
            pass

        for team_letter, team_data, team_name, alive_opp in [
            ("a", team_one, state.team_a, stats_b["alive"]),
            ("b", team_two, state.team_b, stats_a["alive"]),
        ]:
            for p in team_data.get("player_states") or []:
                pid = str(p.get("id") or p.get("provider_id") or "")
                if not pid:
                    continue
                cur_kills = int(p.get("kills_in_round", 0) or 0)
                cur_hp = int(p.get("health", 0) or 0)
                cur_alive = bool(p.get("is_alive", True))
                cur_primary = p.get("primary_weapon", "") or ""
                cur = {"kills": cur_kills, "hp": cur_hp, "alive": cur_alive,
                       "weapon": cur_primary, "team": team_letter,
                       "nickname": p.get("nickname", "?")}
                cur_players[pid] = cur
                old = prev_players.get(pid)
                if not old:
                    continue  # first sighting, no delta

                # KILL event — per incremented kill
                kill_delta = cur_kills - old["kills"]
                if kill_delta > 0:
                    # Is this the first kill of this round? → OPEN_KILL
                    total_team_kills_before = 0
                    for pp_id, pp in prev_players.items():
                        total_team_kills_before += pp.get("kills", 0)
                    is_open = (total_team_kills_before == 0 and
                               self._open_kill_fired.get(match_id) != round_key)
                    for _ in range(kill_delta):
                        ev_type = EventType.OPEN_KILL if is_open else EventType.KILL
                        # alive_opp is post-snapshot; approx post-kill state
                        desc = (f"{p.get('nickname','?')} frag → {team_name} "
                                f"{stats_a['alive'] if team_letter=='a' else stats_b['alive']}v"
                                f"{stats_b['alive'] if team_letter=='a' else stats_a['alive']}")
                        impact = 0.015 if team_letter == "a" else -0.015
                        if is_open:
                            # Open kill is worth more — 60-70% round-win correlation
                            impact = 0.03 if team_letter == "a" else -0.03
                            self._open_kill_fired[match_id] = round_key
                            is_open = False  # only fire once per round
                        self._emit(GameEvent(
                            event_type=ev_type,
                            match_id=match_id, game="cs2", team=team_letter,
                            description=desc,
                            impact=impact, timestamp=now, match_state=state,
                        ))
                        events_emitted += 1

                # HP_DAMAGE event — significant damage without death
                hp_drop = old["hp"] - cur_hp
                if cur_alive and hp_drop >= 30:
                    opp_letter = "b" if team_letter == "a" else "a"
                    self._emit(GameEvent(
                        event_type=EventType.HP_DAMAGE,
                        match_id=match_id, game="cs2", team=opp_letter,
                        description=f"{p.get('nickname','?')} hit -{hp_drop} HP ({cur_hp} left)",
                        impact=0.003 * (hp_drop / 30) * (1 if opp_letter == "a" else -1),
                        timestamp=now, match_state=state,
                    ))
                    events_emitted += 1

                # PLAYER_DIED — alive flipped to false with no attributed kill delta
                # (rare: bomb detonation, suicide, teamkill)
                if old["alive"] and not cur_alive:
                    # Check if any teammate claimed a kill this tick — if not, it's unattributed
                    pass  # kill event above already covers the common case

        # Clutch detection: a team drops to exactly 1 alive vs 2+ opponents
        prev_alive_a = sum(1 for p in prev_players.values() if p["team"] == "a" and p["alive"])
        prev_alive_b = sum(1 for p in prev_players.values() if p["team"] == "b" and p["alive"])
        if prev_players:
            # 1vX clutch started (this tick dropped team to 1 alive with 2+ opps)
            if stats_a["alive"] == 1 and stats_b["alive"] >= 2 and prev_alive_a > 1:
                state.extra["clutch_side"] = "a"
            elif stats_b["alive"] == 1 and stats_a["alive"] >= 2 and prev_alive_b > 1:
                state.extra["clutch_side"] = "b"
            # Clutch won (team cleared opponents while outnumbered)
            clutch_side = state.extra.get("clutch_side")
            if clutch_side == "a" and stats_b["alive"] == 0 and stats_a["alive"] >= 1:
                self._emit(GameEvent(
                    event_type=EventType.CLUTCH_WIN,
                    match_id=match_id, game="cs2", team="a",
                    description=f"1vX clutch by {state.team_a}",
                    impact=0.05, timestamp=now, match_state=state,
                ))
                state.extra.pop("clutch_side", None)
                events_emitted += 1
            elif clutch_side == "b" and stats_a["alive"] == 0 and stats_b["alive"] >= 1:
                self._emit(GameEvent(
                    event_type=EventType.CLUTCH_WIN,
                    match_id=match_id, game="cs2", team="b",
                    description=f"1vX clutch by {state.team_b}",
                    impact=-0.05, timestamp=now, match_state=state,
                ))
                state.extra.pop("clutch_side", None)
                events_emitted += 1

        # Reset open-kill tracker when round ends (round score changed) so the
        # next round can emit its own OPEN_KILL. Handled naturally by round_key.
        if new_score_a != old_score_a or new_score_b != old_score_b:
            state.extra.pop("clutch_side", None)
        self._player_prev[match_id] = cur_players

        # Match end
        if match_status == "ended" or (game_ended and not state.is_live):
            pass  # handled by discovery loop removing stale matches
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed or state.score_b >= maps_needed:
            if state.is_live:
                state.is_live = False
                winner = "a" if state.score_a > state.score_b else "b"
                winner_name = state.team_a if winner == "a" else state.team_b
                logger.info(f"[MATCH END] {winner_name} wins! {state.score_a}-{state.score_b}")
                self._emit(GameEvent(
                    event_type=EventType.MATCH_END,
                    match_id=match_id, game="cs2", team=winner,
                    description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                    impact=1.0 if winner == "a" else -1.0,
                    timestamp=now, match_state=state,
                ))
                events_emitted += 1

        # Always update probability
        state.win_probability_a = self.estimate_win_probability(state)
        state.last_event_time = now

        return events_emitted

    def _sum_team_money(self, team_data: dict) -> int:
        total = 0
        for p in team_data.get("player_states") or []:
            total += p.get("balance", 0)
        return total

    def _sum_round_kills(self, team_data: dict) -> int:
        total = 0
        for p in team_data.get("player_states") or []:
            total += p.get("kills_in_round", 0)
        return total

    # Weapons considered "rifles" for full-buy classification
    _RIFLES = {"ak47", "m4a1", "m4a4", "m4a1_s", "aug", "sg553", "sg556", "famas", "galilar"}
    # Weapons considered AWP-class (high-impact)
    _AWPS = {"awp", "scar20", "g3sg1", "ssg08"}

    def _team_rollup(self, team_data: dict) -> dict:
        """Aggregate per-player state into team-level rollup for qwen context.

        Returns keys: alive, total_hp, avg_hp, damage, has_awp, has_defuse_kit, has_bomb.
        """
        players = team_data.get("player_states") or []
        alive = 0
        total_hp = 0
        damage = 0
        has_awp = False
        has_defuse_kit = False
        has_bomb = False
        for p in players:
            if p.get("is_alive"):
                alive += 1
                total_hp += int(p.get("health", 0) or 0)
            damage += int(p.get("health_damage_in_round", 0) or 0)
            wpn = (p.get("primary_weapon", "") or "").lower().replace("-", "").replace(" ", "")
            if wpn in self._AWPS:
                has_awp = True
            if p.get("has_defuse_kit"):
                has_defuse_kit = True
            if p.get("has_bomb"):
                has_bomb = True
        avg_hp = (total_hp // alive) if alive > 0 else 0
        return {
            "alive": alive, "total_hp": total_hp, "avg_hp": avg_hp,
            "damage": damage, "has_awp": has_awp,
            "has_defuse_kit": has_defuse_kit, "has_bomb": has_bomb,
        }

    def _classify_buy(self, team_data: dict) -> str:
        """Classify team's buy this round from equipment_value_round_start.

        Pistol round (round 1 or 13): < $2,500 total.
        Full buy: >= $20,000 with primary rifles/AWPs.
        Half buy: $12,000 - $20,000.
        Force buy: $5,000 - $12,000 with primaries purchased.
        Eco: < $5,000.
        Returns empty string if we can't classify yet.
        """
        players = team_data.get("player_states") or []
        if not players:
            return ""
        eq_total = 0
        has_primaries = 0
        for p in players:
            # equipment_value_round_start reflects what they started the round with
            eq_total += int(p.get("equipment_value_round_start", 0) or 0)
            wpn = (p.get("primary_weapon", "") or "").lower().replace("-", "").replace(" ", "")
            if wpn in self._RIFLES or wpn in self._AWPS:
                has_primaries += 1
        # Pistol round heuristic: no primaries and low equipment
        if eq_total < 2500 and has_primaries == 0:
            return "pistol"
        if eq_total >= 20000 and has_primaries >= 4:
            return "full_buy"
        if eq_total >= 12000 and has_primaries >= 3:
            return "half_buy"
        if eq_total >= 5000 and has_primaries >= 2:
            return "force_buy"
        if eq_total < 5000:
            return "eco"
        return "light_buy"
