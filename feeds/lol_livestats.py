"""
League of Legends live in-game stats feed via lolesports live stats API.

Uses Riot's undocumented but publicly accessible live stats feed that powers
the lolesports.com broadcast overlay. Provides real-time data:
- Gold, kills, deaths, assists per player
- Champion picks per player
- Team total gold, kills
- Dragons (with types), barons, towers, inhibitors
- Creep score, level

Key events detected:
- TEAMFIGHT_WON: 3+ kills in one polling window (tradeable!)
- KILL_STREAK: 2 kills in one window (tradeable)
- OBJECTIVE_TAKEN: Dragon, Baron, Tower, Inhibitor
- MAP_WIN: Game won in series (from discovery loop)

Polls every 5 seconds during live games. No auth needed for stats feed.
"""
import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Optional

import aiohttp

from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

LOLESPORTS_API = "https://esports-api.lolesports.com/persisted/gw"
LOLESPORTS_KEY = "0TvQnueqKa5mxJntVWt0w4LpLfEkrV1Ta8rQBb9Z"
LIVESTATS_URL = "https://feed.lolesports.com/livestats/v1"


class LoLLiveStatsFeed(GameFeed):
    """Rich LoL live stats — gold, kills, dragons, barons, towers, teamfights."""

    POLL_INTERVAL = 5  # seconds between stat fetches
    DISCOVERY_INTERVAL = 15  # seconds between match discovery

    def __init__(self):
        super().__init__("lol")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        self._discovery_task = None
        self._subscribed_matches: set[str] = set()
        self._game_ids: dict[str, str] = {}  # match_id -> current game_id
        self._last_states: dict[str, dict] = {}  # game_id -> last frame data
        self._game_start_times: dict[str, float] = {}  # game_id -> first frame unix ts
        self._last_fetch_times: dict[str, str] = {}  # game_id -> last frame RFC timestamp

    async def connect(self):
        self._session = aiohttp.ClientSession(headers={
            "x-api-key": LOLESPORTS_KEY,
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
        })
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="lol", team="neutral",
            description="Connected to LoL Live Stats feed", impact=0.0,
        ))
        logger.info("LoL Live Stats feed connected")
        self._discovery_task = asyncio.create_task(self._safe_run(self._discovery_loop(), "lol_disc"))
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "lol_stats"))

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
        if self._discovery_task:
            self._discovery_task.cancel()
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
        """LoL win probability using gold lead + kills + objectives."""
        extra = state.extra

        # Series-level
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01

        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        series_prob = b_needs / (a_needs + b_needs)

        # In-game probability from gold lead
        gold_a = extra.get("gold_a", 0)
        gold_b = extra.get("gold_b", 0)
        game_minutes = extra.get("game_minutes", 0)

        if gold_a > 0 and gold_b > 0:
            gold_diff = gold_a - gold_b
            time_factor = max(0.5, min(2.0, game_minutes / 20)) if game_minutes > 0 else 0.5
            game_prob = 1.0 / (1.0 + math.exp(-gold_diff / (3000 / time_factor)))

            dragons_a = extra.get("dragons_a", 0)
            dragons_b = extra.get("dragons_b", 0)
            towers_a = extra.get("towers_a", 0)
            towers_b = extra.get("towers_b", 0)
            obj_diff = (dragons_a - dragons_b) * 0.03 + (towers_a - towers_b) * 0.02
            game_prob = max(0.05, min(0.95, game_prob + obj_diff))

            raw = 0.5 * series_prob + 0.5 * game_prob
        else:
            raw = series_prob

        raw = max(0.01, min(0.99, raw))

        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    # ─── Discovery Loop ──────────────────────────────────────────────────────

    async def _discovery_loop(self):
        """Discover live LoL matches from lolesports API."""
        while self._running:
            try:
                async with self._session.get(
                    f"{LOLESPORTS_API}/getLive",
                    params={"hl": "en-US"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        await asyncio.sleep(30)
                        continue
                    data = await resp.json()

                events = data.get("data", {}).get("schedule", {}).get("events", [])
                live_count = 0

                for e in events:
                    if e.get("type") != "match" or e.get("state") != "inProgress":
                        continue

                    match = e.get("match", {})
                    if not match:
                        continue

                    teams = match.get("teams", [])
                    if len(teams) < 2:
                        continue

                    mid = str(match.get("id", e.get("id", "")))
                    live_count += 1

                    # Find active game ID
                    games = match.get("games", [])
                    active_game = None
                    for g in games:
                        if g.get("state") == "inProgress" and g.get("id"):
                            active_game = g
                            break

                    if mid not in self._subscribed_matches:
                        t1 = teams[0].get("name", "Blue")
                        t2 = teams[1].get("name", "Red")
                        bo = match.get("strategy", {}).get("count", 1)
                        league = e.get("league", {}).get("name", "")

                        await self.subscribe_match(mid)
                        state = self._matches[mid]
                        state.team_a = t1
                        state.team_b = t2
                        state.total_maps = bo
                        state.score_a = teams[0].get("result", {}).get("gameWins", 0)
                        state.score_b = teams[1].get("result", {}).get("gameWins", 0)

                        logger.info(f"Subscribed: {t1} vs {t2} (Bo{bo}) [{league}] [LoL LiveStats]")
                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=mid, game="lol", team="neutral",
                            description=f"{t1} vs {t2} (Bo{bo}) LIVE [{league}]",
                            impact=0.0, match_state=state,
                        ))

                    # Track active game ID
                    if active_game:
                        old_game_id = self._game_ids.get(mid)
                        new_game_id = str(active_game["id"])
                        if old_game_id != new_game_id:
                            # New game started in series — reset in-game state
                            self._game_ids[mid] = new_game_id
                            if old_game_id:
                                self._last_states.pop(old_game_id, None)
                                self._game_start_times.pop(old_game_id, None)
                                self._last_fetch_times.pop(old_game_id, None)
                            # Clear in-game extra state for new game
                            state = self._matches.get(mid)
                            if state:
                                state.extra.pop("champions_a", None)
                                state.extra.pop("champions_b", None)
                            logger.info(f"[LoL] New game started: {mid} -> game_id={new_game_id}")

                    # Update series scores
                    state = self._matches.get(mid)
                    if state:
                        new_a = teams[0].get("result", {}).get("gameWins", 0)
                        new_b = teams[1].get("result", {}).get("gameWins", 0)
                        if new_a != state.score_a or new_b != state.score_b:
                            old_a, old_b = state.score_a, state.score_b
                            state.score_a = new_a
                            state.score_b = new_b
                            winner = "a" if new_a > old_a else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b

                            old_prob = state.win_probability_a
                            state.win_probability_a = self.estimate_win_probability(state)
                            impact = state.win_probability_a - old_prob

                            logger.info(f"[LoL MAP] {winner_name} wins game! Series: {new_a}-{new_b}")
                            self._emit(GameEvent(
                                event_type=EventType.MAP_WIN,
                                match_id=mid, game="lol", team=winner,
                                description=f"Game to {winner_name} (Series: {new_a}-{new_b})",
                                impact=impact, match_state=state,
                            ))

                # Remove ended matches
                live_ids = set()
                for e in events:
                    if e.get("type") == "match" and e.get("state") == "inProgress":
                        m = e.get("match", {})
                        live_ids.add(str(m.get("id", e.get("id", ""))))

                for mid in list(self._subscribed_matches):
                    if mid not in live_ids and mid in self._matches:
                        state = self._matches[mid]
                        if state.is_live:
                            state.is_live = False
                            winner = "a" if state.score_a > state.score_b else "b"
                            winner_name = state.team_a if winner == "a" else state.team_b
                            logger.info(f"[LoL MATCH END] {winner_name} wins {state.score_a}-{state.score_b}")
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=mid, game="lol", team=winner,
                                description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                                impact=1.0 if winner == "a" else -1.0,
                                match_state=state,
                            ))

                if live_count > 0 or not self._subscribed_matches:
                    logger.info(f"LoL LiveStats: {live_count} live, {len(self._subscribed_matches)} tracked")

            except Exception as e:
                logger.error(f"LoL LiveStats discovery error: {e}")

            await asyncio.sleep(self.DISCOVERY_INTERVAL)

    # ─── Live Stats Polling ──────────────────────────────────────────────────

    async def _poll_loop(self):
        """Poll live stats for active games every 5 seconds."""
        logger.info("LoL LiveStats poll loop started")
        while self._running:
            try:
                for mid, game_id in list(self._game_ids.items()):
                    state = self._matches.get(mid)
                    if not state or not state.is_live:
                        continue

                    try:
                        # Use startingTime to get latest frames
                        url = f"{LIVESTATS_URL}/window/{game_id}"
                        last_ts = self._last_fetch_times.get(game_id)
                        params = {}
                        if last_ts:
                            params["startingTime"] = last_ts

                        async with self._session.get(
                            url, params=params,
                            timeout=aiohttp.ClientTimeout(total=8),
                        ) as resp:
                            if resp.status == 400:
                                # startingTime too far ahead — fetch without it
                                self._last_fetch_times.pop(game_id, None)
                                async with self._session.get(
                                    url, timeout=aiohttp.ClientTimeout(total=8),
                                ) as resp2:
                                    if resp2.status != 200:
                                        continue
                                    data = await resp2.json()
                            elif resp.status != 200:
                                continue
                            else:
                                data = await resp.json()

                        frames = data.get("frames", [])
                        if not frames:
                            continue

                        last_frame = frames[-1]

                        # Store the latest timestamp for next fetch
                        self._last_fetch_times[game_id] = last_frame.get("rfc460Timestamp", "")

                        # Track game start time from first frame we ever see
                        if game_id not in self._game_start_times:
                            first_ts = frames[0].get("rfc460Timestamp", "")
                            if first_ts:
                                try:
                                    dt = datetime.fromisoformat(first_ts.replace("Z", "+00:00"))
                                    self._game_start_times[game_id] = dt.timestamp()
                                except (ValueError, TypeError):
                                    self._game_start_times[game_id] = time.time()

                        self._process_frame(mid, game_id, last_frame, data.get("gameMetadata", {}))

                    except Exception as e:
                        logger.debug(f"LoL LiveStats fetch failed for {game_id}: {e}")

            except Exception as e:
                logger.error(f"LoL LiveStats poll error: {e}")

            await asyncio.sleep(self.POLL_INTERVAL)

    def _process_frame(self, match_id: str, game_id: str, frame: dict, metadata: dict):
        """Process a live stats frame and emit events on significant changes."""
        state = self._matches.get(match_id)
        if not state:
            return

        blue = frame.get("blueTeam", {})
        red = frame.get("redTeam", {})

        new_gold_a = blue.get("totalGold", 0)
        new_gold_b = red.get("totalGold", 0)
        new_kills_a = blue.get("totalKills", 0)
        new_kills_b = red.get("totalKills", 0)
        new_dragons_a_list = blue.get("dragons", [])
        new_dragons_b_list = red.get("dragons", [])
        new_dragons_a = len(new_dragons_a_list)
        new_dragons_b = len(new_dragons_b_list)
        new_towers_a = blue.get("towers", 0)
        new_towers_b = red.get("towers", 0)
        new_barons_a = blue.get("barons", 0)
        new_barons_b = red.get("barons", 0)
        new_inhibs_a = blue.get("inhibitors", 0)
        new_inhibs_b = red.get("inhibitors", 0)

        # Calculate game time from frame timestamp
        ts = frame.get("rfc460Timestamp", "")
        game_minutes = 0.0
        if ts and game_id in self._game_start_times:
            try:
                dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                game_minutes = (dt.timestamp() - self._game_start_times[game_id]) / 60.0
            except (ValueError, TypeError):
                pass

        # Store champion picks from metadata (first frame only)
        if "champions_a" not in state.extra and metadata:
            blue_meta = metadata.get("blueTeamMetadata", {})
            red_meta = metadata.get("redTeamMetadata", {})
            blue_champs = [p.get("championId", "?") for p in blue_meta.get("participantMetadata", [])]
            red_champs = [p.get("championId", "?") for p in red_meta.get("participantMetadata", [])]
            if blue_champs:
                state.extra["champions_a"] = blue_champs
                state.extra["champions_b"] = red_champs
                state.extra["patch"] = metadata.get("patchVersion", "")

        prev = self._last_states.get(game_id, {})
        self._last_states[game_id] = {
            "gold_a": new_gold_a, "gold_b": new_gold_b,
            "kills_a": new_kills_a, "kills_b": new_kills_b,
            "dragons_a": new_dragons_a, "dragons_b": new_dragons_b,
            "towers_a": new_towers_a, "towers_b": new_towers_b,
            "barons_a": new_barons_a, "barons_b": new_barons_b,
            "inhibs_a": new_inhibs_a, "inhibs_b": new_inhibs_b,
        }

        # Update state with all data
        gold_lead = new_gold_a - new_gold_b
        state.extra.update({
            "gold_a": new_gold_a, "gold_b": new_gold_b,
            "gold_lead": gold_lead,
            "kills_a": new_kills_a, "kills_b": new_kills_b,
            "dragons_a": new_dragons_a, "dragons_b": new_dragons_b,
            "dragon_types_a": new_dragons_a_list,
            "dragon_types_b": new_dragons_b_list,
            "towers_a": new_towers_a, "towers_b": new_towers_b,
            "barons_a": new_barons_a, "barons_b": new_barons_b,
            "inhibitors_a": new_inhibs_a, "inhibitors_b": new_inhibs_b,
            "game_minutes": game_minutes,
        })

        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)
        state.last_event_time = time.time()

        if not prev:
            # First frame — initialize
            logger.info(
                f"[INIT] LoL LiveStats {state.team_a} vs {state.team_b} | "
                f"gold={new_gold_a:,}-{new_gold_b:,} kills={new_kills_a}-{new_kills_b} "
                f"({game_minutes:.0f}min)"
            )
            return

        # ─── Detect significant events ───────────────────────────────────

        old_kills_a = prev.get("kills_a", 0)
        old_kills_b = prev.get("kills_b", 0)
        old_dragons_a = prev.get("dragons_a", 0)
        old_dragons_b = prev.get("dragons_b", 0)
        old_towers_a = prev.get("towers_a", 0)
        old_towers_b = prev.get("towers_b", 0)
        old_barons_a = prev.get("barons_a", 0)
        old_barons_b = prev.get("barons_b", 0)
        old_inhibs_a = prev.get("inhibs_a", 0)
        old_inhibs_b = prev.get("inhibs_b", 0)

        # Kill changes → detect teamfights vs solo kills
        new_a_kills = new_kills_a - old_kills_a
        new_b_kills = new_kills_b - old_kills_b
        total_new_kills = new_a_kills + new_b_kills

        if total_new_kills >= 3:
            # ─── TEAMFIGHT — 3+ total kills in one polling window ─────
            # This is the KEY tradeable event for LoL
            if new_a_kills > new_b_kills:
                team = "a"
                team_name = state.team_a
            elif new_b_kills > new_a_kills:
                team = "b"
                team_name = state.team_b
            else:
                team = "a" if gold_lead > 0 else "b"
                team_name = state.team_a if team == "a" else state.team_b

            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.TEAMFIGHT_WON,
                match_id=match_id, game="lol", team=team,
                description=(
                    f"Teamfight: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} "
                    f"(+{new_a_kills}v{new_b_kills}) | Gold: {gold_lead:+,} | {game_minutes:.0f}min"
                ),
                impact=state.win_probability_a - old_prob,
                match_state=state,
            ))
            logger.info(
                f"[LoL TEAMFIGHT] {team_name} wins (+{new_a_kills}v{new_b_kills}) | "
                f"Kills: {new_kills_a}-{new_kills_b} | Gold: {gold_lead:+,} | {game_minutes:.0f}min"
            )

        elif total_new_kills == 2:
            # 2 kills — kill streak (tradeable)
            if new_a_kills > new_b_kills:
                team = "a"
                team_name = state.team_a
            elif new_b_kills > new_a_kills:
                team = "b"
                team_name = state.team_b
            else:
                team = "a" if gold_lead > 0 else "b"
                team_name = state.team_a if team == "a" else state.team_b

            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.KILL_STREAK,
                match_id=match_id, game="lol", team=team,
                description=(
                    f"Kills: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} "
                    f"(+{new_a_kills}v{new_b_kills}) | Gold: {gold_lead:+,} | {game_minutes:.0f}min"
                ),
                impact=state.win_probability_a - old_prob,
                match_state=state,
            ))

        elif total_new_kills == 1:
            # Single kill — score_update (tracked but not directly tradeable)
            team = "a" if new_a_kills > 0 else "b"
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.SCORE_UPDATE,
                match_id=match_id, game="lol", team=team,
                description=(
                    f"Kill: {state.team_a} {new_kills_a}-{new_kills_b} {state.team_b} "
                    f"| Gold: {gold_lead:+,} | {game_minutes:.0f}min"
                ),
                impact=state.win_probability_a - old_prob,
                match_state=state,
            ))

        # Dragon taken
        if new_dragons_a > old_dragons_a:
            dragon_type = new_dragons_a_list[-1] if new_dragons_a_list else "unknown"
            is_soul = new_dragons_a >= 4
            label = "DRAGON SOUL" if is_soul else f"Dragon ({dragon_type})"
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="a",
                description=f"{label} to {state.team_a} ({new_dragons_a} total) | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=0.06 if is_soul else 0.03, match_state=state,
            ))
            logger.info(f"[LoL] {label} to {state.team_a}")

        elif new_dragons_b > old_dragons_b:
            dragon_type = new_dragons_b_list[-1] if new_dragons_b_list else "unknown"
            is_soul = new_dragons_b >= 4
            label = "DRAGON SOUL" if is_soul else f"Dragon ({dragon_type})"
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="b",
                description=f"{label} to {state.team_b} ({new_dragons_b} total) | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=-(0.06 if is_soul else 0.03), match_state=state,
            ))
            logger.info(f"[LoL] {label} to {state.team_b}")

        # Baron taken — HUGE event
        if new_barons_a > old_barons_a:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="a",
                description=f"BARON to {state.team_a}! | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=0.08, match_state=state,
            ))
            logger.info(f"[LoL BARON] {state.team_a} takes Baron! Gold: {gold_lead:+,}")

        elif new_barons_b > old_barons_b:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="b",
                description=f"BARON to {state.team_b}! | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=-0.08, match_state=state,
            ))
            logger.info(f"[LoL BARON] {state.team_b} takes Baron! Gold: {gold_lead:+,}")

        # Tower destroyed
        if new_towers_a > old_towers_a:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="a",
                description=f"Tower to {state.team_a} ({new_towers_a}-{new_towers_b}) | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=0.02, match_state=state,
            ))
        if new_towers_b > old_towers_b:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="b",
                description=f"Tower to {state.team_b} ({new_towers_a}-{new_towers_b}) | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=-0.02, match_state=state,
            ))

        # Inhibitor destroyed — late game, team about to win
        if new_inhibs_a > old_inhibs_a:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="a",
                description=f"INHIBITOR to {state.team_a}! | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=0.05, match_state=state,
            ))
            logger.info(f"[LoL INHIB] {state.team_a} destroys inhibitor!")

        if new_inhibs_b > old_inhibs_b:
            self._event_count += 1
            self._emit(GameEvent(
                event_type=EventType.OBJECTIVE_TAKEN,
                match_id=match_id, game="lol", team="b",
                description=f"INHIBITOR to {state.team_b}! | Gold: {gold_lead:+,} | {game_minutes:.0f}min",
                impact=-0.05, match_state=state,
            ))
            logger.info(f"[LoL INHIB] {state.team_b} destroys inhibitor!")
