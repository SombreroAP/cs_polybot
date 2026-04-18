"""
CS2 live match feed via HLTV Scorebot WebSocket.

HLTV's scorebot provides real-time round-by-round updates for CS2 matches.
Connect via Socket.IO, send readyForMatch with a match ID, and receive events.

Latency: typically 2-5 seconds behind the live game server.

This feed tracks:
- Side assignments (CT/T) per team, flipping at halftime/OT
- Team economy from scoreboard player data
- Kill events and multi-kill streaks
- Round, map, and match outcomes
"""
import asyncio
import json
import logging
import re
import time
from typing import Optional

import aiohttp
import websockets

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType
from feeds.cs2_model import CS2WinProbabilityModel, classify_buy, get_side_for_round

logger = logging.getLogger(__name__)


class CS2HLTVFeed(GameFeed):
    """Real-time CS2 match data from HLTV Scorebot."""

    def __init__(self):
        super().__init__("cs2")
        self._ws = None
        self._subscribed_matches: set[str] = set()
        self._reconnect_delay = 1.0
        self._keepalive_task = None
        self._model = CS2WinProbabilityModel()

    async def connect(self):
        """Connect to HLTV Scorebot WebSocket."""
        try:
            self._ws = await websockets.connect(
                config.HLTV_SCOREBOT_URL,
                additional_headers={
                    "Origin": "https://www.hltv.org",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                },
                ping_interval=25,
                ping_timeout=10,
            )
            self._connected = True
            self._connect_time = time.time()
            self._reconnect_delay = 1.0

            self._emit(GameEvent(
                event_type=EventType.FEED_CONNECTED,
                match_id="",
                game="cs2",
                team="neutral",
                description="Connected to HLTV Scorebot",
                impact=0.0,
            ))

            logger.info("Connected to HLTV Scorebot")
            self._keepalive_task = asyncio.create_task(self._keepalive_loop())
            await self._listen()

        except Exception as e:
            logger.error(f"HLTV connection failed: {e}")
            self._connected = False
            self._emit(GameEvent(
                event_type=EventType.FEED_DISCONNECTED,
                match_id="",
                game="cs2",
                team="neutral",
                description=f"HLTV connection failed: {e}",
                impact=0.0,
            ))

    async def disconnect(self):
        self._running = False
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._ws:
            await self._ws.close()
        self._connected = False
        logger.info("Disconnected from HLTV Scorebot")

    async def _keepalive_loop(self):
        while self._connected:
            try:
                await asyncio.sleep(25)
                if self._ws and self._connected:
                    await self._ws.send("2")
            except Exception:
                break

    async def subscribe_match(self, match_id: str):
        """Subscribe to live updates for a CS2 match on HLTV."""
        if not self._ws or not self._connected:
            logger.error("Cannot subscribe — not connected to HLTV")
            return

        msg = f'42["readyForMatch","{match_id}"]'
        await self._ws.send(msg)
        self._subscribed_matches.add(match_id)

        self._matches[match_id] = MatchState(
            match_id=match_id,
            game="cs2",
            team_a="",
            team_b="",
            is_live=True,
            started_at=time.time(),
            extra={
                "team_a_start_side": "",   # determined from first round
                "team_a_current_side": "",
                "side_determined": False,
                "team_a_money": 0,
                "team_b_money": 0,
                "round_kills_a": 0,
                "round_kills_b": 0,
                "economy_history": [],
            },
        )

        logger.info(f"Subscribed to CS2 match {match_id}")

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        """Return tracked live matches."""
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        """
        Discover live CS2 match IDs from HLTV matches page.
        Parses match IDs from URLs like /matches/2375000/team-a-vs-team-b-tournament.
        """
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    config.HLTV_MATCHES_URL,
                    headers={
                        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                        "Accept": "text/html",
                    },
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status != 200:
                        logger.warning(f"HLTV matches page returned {resp.status}")
                        return []
                    html = await resp.text()

            # Extract match IDs from live match links
            # HLTV uses data-livescore-match attribute or /matches/<id>/ URL pattern
            match_ids = set()

            # Pattern 1: data-livescore-match="<id>"
            for m in re.finditer(r'data-livescore-match="(\d+)"', html):
                match_ids.add(m.group(1))

            # Pattern 2: /matches/<id>/ in links within live section
            # Look for matches in the "live" section of the page
            live_section = re.search(r'liveMatchesSection.*?</div>\s*</div>\s*</div>', html, re.DOTALL)
            if live_section:
                for m in re.finditer(r'/matches/(\d+)/', live_section.group()):
                    match_ids.add(m.group(1))

            # Pattern 3: general match link pattern as fallback
            if not match_ids:
                for m in re.finditer(r'href="/matches/(\d+)/[^"]*"[^>]*class="[^"]*live', html):
                    match_ids.add(m.group(1))

            logger.info(f"HLTV discovery: found {len(match_ids)} live matches")
            return list(match_ids)

        except Exception as e:
            logger.error(f"HLTV match discovery failed: {e}")
            return []

    # ─── Win Probability (override base) ──────────────────────────────────────

    def estimate_win_probability(self, state: MatchState) -> float:
        """CS2-specific win probability using the recursive model."""
        return self._model.calculate(
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

    # ─── Side Tracking ────────────────────────────────────────────────────────

    def _get_current_side(self, state: MatchState) -> str:
        """Determine team_a's current side based on start side and round count."""
        start_side = state.extra.get("team_a_start_side", "")
        if not start_side:
            return "ct"  # default assumption until determined

        total_rounds = state.round_score_a + state.round_score_b
        current_round = total_rounds + 1
        return get_side_for_round(start_side, current_round)

    def _determine_starting_side(self, state: MatchState, winner_side: str, winning_team: str):
        """
        Determine team_a's starting side from the first round result.
        If team_a won round 1 and the winner was "CT", then team_a started CT.
        """
        if state.extra.get("side_determined"):
            return

        total_rounds = state.round_score_a + state.round_score_b
        if total_rounds > 1:
            # Can only reliably determine from round 1
            return

        if winning_team == "a":
            # team_a won, and winner_side tells us which side won
            state.extra["team_a_start_side"] = winner_side.lower()
        else:
            # team_b won, so team_a was on the OTHER side
            state.extra["team_a_start_side"] = "t" if winner_side.lower() == "ct" else "ct"

        state.extra["side_determined"] = True
        state.extra["team_a_current_side"] = state.extra["team_a_start_side"]
        logger.info(
            f"Side determined: {state.team_a} started on "
            f"{state.extra['team_a_start_side'].upper()} side"
        )

    # ─── Message Handling ─────────────────────────────────────────────────────

    async def _listen(self):
        self._running = True
        while self._running and self._ws:
            try:
                message = await self._ws.recv()
                await self._handle_message(message)
            except websockets.ConnectionClosed:
                logger.warning("HLTV WebSocket closed")
                self._connected = False
                await self._attempt_reconnect()
                break
            except Exception as e:
                logger.error(f"HLTV message error: {e}")

    async def _attempt_reconnect(self):
        while self._running:
            logger.info(f"Reconnecting to HLTV in {self._reconnect_delay:.1f}s...")
            await asyncio.sleep(self._reconnect_delay)
            self._reconnect_delay = min(self._reconnect_delay * 2, 30.0)
            try:
                await self.connect()
                for match_id in list(self._subscribed_matches):
                    await self.subscribe_match(match_id)
                return
            except Exception as e:
                logger.error(f"Reconnection failed: {e}")

    async def _handle_message(self, raw: str):
        """Parse Socket.IO message from HLTV and emit events."""
        if raw == "3":
            return
        if not raw.startswith("42"):
            return

        try:
            payload = json.loads(raw[2:])
            if not isinstance(payload, list) or len(payload) < 2:
                return

            event_name = payload[0]
            event_data = payload[1] if isinstance(payload[1], dict) else {}
            now = time.time()

            handler = {
                "scoreboard": self._handle_scoreboard,
                "roundEnd": self._handle_round_end,
                "mapEnd": self._handle_map_end,
                "matchEnd": self._handle_match_end,
                "kill": self._handle_kill,
                "bombPlanted": lambda d, t: self._handle_bomb_event(d, "planted", t),
                "bombDefused": lambda d, t: self._handle_bomb_event(d, "defused", t),
                "roundStart": self._handle_round_start,
            }.get(event_name)

            if handler:
                handler(event_data, now)

        except (json.JSONDecodeError, IndexError, KeyError) as e:
            logger.debug(f"Failed to parse HLTV message: {e}")

    def _find_match(self, data: dict) -> Optional[MatchState]:
        match_id = str(data.get("matchId", ""))
        if match_id in self._matches:
            return self._matches[match_id]
        if len(self._matches) == 1:
            return next(iter(self._matches.values()))
        return None

    # ─── Event Handlers ───────────────────────────────────────────────────────

    def _handle_scoreboard(self, data: dict, now: float):
        """
        Handle scoreboard update — team names, scores, player stats, economy.
        This is the richest data event from HLTV.
        """
        state = self._find_match(data)
        if not state:
            return

        teams = data.get("teams", [])
        if len(teams) >= 2:
            # Set team names on first scoreboard
            if not state.team_a:
                state.team_a = teams[0].get("name", "Team A")
            if not state.team_b:
                state.team_b = teams[1].get("name", "Team B")

            # Extract economy from player data
            team_a_money = self._extract_team_money(teams[0])
            team_b_money = self._extract_team_money(teams[1])

            old_a_buy = classify_buy(state.extra.get("team_a_money", 0))
            old_b_buy = classify_buy(state.extra.get("team_b_money", 0))

            state.extra["team_a_money"] = team_a_money
            state.extra["team_b_money"] = team_b_money

            new_a_buy = classify_buy(team_a_money)
            new_b_buy = classify_buy(team_b_money)

            # Detect significant economy shifts
            if (team_a_money > 0 and team_b_money > 0 and
                    (old_a_buy != new_a_buy or old_b_buy != new_b_buy) and
                    old_a_buy != "unknown" and old_b_buy != "unknown"):

                # Economy changed — this affects round win probability
                old_prob = state.win_probability_a
                state.win_probability_a = self.estimate_win_probability(state)
                impact = state.win_probability_a - old_prob

                if abs(impact) > 0.02:
                    self._emit(GameEvent(
                        event_type=EventType.ECONOMY_SHIFT,
                        match_id=state.match_id,
                        game="cs2",
                        team="a" if impact > 0 else "b",
                        description=(
                            f"Economy shift: {state.team_a}={new_a_buy} "
                            f"vs {state.team_b}={new_b_buy}"
                        ),
                        impact=impact,
                        timestamp=now,
                        match_state=state,
                    ))

        # Update round scores — map to correct team based on side tracking
        ct_score = data.get("ctScore", 0)
        t_score = data.get("tScore", 0)
        self._update_round_scores(state, ct_score, t_score)

        state.last_event_time = now
        state.win_probability_a = self.estimate_win_probability(state)

    def _extract_team_money(self, team_data: dict) -> int:
        """Sum player money from a team's scoreboard data."""
        total = 0
        players = team_data.get("players", [])
        if isinstance(players, list):
            for player in players:
                if isinstance(player, dict):
                    total += int(player.get("money", 0))
        return total

    def _update_round_scores(self, state: MatchState, ct_score: int, t_score: int):
        """Map CT/T scores to team_a/team_b based on side tracking."""
        current_side = state.extra.get("team_a_current_side", "")

        if current_side == "ct":
            state.round_score_a = ct_score
            state.round_score_b = t_score
        elif current_side == "t":
            state.round_score_a = t_score
            state.round_score_b = ct_score
        else:
            # Side not yet determined — use raw scores as placeholder
            state.round_score_a = ct_score
            state.round_score_b = t_score

    def _handle_round_end(self, data: dict, now: float):
        """Handle end of round — the most important event for probability shifts."""
        state = self._find_match(data)
        if not state:
            return

        winner_side = data.get("winner", "")  # "CT" or "T"

        # Determine which team won based on side tracking
        current_side = state.extra.get("team_a_current_side", "")
        if current_side:
            if winner_side.lower() == current_side:
                winning_team = "a"
            else:
                winning_team = "b"
        else:
            # Side unknown — infer from score direction
            winning_team = "a"  # default, will be corrected

        # Try to determine starting side from round 1
        self._determine_starting_side(state, winner_side, winning_team)

        # Update round scores
        if winning_team == "a":
            state.round_score_a += 1
        else:
            state.round_score_b += 1

        # Update current side based on round count
        if state.extra.get("team_a_start_side"):
            state.extra["team_a_current_side"] = self._get_current_side(state)

        # Record economy for this round
        state.extra.setdefault("economy_history", []).append({
            "round": state.round_score_a + state.round_score_b,
            "team_a_money": state.extra.get("team_a_money", 0),
            "team_b_money": state.extra.get("team_b_money", 0),
            "winner": winning_team,
        })

        state.last_event_time = now
        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)
        impact = state.win_probability_a - old_prob

        winner_name = state.team_a if winning_team == "a" else state.team_b
        self._emit(GameEvent(
            event_type=EventType.ROUND_END,
            match_id=state.match_id,
            game="cs2",
            team=winning_team,
            description=(
                f"Round to {winner_name} "
                f"({state.round_score_a}-{state.round_score_b})"
            ),
            impact=impact,
            timestamp=now,
            match_state=state,
            raw_data=data,
        ))

    def _handle_map_end(self, data: dict, now: float):
        """Handle end of map — major probability shift."""
        state = self._find_match(data)
        if not state:
            return

        if state.round_score_a > state.round_score_b:
            state.score_a += 1
            winning_team = "a"
        else:
            state.score_b += 1
            winning_team = "b"

        # Reset for next map
        state.round_score_a = 0
        state.round_score_b = 0
        state.current_map += 1
        state.extra["team_a_start_side"] = ""
        state.extra["side_determined"] = False
        state.extra["team_a_current_side"] = ""
        state.extra["team_a_money"] = 0
        state.extra["team_b_money"] = 0
        state.extra["round_kills_a"] = 0
        state.extra["round_kills_b"] = 0
        state.extra["economy_history"] = []
        state.last_event_time = now

        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)
        impact = state.win_probability_a - old_prob

        winner_name = state.team_a if winning_team == "a" else state.team_b
        self._emit(GameEvent(
            event_type=EventType.MAP_WIN,
            match_id=state.match_id,
            game="cs2",
            team=winning_team,
            description=(
                f"Map {state.current_map - 1} to {winner_name} "
                f"(Series: {state.score_a}-{state.score_b})"
            ),
            impact=impact,
            timestamp=now,
            match_state=state,
            raw_data=data,
        ))

    def _handle_match_end(self, data: dict, now: float):
        state = self._find_match(data)
        if not state:
            return

        state.is_live = False
        state.last_event_time = now
        winner = "a" if state.score_a > state.score_b else "b"
        winner_name = state.team_a if winner == "a" else state.team_b

        self._emit(GameEvent(
            event_type=EventType.MATCH_END,
            match_id=state.match_id,
            game="cs2",
            team=winner,
            description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
            impact=1.0 if winner == "a" else -1.0,
            timestamp=now,
            match_state=state,
            raw_data=data,
        ))

    def _handle_kill(self, data: dict, now: float):
        """Track kills per team per round, emit events for multi-kills (3k+)."""
        state = self._find_match(data)
        if not state:
            return

        killer_side = data.get("killerSide", "").lower()  # "ct" or "t"
        current_side = state.extra.get("team_a_current_side", "")

        if not current_side:
            return  # can't map kills without side info

        if killer_side == current_side:
            team = "a"
            state.extra["round_kills_a"] = state.extra.get("round_kills_a", 0) + 1
            kills = state.extra["round_kills_a"]
        else:
            team = "b"
            state.extra["round_kills_b"] = state.extra.get("round_kills_b", 0) + 1
            kills = state.extra["round_kills_b"]

        # Emit for significant streaks (3k+)
        if kills >= 3:
            team_name = state.team_a if team == "a" else state.team_b
            self._emit(GameEvent(
                event_type=EventType.KILL_STREAK,
                match_id=state.match_id,
                game="cs2",
                team=team,
                description=f"{kills}k by {team_name} this round",
                impact=0.02 * kills,
                timestamp=now,
                match_state=state,
                raw_data=data,
            ))

    def _handle_bomb_event(self, data: dict, event: str, now: float):
        state = self._find_match(data)
        if not state:
            return

        # Bomb planted favors T side, defused favors CT
        current_side = state.extra.get("team_a_current_side", "ct")
        if event == "planted":
            # T side planted — favors the T team
            team = "b" if current_side == "ct" else "a"
            impact = -0.05 if current_side == "ct" else 0.05
            desc = "Bomb planted"
        else:
            # CT defused — favors CT team
            team = "a" if current_side == "ct" else "b"
            impact = 0.05 if current_side == "ct" else -0.05
            desc = "Bomb defused"

        self._emit(GameEvent(
            event_type=EventType.OBJECTIVE_TAKEN,
            match_id=state.match_id,
            game="cs2",
            team=team,
            description=desc,
            impact=impact,
            timestamp=now,
            match_state=state,
            raw_data=data,
        ))

    def _handle_round_start(self, data: dict, now: float):
        state = self._find_match(data)
        if not state:
            return

        # Reset per-round kill counters
        state.extra["round_kills_a"] = 0
        state.extra["round_kills_b"] = 0

        # Update current side
        if state.extra.get("team_a_start_side"):
            state.extra["team_a_current_side"] = self._get_current_side(state)

        round_num = state.round_score_a + state.round_score_b + 1
        self._emit(GameEvent(
            event_type=EventType.ROUND_START,
            match_id=state.match_id,
            game="cs2",
            team="neutral",
            description=f"Round {round_num} starting",
            impact=0.0,
            timestamp=now,
            match_state=state,
        ))
