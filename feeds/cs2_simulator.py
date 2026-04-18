"""
CS2 Match Simulator — generates realistic CS2 events for testing the full pipeline.

Simulates a live CS2 match with:
- Realistic round timing (1:30-2:30 per round)
- Economy cycles
- Side tracking
- Kill events, bomb plants
- Scoreboard updates with player money

Use this until HLTV/GRID live feeds are available.
"""
import asyncio
import logging
import random
import time
from typing import Optional

from feeds.base import GameFeed, GameEvent, MatchState, EventType
from feeds.cs2_model import CS2WinProbabilityModel, get_side_for_round

logger = logging.getLogger(__name__)

# Realistic CS2 team names from current pro scene
SAMPLE_MATCHES = [
    ("Natus Vincere", "FaZe Clan", 3),
    ("Team Vitality", "G2 Esports", 3),
    ("Team Spirit", "MOUZ", 3),
    ("Heroic", "BetBoom Team", 3),
    ("Cloud9", "Complexity", 1),
    ("GamerLegion", "9INE", 3),
]


class CS2SimulatorFeed(GameFeed):
    """Simulates live CS2 matches for testing."""

    def __init__(self, speed: float = 1.0):
        """
        Args:
            speed: Simulation speed multiplier. 1.0 = real-time (2 min rounds),
                   10.0 = 10x speed (12 sec rounds), etc.
        """
        super().__init__("cs2")
        self._speed = speed
        self._model = CS2WinProbabilityModel()
        self._sim_task = None
        self._subscribed_matches: set[str] = set()

    async def connect(self):
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="",
            game="cs2",
            team="neutral",
            description="CS2 Simulator connected",
            impact=0.0,
        ))
        logger.info("CS2 Simulator feed connected")

        # Auto-start a simulated match
        self._sim_task = asyncio.create_task(self._run_simulation())

    async def disconnect(self):
        self._running = False
        if self._sim_task:
            self._sim_task.cancel()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._subscribed_matches.add(match_id)

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._matches.keys())

    def estimate_win_probability(self, state: MatchState) -> float:
        return self._model.calculate(
            maps_a=state.score_a,
            maps_b=state.score_b,
            best_of=state.total_maps,
            round_score_a=state.round_score_a,
            round_score_b=state.round_score_b,
            team_a_side=state.extra.get("team_a_current_side", "ct"),
            team_a_start_side=state.extra.get("team_a_start_side", "ct"),
            team_a_money=state.extra.get("team_a_money", 0),
            team_b_money=state.extra.get("team_b_money", 0),
        )

    async def _run_simulation(self):
        """Run continuous match simulations."""
        match_num = 0
        while self._running:
            team_a, team_b, best_of = SAMPLE_MATCHES[match_num % len(SAMPLE_MATCHES)]
            match_id = f"SIM-{match_num + 1}"

            logger.info(f"[SIM] Starting match: {team_a} vs {team_b} (Bo{best_of})")
            await self._simulate_match(match_id, team_a, team_b, best_of)

            match_num += 1
            # Brief pause between matches
            await asyncio.sleep(5 / self._speed)

    async def _simulate_match(self, match_id: str, team_a: str, team_b: str, best_of: int):
        """Simulate a full CS2 match."""
        maps_needed = (best_of // 2) + 1

        state = MatchState(
            match_id=match_id,
            game="cs2",
            team_a=team_a,
            team_b=team_b,
            total_maps=best_of,
            is_live=True,
            started_at=time.time(),
            extra={
                "team_a_start_side": random.choice(["ct", "t"]),
                "team_a_current_side": "",
                "side_determined": True,
                "team_a_money": 800 * 5,
                "team_b_money": 800 * 5,
                "round_kills_a": 0,
                "round_kills_b": 0,
                "economy_history": [],
            },
        )
        state.extra["team_a_current_side"] = state.extra["team_a_start_side"]
        self._matches[match_id] = state
        self._subscribed_matches.add(match_id)

        self._emit(GameEvent(
            event_type=EventType.MATCH_START,
            match_id=match_id,
            game="cs2",
            team="neutral",
            description=f"{team_a} vs {team_b} (Bo{best_of}) LIVE",
            impact=0.0,
            match_state=state,
        ))

        # Simulate maps
        while state.score_a < maps_needed and state.score_b < maps_needed and self._running:
            await self._simulate_map(state)

        # Match end
        if self._running:
            state.is_live = False
            winner = "a" if state.score_a > state.score_b else "b"
            winner_name = team_a if winner == "a" else team_b
            self._emit(GameEvent(
                event_type=EventType.MATCH_END,
                match_id=match_id,
                game="cs2",
                team=winner,
                description=f"Match won by {winner_name} ({state.score_a}-{state.score_b})",
                impact=1.0 if winner == "a" else -1.0,
                match_state=state,
            ))

    async def _simulate_map(self, state: MatchState):
        """Simulate one map of a CS2 match."""
        state.round_score_a = 0
        state.round_score_b = 0
        state.extra["team_a_start_side"] = random.choice(["ct", "t"])
        state.extra["team_a_current_side"] = state.extra["team_a_start_side"]
        state.extra["team_a_money"] = 800 * 5
        state.extra["team_b_money"] = 800 * 5

        # Team strength — slight random advantage for variety
        team_a_skill = random.uniform(0.45, 0.55)

        round_num = 0
        while state.round_score_a < 13 and state.round_score_b < 13 and self._running:
            round_num += 1

            # Update side
            state.extra["team_a_current_side"] = get_side_for_round(
                state.extra["team_a_start_side"], round_num
            )

            # Round start
            state.extra["round_kills_a"] = 0
            state.extra["round_kills_b"] = 0
            self._emit(GameEvent(
                event_type=EventType.ROUND_START,
                match_id=state.match_id,
                game="cs2",
                team="neutral",
                description=f"Round {round_num} starting",
                impact=0.0,
                match_state=state,
            ))

            # Simulate round duration (15-25 seconds at 10x speed)
            round_duration = random.uniform(90, 150) / self._speed

            # Simulate kills during the round
            kills_this_round = random.randint(3, 10)
            kill_interval = round_duration / (kills_this_round + 1)

            for k in range(kills_this_round):
                await asyncio.sleep(kill_interval)
                if not self._running:
                    return

                # Determine killer based on skill + side + economy
                side = state.extra["team_a_current_side"]
                ct_bonus = 0.03 if side == "ct" else -0.03
                p_team_a_kill = team_a_skill + ct_bonus

                if random.random() < p_team_a_kill:
                    team = "a"
                    state.extra["round_kills_a"] = state.extra.get("round_kills_a", 0) + 1
                else:
                    team = "b"
                    state.extra["round_kills_b"] = state.extra.get("round_kills_b", 0) + 1

                kills = state.extra.get(f"round_kills_{team}", 0)
                if kills >= 3:
                    team_name = state.team_a if team == "a" else state.team_b
                    self._emit(GameEvent(
                        event_type=EventType.KILL_STREAK,
                        match_id=state.match_id,
                        game="cs2",
                        team=team,
                        description=f"{kills}k by {team_name} this round",
                        impact=0.02 * kills,
                        match_state=state,
                    ))

            # Bomb plant (30% chance if T side has advantage)
            t_team = "b" if state.extra["team_a_current_side"] == "ct" else "a"
            if random.random() < 0.4:
                self._emit(GameEvent(
                    event_type=EventType.OBJECTIVE_TAKEN,
                    match_id=state.match_id,
                    game="cs2",
                    team=t_team,
                    description="Bomb planted",
                    impact=-0.05 if t_team == "b" else 0.05,
                    match_state=state,
                ))
                await asyncio.sleep(3 / self._speed)

            # Determine round winner
            side = state.extra["team_a_current_side"]
            ct_win_rate = 0.53
            p_a = ct_win_rate if side == "ct" else (1 - ct_win_rate)
            p_a = p_a * (team_a_skill / 0.5)  # adjust for skill
            p_a = max(0.2, min(0.8, p_a))

            if random.random() < p_a:
                winning_team = "a"
                state.round_score_a += 1
            else:
                winning_team = "b"
                state.round_score_b += 1

            # Update economy (simplified)
            if winning_team == "a":
                state.extra["team_a_money"] = random.randint(20000, 40000)
                state.extra["team_b_money"] = random.randint(5000, 20000)
            else:
                state.extra["team_b_money"] = random.randint(20000, 40000)
                state.extra["team_a_money"] = random.randint(5000, 20000)

            # Economy event
            self._emit(GameEvent(
                event_type=EventType.ECONOMY_SHIFT,
                match_id=state.match_id,
                game="cs2",
                team=winning_team,
                description=f"Economy: {state.team_a}=${state.extra['team_a_money']:,} vs {state.team_b}=${state.extra['team_b_money']:,}",
                impact=0.02 if winning_team == "a" else -0.02,
                match_state=state,
            ))

            # Round end
            old_prob = state.win_probability_a
            state.win_probability_a = self.estimate_win_probability(state)
            impact = state.win_probability_a - old_prob

            winner_name = state.team_a if winning_team == "a" else state.team_b
            self._emit(GameEvent(
                event_type=EventType.ROUND_END,
                match_id=state.match_id,
                game="cs2",
                team=winning_team,
                description=f"Round to {winner_name} ({state.round_score_a}-{state.round_score_b})",
                impact=impact,
                match_state=state,
            ))

            state.last_event_time = time.time()

        # Handle overtime (simplified — just pick a winner)
        if state.round_score_a == 12 and state.round_score_b == 12:
            if random.random() < 0.5:
                state.round_score_a = 13
            else:
                state.round_score_b = 13

        # Map end
        if state.round_score_a >= 13:
            state.score_a += 1
            winning_team = "a"
        else:
            state.score_b += 1
            winning_team = "b"

        winner_name = state.team_a if winning_team == "a" else state.team_b
        old_prob = state.win_probability_a
        state.win_probability_a = self.estimate_win_probability(state)

        self._emit(GameEvent(
            event_type=EventType.MAP_WIN,
            match_id=state.match_id,
            game="cs2",
            team=winning_team,
            description=f"Map to {winner_name} (Series: {state.score_a}-{state.score_b})",
            impact=state.win_probability_a - old_prob,
            match_state=state,
        ))

        state.current_map += 1
