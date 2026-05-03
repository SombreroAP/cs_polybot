"""
Base classes for real-time esports game data feeds.
Each game implements this interface to provide normalized match events.
"""
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Types of in-game events that can shift match probability."""
    # Universal
    MATCH_START = "match_start"
    MATCH_END = "match_end"
    ROUND_START = "round_start"
    ROUND_END = "round_end"

    # Score changes
    SCORE_UPDATE = "score_update"           # any score change
    MAP_WIN = "map_win"                     # team wins a map/game in a series

    # Momentum shifts (game-specific mapped to universal)
    KILL_STREAK = "kill_streak"             # multi-kill, ace, etc.
    OBJECTIVE_TAKEN = "objective_taken"     # bomb plant, dragon, roshan, etc.
    ECONOMY_SHIFT = "economy_shift"         # eco round win, big buy, etc.
    TEAMFIGHT_WON = "teamfight_won"         # decisive teamfight

    # Granular per-tick events (CS2 — derived from bo3.gg player_state deltas)
    KILL = "kill"                           # single kill by any player
    OPEN_KILL = "open_kill"                 # first kill of the round
    PLAYER_DIED = "player_died"             # is_alive flipped, no kill attributed
    HP_DAMAGE = "hp_damage"                 # player took ≥30 HP without dying (big trade)
    CLUTCH_WIN = "clutch_win"               # 1vX clutch closed

    # Meta
    FEED_CONNECTED = "feed_connected"
    FEED_DISCONNECTED = "feed_disconnected"
    FEED_ERROR = "feed_error"


@dataclass
class MatchState:
    """Current state of a live match."""
    match_id: str
    game: str  # cs2, lol, dota2, valorant
    team_a: str
    team_b: str
    score_a: int = 0  # maps/games won
    score_b: int = 0
    current_map: int = 1
    total_maps: int = 3  # best of N
    round_score_a: int = 0  # rounds within current map
    round_score_b: int = 0
    is_live: bool = False
    started_at: float = 0.0
    win_probability_a: float = 0.5  # our estimate based on game state
    last_event_time: float = field(default_factory=time.time)
    extra: dict = field(default_factory=dict)  # game-specific data


@dataclass
class GameEvent:
    """A single game event with timestamp for latency measurement."""
    event_type: EventType
    match_id: str
    game: str
    team: str  # which team this event favors ("a", "b", or "neutral")
    description: str
    impact: float  # estimated probability shift (-1.0 to +1.0 for team A)
    timestamp: float = field(default_factory=time.time)  # when WE received this event
    source_timestamp: Optional[float] = None  # when the source says it happened (if available)
    match_state: Optional[MatchState] = None
    raw_data: Optional[dict] = None


# Type for event callback
EventCallback = Callable[[GameEvent], None]


class GameFeed(ABC):
    """
    Abstract base class for real-time game data feeds.
    Each game (CS2, LoL, Dota 2, Valorant) implements this.
    """

    def __init__(self, game: str):
        self.game = game
        self._callbacks: list[EventCallback] = []
        self._raw_snapshot_callbacks: list = []  # callback(match_id: str, raw_payload: dict)
        self._running = False
        self._matches: dict[str, MatchState] = {}
        self._event_count = 0
        self._connected = False
        self._connect_time: float = 0.0

    def on_event(self, callback: EventCallback):
        """Register a callback for game events."""
        self._callbacks.append(callback)

    def on_raw_snapshot(self, callback):
        """Register a callback that receives the RAW provider payload for each
        snapshot update. Used by the MatchRecorder so recordings preserve the
        full per-player / per-round detail the provider sends (instead of the
        minimal synthesized state we build for the trading loop).

        Callback signature: (match_id: str, raw_payload: dict) -> None
        """
        self._raw_snapshot_callbacks.append(callback)

    def _emit_raw_snapshot(self, match_id: str, raw_payload: dict):
        """Invoke raw-snapshot callbacks. Safe to call every message — errors
        in individual callbacks are swallowed so they can't break the feed."""
        for cb in self._raw_snapshot_callbacks:
            try:
                cb(match_id, raw_payload)
            except Exception as e:
                logger.error(f"Raw-snapshot callback error: {e}")

    def _emit(self, event: GameEvent):
        """Emit an event to all registered callbacks."""
        self._event_count += 1
        for cb in self._callbacks:
            try:
                cb(event)
            except Exception as e:
                # Include the originating callback + full traceback so we can
                # diagnose silent breakage. The bare error message has been
                # too opaque (e.g. "<= not supported between NoneType and int"
                # with no hint of where it came from).
                import traceback
                cb_name = getattr(cb, "__qualname__", getattr(cb, "__name__", repr(cb)))
                logger.error(
                    f"Event callback error in {cb_name}: {e}\n{traceback.format_exc()}"
                )

    @abstractmethod
    async def connect(self):
        """Connect to the game data source."""
        pass

    @abstractmethod
    async def disconnect(self):
        """Disconnect from the game data source."""
        pass

    @abstractmethod
    async def subscribe_match(self, match_id: str):
        """Subscribe to live updates for a specific match."""
        pass

    @abstractmethod
    async def unsubscribe_match(self, match_id: str):
        """Unsubscribe from a match."""
        pass

    @abstractmethod
    async def get_live_matches(self) -> list[MatchState]:
        """Get all currently live matches for this game."""
        pass

    def get_match_state(self, match_id: str) -> Optional[MatchState]:
        """Get current state of a tracked match."""
        return self._matches.get(match_id)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def event_count(self) -> int:
        return self._event_count

    def estimate_win_probability(self, state: MatchState) -> float:
        """
        Estimate team A's win probability based on current match state.
        This is a basic model — can be overridden per-game for better estimates.

        For a Bo3: if team A leads 1-0, they need 1 more map.
        Basic model: P(win series) based on map score + round score momentum.
        """
        maps_needed = (state.total_maps // 2) + 1

        if state.score_a >= maps_needed:
            return 1.0
        if state.score_b >= maps_needed:
            return 0.0

        # Base probability from map score
        # Simple model: treat each remaining map as 50/50
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        maps_remaining = a_needs + b_needs - 1

        # Binomial-ish: P(A wins) ≈ proportion of paths where A gets enough maps
        if maps_remaining <= 0:
            return 0.5

        # Use round score as momentum indicator for current map
        round_total = state.round_score_a + state.round_score_b
        if round_total > 0:
            round_momentum = state.round_score_a / round_total
        else:
            round_momentum = 0.5

        # Weighted: 70% map score model, 30% round momentum
        map_prob = b_needs / (a_needs + b_needs)  # simple ratio
        combined = 0.7 * map_prob + 0.3 * round_momentum

        raw = max(0.01, min(0.99, combined))
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw
