"""
Latency Analyzer — the core edge detection engine.

Compares real-time game feed events against Polymarket order book state
to identify windows where the market hasn't yet reacted to in-game events.

The key insight: when a significant in-game event occurs (round win, map win,
teamfight), the game feed receives it 2-15 seconds before the general public
(watching delayed Twitch/YouTube streams). During this window, Polymarket odds
are stale — they reflect pre-event probabilities.

This module:
1. Receives game events from feeds
2. Checks current Polymarket odds for the same match
3. Measures the gap between our probability estimate and market price
4. Generates trade signals when the edge exceeds thresholds
"""
import asyncio
import time
import logging
from collections import deque
from dataclasses import dataclass, field
from team_match import team_in_text
from typing import Optional

import config
from feeds.base import GameEvent, EventType
from market import MarketFinder, EsportsMarket

logger = logging.getLogger(__name__)


@dataclass
class LatencyMeasurement:
    """Record of a detected latency gap."""
    match_id: str
    game: str
    event_type: str
    event_time: float           # when we received the game event
    market_check_time: float    # when we checked Polymarket
    market_price_team_a: float  # Polymarket's price for team A
    our_estimate_team_a: float  # our probability estimate for team A
    price_gap: float            # absolute difference
    odds_staleness: float       # seconds since Polymarket price last moved
    latency_edge_ms: float      # how fast we got the event vs market reaction


@dataclass
class TradeSignal:
    """A signal to place a trade based on detected latency edge."""
    match_id: str
    game: str
    market: EsportsMarket
    team: str                   # "a" or "b"
    team_name: str
    token_id: str
    confidence: float           # 0-1
    our_price: float            # what we think the probability is
    market_price: float         # what Polymarket says
    edge: float                 # our_price - market_price
    latency_edge_ms: float
    bet_amount: float
    reason: str
    timestamp: float = field(default_factory=time.time)


class LatencyAnalyzer:
    """
    Detects and measures latency gaps between game feeds and Polymarket.
    Generates trade signals when actionable edges are found.
    """

    # Minimum probability gap to consider trading
    MIN_EDGE = 0.01  # 1 cent — aggressive arb mode, trade volume

    # Events that typically cause significant probability shifts
    HIGH_IMPACT_EVENTS = {
        EventType.MAP_WIN,
        EventType.MATCH_END,
    }

    MEDIUM_IMPACT_EVENTS = {
        EventType.ROUND_END,
        EventType.SCORE_UPDATE,
        EventType.OBJECTIVE_TAKEN,
        EventType.TEAMFIGHT_WON,
        EventType.ECONOMY_SHIFT,
        EventType.KILL_STREAK,
        EventType.OPEN_KILL,        # first kill of round — high correlation with round win
        EventType.CLUTCH_WIN,       # 1vX closed
    }

    # AI-driven mode: qwen evaluates every event type. Only exclude meta events
    # (FEED_CONNECTED / FEED_DISCONNECTED / FEED_ERROR) which have no trading context.
    EDGE_TRADEABLE_EVENTS = {
        EventType.MAP_WIN,
        EventType.MATCH_START,
        EventType.MATCH_END,
        EventType.ROUND_START,
        EventType.ROUND_END,
        EventType.TEAMFIGHT_WON,
        EventType.KILL_STREAK,
        EventType.OBJECTIVE_TAKEN,
        EventType.ECONOMY_SHIFT,    # CS2 buy/eco signals
        EventType.SCORE_UPDATE,
        # Granular per-tick events (CS2, from bo3.gg player_state deltas)
        EventType.KILL,             # single frag
        EventType.OPEN_KILL,        # first frag of round
        EventType.HP_DAMAGE,        # big damage without death
        EventType.CLUTCH_WIN,       # 1vX win
        EventType.PLAYER_DIED,      # unattributed death
    }

    # CS2 only (Dota2 disabled — awaiting GRID data access)
    EDGE_GAMES = {"cs2"}

    # Minimum seconds between CLOB price fetches for the same market
    PRICE_FETCH_INTERVAL = 5.0

    def __init__(self, market_finder: MarketFinder):
        self.market_finder = market_finder
        self._measurements: deque[LatencyMeasurement] = deque(maxlen=1000)
        self._signals: deque[TradeSignal] = deque(maxlen=100)
        self._match_to_market: dict[str, EsportsMarket] = {}  # primary (series preferred)
        self._match_to_map_winner: dict[str, EsportsMarket] = {}  # secondary map_winner
        self._signal_callbacks: list = []
        self._state_callbacks: list = []  # called after Claude decisions to refresh dashboard
        self._last_price_fetch: dict[str, float] = {}
        self._claude_analyst = None
        self._executor = None  # set by bot for position checks
        self._use_claude = False
        self._match_decisions: dict[str, dict] = {}  # match_id -> last Claude decision
        self._ws = None  # PolymarketWebSocket — set by bot for real-time prices
        self._edge_analyst = None  # EdgeAnalyst — set by bot for Claude decisions
        self._event_windows: dict[str, list] = {}  # match_id -> recent events window

    def on_state_update(self, callback):
        """Register callback to refresh dashboard state after async Claude decisions."""
        self._state_callbacks.append(callback)

    def _notify_state_update(self):
        for cb in self._state_callbacks:
            try:
                cb()
            except Exception:
                pass

    def enable_claude(self, analyst, executor=None):
        """Enable Claude AI expert panel for confidence scoring."""
        self._claude_analyst = analyst
        self._executor = executor
        self._use_claude = True
        logger.info("Claude AI analyst enabled for confidence scoring")

    def _should_use_claude(self, event, edge: float) -> bool:
        """Decide whether this event warrants a Claude API call (cost control)."""
        if not self._use_claude or not self._claude_analyst:
            return False
        if event.event_type not in self.HIGH_IMPACT_EVENTS | self.MEDIUM_IMPACT_EVENTS:
            return False

        # Rate-limit synthetic/market-only matches — analyze at most once every 5 minutes.
        # These have no real game data, so frequent calls waste tokens.
        if event.match_id.startswith("mkt_") and event.event_type not in self.HIGH_IMPACT_EVENTS:
            last_call = self._claude_analyst._match_call_counts.get(event.match_id, 0)
            last_decision_time = self._match_decisions.get(event.match_id, {}).get("time", 0)
            if time.time() - last_decision_time < 120:  # 2 minutes between calls for synthetic
                return False

        # Always call Claude for high-impact events (MAP_WIN, MATCH_END)
        # even without edge — panel can bet on pure conviction
        if event.event_type in self.HIGH_IMPACT_EVENTS:
            return True
        # For medium-impact events, require minimum edge
        if abs(edge) < config.CLAUDE_MIN_EDGE_FOR_CALL:
            return False
        return True

    def on_signal(self, callback):
        """Register callback for trade signals."""
        self._signal_callbacks.append(callback)

    def _emit_signal(self, signal: TradeSignal):
        self._signals.append(signal)
        for cb in self._signal_callbacks:
            try:
                cb(signal)
            except Exception as e:
                logger.error(f"Signal callback error: {e}")

    def link_match_to_market(self, match_id: str, market: EsportsMarket):
        """Associate a game match ID with a Polymarket market.
        Keeps both series moneyline AND map_winner for dual trading."""
        mtype = getattr(market, '_market_type', 'series')
        if mtype == 'map_winner':
            self._match_to_map_winner[match_id] = market
            # Only set as primary if no series moneyline exists
            if match_id not in self._match_to_market:
                self._match_to_market[match_id] = market
        else:
            # Series moneyline — always preferred as primary
            self._match_to_market[match_id] = market
        logger.info(f"Linked match {match_id} to market: {market.question}")

    def process_event(self, event: GameEvent) -> Optional[TradeSignal]:
        """
        Process a game event and check for latency arbitrage opportunity.

        This is called for every game event from every feed.
        Only generates signals for significant events with measurable edge.
        """
        # Skip non-impactful events
        if event.event_type not in self.HIGH_IMPACT_EVENTS | self.MEDIUM_IMPACT_EVENTS:
            return None

        if not event.match_state:
            return None

        # Find the corresponding Polymarket market
        market = self._match_to_market.get(event.match_id)
        if not market:
            market = self._find_market_for_match(event)
            if not market:
                if event.match_id not in self._match_decisions:
                    self._match_decisions[event.match_id] = {
                        "action": "no_market", "reason": "No Polymarket market linked",
                        "votes": 0, "confidence": 0, "time": time.time(),
                    }
                return None

        # Get real-time prices from WebSocket (preferred) or REST fallback
        if self._ws and market.token_id_a:
            ws_a = self._ws.get_price(market.token_id_a)
            ws_b = self._ws.get_price(market.token_id_b) if market.token_id_b else None
            if ws_a and (ws_a["last_trade"] > 0.01 or ws_a["best_bid"] > 0.01):
                # Use WS real-time prices
                if ws_a["last_trade"] > 0.01:
                    market.price_a = ws_a["last_trade"]
                elif ws_a["best_bid"] > 0 and ws_a["best_ask"] > 0:
                    market.price_a = (ws_a["best_bid"] + ws_a["best_ask"]) / 2
                market.best_bid_a = ws_a["best_bid"]
                market.best_ask_a = ws_a["best_ask"]
                if ws_b and (ws_b["last_trade"] > 0.01 or ws_b["best_bid"] > 0.01):
                    if ws_b["last_trade"] > 0.01:
                        market.price_b = ws_b["last_trade"]
                    elif ws_b["best_bid"] > 0 and ws_b["best_ask"] > 0:
                        market.price_b = (ws_b["best_bid"] + ws_b["best_ask"]) / 2
                    market.best_bid_b = ws_b["best_bid"]
                    market.best_ask_b = ws_b["best_ask"]
                else:
                    # No WS data for token_b — derive from token_a
                    market.price_b = 1.0 - market.price_a
                market.last_price_update = time.time()
            else:
                # WS has no data yet, use REST fallback
                now = time.time()
                last_fetch = self._last_price_fetch.get(market.market_id, 0)
                if now - last_fetch >= self.PRICE_FETCH_INTERVAL:
                    market = self.market_finder.update_market_prices(market)
                    self._last_price_fetch[market.market_id] = time.time()
        else:
            # No WS, use REST
            now = time.time()
            last_fetch = self._last_price_fetch.get(market.market_id, 0)
            if now - last_fetch >= self.PRICE_FETCH_INTERVAL:
                market = self.market_finder.update_market_prices(market)
                self._last_price_fetch[market.market_id] = time.time()
        market_check_time = time.time()

        # Check team alignment: does market.team_a correspond to feed's team_a?
        from team_match import teams_match as _tm
        feed_a = event.match_state.team_a
        feed_b = event.match_state.team_b
        # Check if feed's team_a matches market's team_a OR team_b
        a_matches_a = _tm(feed_a, market.team_a)
        a_matches_b = _tm(feed_a, market.team_b)
        if a_matches_a and not a_matches_b:
            aligned = True
        elif a_matches_b and not a_matches_a:
            aligned = False
        else:
            # Ambiguous — try team_b
            b_matches_a = _tm(feed_b, market.team_a)
            b_matches_b = _tm(feed_b, market.team_b)
            if b_matches_b and not b_matches_a:
                aligned = True
            elif b_matches_a and not b_matches_b:
                aligned = False
            else:
                aligned = True  # default assume aligned

        # Our estimate vs market price — adjust for alignment
        our_prob_a = event.match_state.win_probability_a
        if aligned:
            market_prob_a = market.price_a
            ask_a = market.best_ask_a if market.best_ask_a > 0 else market.price_a
            ask_b = market.best_ask_b if market.best_ask_b > 0 else market.price_b
        else:
            # Feed's team_a = market's team_b → flip market prices
            market_prob_a = market.price_b
            ask_a = market.best_ask_b if market.best_ask_b > 0 else market.price_b
            ask_b = market.best_ask_a if market.best_ask_a > 0 else market.price_a

        # Edge from midpoint (for logging/display)
        edge_a = our_prob_a - market_prob_a
        edge_b = -edge_a

        # Record measurement — use current time for staleness (not the earlier `now`)
        odds_staleness = time.time() - market.last_price_update if market.last_price_update > 0 else 999
        measurement = LatencyMeasurement(
            match_id=event.match_id,
            game=event.game,
            event_type=event.event_type.value,
            event_time=event.timestamp,
            market_check_time=market_check_time,
            market_price_team_a=market_prob_a,
            our_estimate_team_a=our_prob_a,
            price_gap=abs(edge_a),
            odds_staleness=odds_staleness,
            latency_edge_ms=(market_check_time - event.timestamp) * 1000,
        )
        self._measurements.append(measurement)

        # Skip dead markets (staleness > 10 min = market is frozen, not tradeable)
        if odds_staleness > 600:
            return None

        # Skip suspicious edges — only when CLOB prices have been fetched multiple times
        # (confirmed fresh, not stale defaults). Check price_fetch_count for this market.
        fetch_count = sum(1 for mid, t in self._last_price_fetch.items() if mid == market.market_id)
        if not self._use_claude and fetch_count >= 2 and odds_staleness < 15:
            if abs(edge_a) > 0.40:
                logger.info(f"[LATENCY] SKIP {event.game} | edge={abs(edge_a):.3f} too large (>40%) — confirmed model error")
                return None

        # Log latency measurement (only for viable events)
        logger.info(
            f"[LATENCY] {event.game} | {event.event_type.value} | "
            f"our={our_prob_a:.3f} vs market={market_prob_a:.3f} | "
            f"edge={abs(edge_a):.3f} | staleness={odds_staleness:.1f}s"
        )

        # No model-based filtering — trade on events directly

        # Determine which side to bet
        # edge_a > 0 means feed's team_a is undervalued → buy the token for feed's team_a
        if edge_a > 0:
            # Bet on feed's team_a
            team = "a"
            team_name = event.match_state.team_a
            # Get the correct market token for feed's team_a
            if aligned:
                token_id = market.token_id_a
                market_price = market_prob_a
            else:
                token_id = market.token_id_b  # feed's A = market's B
                market_price = market_prob_a
            our_price = our_prob_a
            edge = our_prob_a - ask_a
        else:
            # Bet on feed's team_b
            team = "b"
            team_name = event.match_state.team_b
            if aligned:
                token_id = market.token_id_b
                market_price = 1 - market_prob_a
            else:
                token_id = market.token_id_a  # feed's B = market's A
                market_price = 1 - market_prob_a
            our_price = 1 - our_prob_a
            edge = (1 - our_prob_a) - ask_b

        # ─── EDGE BOT: Pure event-driven trading ─────────────────────
        if True:
            # ─── EDGE BOT ──────────────────────────────────────────────
            # Buy the team that BENEFITS from the event.
            # The event.team tells us which side just had something good happen
            # (won a round, got a kill, took an objective).
            # Their price should go UP → buy them BEFORE market reacts.
            #
            if event.game not in self.EDGE_GAMES:
                return None
            if event.event_type not in self.EDGE_TRADEABLE_EVENTS:
                return None
            # Allow synthetic events through for Claude analysis (Claude decides if worth trading)
            logger.info(f"[EDGE-TRACE] passed filters | {event.game} {event.event_type.value} team={event.team} mid={event.match_id[:10]}")

            # One trade per event TYPE per match per 3s
            cooldown_key = f"{event.match_id}_{event.event_type.value}"
            last_trade_time = self._match_decisions.get(cooldown_key, {}).get("_last_trade", 0)
            if time.time() - last_trade_time < 3:
                return None

            # Per-token SL cooldown check moved below — needs buy_token resolved first.

            # LoL/Val: skip score_update UNLESS it represents a series score change
            # PandaScore heartbeats are team=neutral with no actual score change
            if event.game in ("lol", "valorant") and event.event_type == EventType.SCORE_UPDATE:
                # Allow through if series score changed (acts as synthetic map_win)
                _prev_score_key = f"_series_score_{event.match_id}"
                curr_score = (event.match_state.score_a, event.match_state.score_b) if event.match_state else (0, 0)
                prev_entry = self._match_decisions.get(_prev_score_key)
                if prev_entry is None:
                    # First time seeing this match — initialize without triggering
                    self._match_decisions[_prev_score_key] = {"score": curr_score}
                    return None
                prev_score = prev_entry.get("score", (0, 0))
                if curr_score != prev_score and (curr_score[0] + curr_score[1]) > (prev_score[0] + prev_score[1]):
                    self._match_decisions[_prev_score_key] = {"score": curr_score}
                    # Upgrade to map_win — a series score change IS a map completion
                    winning_team = "a" if curr_score[0] > prev_score[0] else "b"
                    logger.info(f"[EDGE] LIVE map_win detected: {prev_score} → {curr_score} (team {winning_team})")
                    event = GameEvent(
                        event_type=EventType.MAP_WIN,
                        match_id=event.match_id, game=event.game,
                        team=winning_team,
                        description=f"Map win! Series now {curr_score[0]}-{curr_score[1]}",
                        impact=event.impact, timestamp=event.timestamp, match_state=event.match_state,
                    )
                else:
                    return None  # regular PandaScore heartbeat

            # Skip stale map_win events — if the match state shows a DIFFERENT game
            # is already in progress (kills happening), this map_win is from an old game
            if event.event_type == EventType.MAP_WIN and event.match_state:
                # If the current game has kills > 0, a "map win" is stale (from previous game)
                kills_a = event.match_state.extra.get("kills_a", 0)
                kills_b = event.match_state.extra.get("kills_b", 0)
                game_minutes = event.match_state.extra.get("game_minutes", 0)
                if game_minutes > 5 or (kills_a + kills_b) > 5:
                    logger.info(f"[EDGE] SKIP — stale map_win (current game at {game_minutes:.0f}min, {kills_a}-{kills_b} kills)")
                    return None

            # ── AI-DRIVEN MODE: all pre-filters removed. Qwen gets raw context and decides ──
            # Previous filters (price range, spread, market-ahead, eco warnings, map_winner blocks,
            # exposure caps) have been stripped. Qwen sees the full game state + orderbook and
            # returns action + bet_size + tp_pct + sl_pct per trade.
            _warnings = []
            mtype = getattr(market, '_market_type', '')
            # Minimal mechanical protection: only reject if the token has REAL orderbook
            # data showing it's resolved (bid ≈ ask near 0 or 1). bid=0 with ask=0 means we
            # haven't subscribed yet — let qwen decide based on whatever data we have.
            if self._ws:
                ws_tok = self._ws.get_price(market.token_id_a)
                if ws_tok:
                    bid = ws_tok.get("best_bid", 0)
                    ask = ws_tok.get("best_ask", 0)
                    has_real_book = bid > 0 and ask > 0
                    if has_real_book:
                        if bid < 0.01 and ask < 0.05:
                            logger.info(f"[EDGE] SKIP — token A bid={bid*100:.2f}¢ ask={ask*100:.2f}¢ (resolved to 0)")
                            return None
                        if bid > 0.99 and ask > 0.99:
                            logger.info(f"[EDGE] SKIP — token A bid={bid*100:.2f}¢ ask={ask*100:.2f}¢ (resolved to 1)")
                            return None
            # Determine which team benefits from this event
            # event.team is "a", "b", or "neutral"
            if event.team == "a":
                buy_team = "a"
                buy_name = event.match_state.team_a
                buy_price = market_prob_a  # current market price for team_a
                model_price = our_prob_a
                if aligned:
                    buy_token = market.token_id_a
                else:
                    buy_token = market.token_id_b
            elif event.team == "b":
                buy_team = "b"
                buy_name = event.match_state.team_b
                buy_price = 1 - market_prob_a  # team_b price
                model_price = 1 - our_prob_a
                if aligned:
                    buy_token = market.token_id_b
                else:
                    buy_token = market.token_id_a
            else:
                # Neutral event — use edge direction to pick side for Claude analysis
                if edge_a > 0:
                    buy_team = "a"
                    buy_name = event.match_state.team_a if event.match_state else "?"
                    buy_price = market_prob_a
                    model_price = our_prob_a
                    buy_token = market.token_id_a if aligned else market.token_id_b
                else:
                    buy_team = "b"
                    buy_name = event.match_state.team_b if event.match_state else "?"
                    buy_price = 1 - market_prob_a
                    model_price = 1 - our_prob_a
                    buy_token = market.token_id_b if aligned else market.token_id_a

            # Stacking ALLOWED: multiple positions per token per match are fine —
            # momentum runs are real. The SL cooldown below prevents doubling down on losers.

            # Per-token SL cooldown: if THIS specific token was stop-lossed in last 10 min, skip.
            # Other side of the match remains tradeable.
            # Check under BOTH match_id and market_id keys — B8 19:19 SL didn't block 19:20
            # re-entry because match_id swaps between series/map events. Market_id is stable.
            _stop_keys = [
                f"{event.match_id}_{buy_token}_stopped",
                f"mkt_{market.market_id}_{buy_token}_stopped",
            ]
            _cooled = False
            for _sk in _stop_keys:
                _si = self._match_decisions.get(_sk, {})
                if _si.get("stopped") and time.time() - _si.get("stopped_at", 0) < 600:
                    _cooled = True
                    break
            # Token-only fallback (QUAZAR pattern): if bot.py stamp under market_id didn't
            # match because market_id linkage drifted, a plain token-keyed cooldown catches it.
            if not _cooled:
                _tok_cd = getattr(self, "_sl_token_cooldowns", {}).get(buy_token, 0)
                if _tok_cd and time.time() - _tok_cd < 600:
                    _cooled = True
            if _cooled:
                logger.info(f"[SL-COOLDOWN] {buy_name} — token stopped in last 10min, skipping")
                return None

            # Check for decisive map win BEFORE price range filter
            # A series-deciding map win should trade even at extreme prices
            _is_series_deciding = False
            if event.event_type == EventType.MAP_WIN and event.match_state and mtype != 'map_winner':
                sa = event.match_state.score_a
                sb = event.match_state.score_b
                maps_to_win = ((event.match_state.total_maps or 3) // 2) + 1
                if (event.team == "a" and sa >= maps_to_win) or (event.team == "b" and sb >= maps_to_win):
                    _is_series_deciding = True

            # No price/spread hard filters — qwen sees the numbers and decides.
            spread = 0
            ws_check = None
            if self._ws:
                ws_check = self._ws.get_price(buy_token)
                if not ws_check or ws_check.get("best_bid", 0) <= 0 or ws_check.get("best_ask", 0) <= 0:
                    ws_check = self._ws.get_price(market.token_id_a)
                if ws_check and ws_check.get("best_bid", 0) > 0 and ws_check.get("best_ask", 0) > 0:
                    spread = ws_check["best_ask"] - ws_check["best_bid"]
            actual_edge = model_price - buy_price
            # No market-ahead filter in AI-driven mode. Qwen sees our_price, buy_price, and
            # momentum — if the market moved 30¢ past our model, qwen can decide whether that's
            # a mispricing to fade or a legit information advantage to respect.

            # ─── EVENT WINDOW + CLAUDE ANALYSIS ────────────────────────
            # Track events in a rolling window per match
            window_key = event.match_id
            if window_key not in self._event_windows:
                self._event_windows[window_key] = []

            # FIREHOSE MODE: every event goes to qwen. The only thing we still skip is
            # FEED_CONNECTED / FEED_DISCONNECTED / FEED_ERROR meta events. Qwen sees all
            # actual game state changes (kills, rounds, scores, economy, objectives, etc.)
            # and decides what to do with each one.
            _meta_events = {EventType.FEED_CONNECTED, EventType.FEED_DISCONNECTED, EventType.FEED_ERROR}
            is_meaningful = event.event_type not in _meta_events

            self._event_windows[window_key].append({
                "type": event.event_type.value,
                "team": event.team,
                "description": event.description[:80],
                "timestamp": time.time(),
                "meaningful": is_meaningful,
            })

            now = time.time()
            # Keep last 20 events per match
            if len(self._event_windows[window_key]) > 20:
                self._event_windows[window_key] = self._event_windows[window_key][-20:]

            window = self._event_windows[window_key]
            meaningful_count = sum(1 for e in window if e.get("meaningful"))

            # BETWEEN-MAPS GUARD: after a map_win, the next map's round events haven't
            # started yet but stale eco/round_score from the previous map linger for ~30-60s.
            # qwen confabulates "10-13 round lead" etc. Block qwen until fresh round_end
            # evidence arrives (i.e. map_win is NOT the most recent meaningful event).
            _recent_meaningful = [e for e in window[-5:] if e.get("meaningful")
                                   and e.get("type") not in ("score_update",)]
            if _recent_meaningful and _recent_meaningful[-1].get("type") == "map_win":
                # last meaningful event is a map_win with nothing after → between maps
                logger.info(f"[BLOCK] {event.match_id} between maps — last event was map_win, stale state")
                return None

            # Per-match qwen cooldown. Qwen latency ~700ms and semaphore serializes, so
            # we can fire often. Drop to 2s to give qwen a chance to re-evaluate on
            # every fresh event cluster (user request: feed more data, latency allows it).
            _qwen_cd_key = f"_qwen_call_{event.match_id}"
            _last_qwen = self._match_decisions.get(_qwen_cd_key, {}).get("t", 0)
            qwen_cooldown_ok = (time.time() - _last_qwen) >= 2

            # PRE-FILTER: skip qwen for trades we can't execute or that lack data.
            _can_execute = True
            if buy_price >= 0.75:
                _can_execute = False  # SIZE-CAP would block
            elif model_price < 0.10:
                # Our model says this team has ≤10% chance — series likely decided.
                # The 14:11 BRUTE -$30 case: our_price=0.01, ask=0.48 → instant loss on resolve.
                logger.info(f"[BLOCK] {buy_name} model_price={model_price:.2f} <0.10 — series likely resolved")
                _can_execute = False
            elif (model_price - buy_price) < -0.20:
                # Edge worse than -20pp: we're paying way more than our model thinks fair.
                # Qwen sometimes confuses "ask is low" with "model says high" — guard against.
                logger.info(f"[BLOCK] {buy_name} edge={model_price-buy_price:+.2f} < -0.20")
                _can_execute = False
            elif buy_price > model_price + 0.05:
                # Overpay: ask is >5pp above our own model price. QUAZAR pattern — paying
                # 0.74 while model says 0.62 fair is a guaranteed loss post-mean-reversion.
                logger.info(f"[BLOCK] {buy_name} ask={buy_price:.3f} > model={model_price:.3f}+0.05 — overpay")
                _can_execute = False
            else:
                # STALE-INIT guard: the CS2 feed flags match_state with stale_init=True
                # when the first snapshot after a restart carries the previous (finished)
                # map's round score + economy. Block any trade until fresh per-round data
                # arrives (stale_init gets cleared on the next real round change).
                try:
                    _ms = event.match_state
                    if _ms and getattr(_ms, "extra", {}).get("stale_init"):
                        logger.info(f"[BLOCK] {buy_name} stale_init — waiting for fresh per-round data")
                        _can_execute = False
                except Exception:
                    pass
                # Wide-spread / broken-book / low-liquidity guards.
                # QUAZAR lesson: we bought with ask=0 (no sell side), bid=0.06, $731 volume —
                # instant no-exit trap. Block these before signal emission.
                try:
                    _p = self._ws.prices.get(buy_token, {}) if self._ws else {}
                    _bid = float(_p.get("best_bid", 0) or 0)
                    _ask = float(_p.get("best_ask", 0) or 0)
                    # Untradable ask (dust or missing)
                    if _ask <= 0.01 or _ask >= 0.99:
                        logger.info(f"[BLOCK] {buy_name} ask={_ask:.3f} untradable (dust/empty)")
                        _can_execute = False
                    # Wide spread OR missing-ask with bid far below model (broken book)
                    elif (_ask - _bid) > 0.12:
                        logger.info(f"[BLOCK] {buy_name} spread={(_ask-_bid):.3f} > 0.12 — wide book, SL trap")
                        _can_execute = False
                    elif _ask <= 0 and _bid > 0 and _bid < (model_price - 0.15):
                        logger.info(f"[BLOCK] {buy_name} no-ask + bid={_bid:.3f} <<model={model_price:.3f} — broken book")
                        _can_execute = False
                    # Liquidity / volume floor — kills dead markets
                    _liq = float(getattr(market, "liquidity", 0) or 0)
                    _vol = float(getattr(market, "volume", 0) or 0)
                    if _can_execute and (_liq < 500 or _vol < 1000):
                        logger.info(f"[BLOCK] {buy_name} low-liq: liq=${_liq:.0f} vol=${_vol:.0f}")
                        _can_execute = False
                except Exception:
                    pass
                # Kambi removed entirely — user preference: game events + PM orderbook only.
                # Stale/missing orderbook (no WS data ever received → 999 sentinel).
                # Try REST snapshot fallback to refresh; if that also fails, skip qwen.
                if self._ws:
                    _stale = self._ws.get_staleness(buy_token)
                    if _stale > 60:
                        try:
                            _snap = self.market_finder.fetch_orderbook(buy_token)
                            if _snap and _snap.best_bid > 0 and _snap.best_ask > 0:
                                self._ws.prices[buy_token] = {
                                    "best_bid": _snap.best_bid,
                                    "best_ask": _snap.best_ask,
                                    "last_trade": 0,
                                    "timestamp": time.time(),
                                }
                                logger.info(f"[REST-SNAP] Refreshed {buy_name[:14]} bid={_snap.best_bid:.3f} ask={_snap.best_ask:.3f}")
                                _stale = 0
                        except Exception as e:
                            logger.debug(f"[REST-SNAP] failed for {buy_token[:8]}: {e}")
                    if _stale > 300:
                        _can_execute = False
                if _can_execute and self._executor:
                    # Anti-hedge only: opposing side already open
                    _open_opp = [p for p in self._executor.positions if not p.resolved
                                 and p.market_id == market.market_id
                                 and p.token_id != buy_token]
                    if _open_opp:
                        _can_execute = False

            should_call_claude = (
                self._edge_analyst and
                is_meaningful and
                qwen_cooldown_ok and
                _can_execute
            )
            if should_call_claude:
                self._match_decisions[_qwen_cd_key] = {"t": time.time()}

            if should_call_claude:
                # Build game state for Claude
                game_state = {
                    "game": event.game,
                    "team_a": event.match_state.team_a if event.match_state else "?",
                    "team_b": event.match_state.team_b if event.match_state else "?",
                    "score_a": event.match_state.score_a if event.match_state else 0,
                    "score_b": event.match_state.score_b if event.match_state else 0,
                    "round_a": event.match_state.round_score_a if event.match_state else 0,
                    "round_b": event.match_state.round_score_b if event.match_state else 0,
                    "buy_team": buy_name,
                }
                if event.match_state and event.match_state.extra:
                    ex = event.match_state.extra
                    game_state.update({
                        "kills_a": ex.get("kills_a", 0),
                        "kills_b": ex.get("kills_b", 0),
                        "gold_lead": ex.get("gold_lead", 0),
                        "game_minutes": ex.get("game_minutes", 0),
                        "economy_a": ex.get("team_a_money", 0),
                        "economy_b": ex.get("team_b_money", 0),
                        "side": ex.get("team_a_current_side", ""),
                        "round_kills_a": ex.get("round_kills_a", 0),
                        "round_kills_b": ex.get("round_kills_b", 0),
                        "model_price": model_price,
                        # Rich CS2 live state (populated by cs2_bo3._team_rollup per 1s poll)
                        "alive_a": ex.get("alive_a", 5),
                        "alive_b": ex.get("alive_b", 5),
                        "avg_hp_a": ex.get("avg_hp_a", 100),
                        "avg_hp_b": ex.get("avg_hp_b", 100),
                        "damage_a": ex.get("damage_a", 0),
                        "damage_b": ex.get("damage_b", 0),
                        "has_awp_a": bool(ex.get("has_awp_a", False)),
                        "has_awp_b": bool(ex.get("has_awp_b", False)),
                        "has_defuse_kit_a": bool(ex.get("has_defuse_kit_a", False)),
                        "bomb_carrier": ex.get("bomb_carrier", ""),
                        "buy_type_a": ex.get("buy_type_a", ""),
                        "buy_type_b": ex.get("buy_type_b", ""),
                        "map_name": ex.get("map_name", ""),
                    })

                # Orderbook momentum across 10s / 30s / 60s windows — feed qwen enough
                # trajectory data to spot accelerations, reversals, and consolidation.
                momentum_10 = {}
                momentum_30 = {}
                momentum_60 = {}
                staleness = 999
                if self._ws:
                    try:
                        momentum_10 = self._ws.get_momentum(buy_token, window=10)
                    except TypeError:
                        momentum_10 = self._ws.get_momentum(buy_token)
                    momentum_30 = self._ws.get_momentum(buy_token, window=30)
                    momentum_60 = self._ws.get_momentum(buy_token, window=60)
                    staleness = self._ws.get_staleness(buy_token)

                market_state = {
                    "price": buy_price,
                    "bid": ws_check["best_bid"] if ws_check else 0,
                    "ask": ws_check["best_ask"] if ws_check else 0,
                    "spread": spread,
                    "volume": market.volume,
                    "liquidity": getattr(market, 'liquidity', 0),
                    "market_type": mtype or 'series',
                    "question": market.question[:80],
                    "momentum_10": momentum_10,
                    "momentum": momentum_30,      # backward-compat alias
                    "momentum_30": momentum_30,
                    "momentum_60": momentum_60,
                    "staleness_seconds": staleness,
                    # Balance context — qwen can scale stake as % of bankroll
                    "balance": float(getattr(self._executor, "balance", 0)) if self._executor else 0,
                    "open_positions": len([p for p in getattr(self._executor, "positions", []) if not p.resolved]) if self._executor else 0,
                }
                # Kambi fully removed — only Polymarket orderbook + game events go to qwen.
                # Warnings are now only informational (no blocking filters above)
                if _warnings:
                    game_state["warnings"] = _warnings

                # Call Claude asynchronously
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(self._async_claude_edge(
                        window, game_state, market_state,
                        event, buy_team, buy_name, buy_token, buy_price, model_price,
                        actual_edge, measurement, market, _is_series_deciding, cooldown_key,
                    ))
                except RuntimeError:
                    pass
                return None  # Claude will emit signal if it approves

            # No Claude available — skip trade (Claude is required for all decisions)
            if not self._edge_analyst:
                return None

            # Fallback: if Claude didn't handle it above, skip
            return None

            # Clinch detection (kept for reference but unreachable):
            is_clinch = False
            clinch_reason = ""

            if event.match_state:
                ms = event.match_state

                # CS2: team at match point (14+ rounds in regulation, or ahead in OT)
                if event.game == "cs2":
                    ra = ms.round_score_a
                    rb = ms.round_score_b
                    if event.team == "a" and ra >= 14 and ra > rb:
                        is_clinch = True
                        clinch_reason = f"match point {ra}-{rb}"
                    elif event.team == "b" and rb >= 14 and rb > ra:
                        is_clinch = True
                        clinch_reason = f"match point {rb}-{ra}"

                # Dota2: massive gold lead (>20k) + kill lead (>15) = game almost over
                if event.game == "dota2":
                    gold = ms.extra.get("gold_lead", 0)
                    kills_a = ms.extra.get("kills_a", 0)
                    kills_b = ms.extra.get("kills_b", 0)
                    game_min = ms.extra.get("game_minutes", 0)
                    if event.team == "a" and gold > 20000 and kills_a > kills_b + 15 and game_min > 25:
                        is_clinch = True
                        clinch_reason = f"dominant {kills_a}-{kills_b} +{gold:,}g @{game_min:.0f}min"
                    elif event.team == "b" and gold < -20000 and kills_b > kills_a + 15 and game_min > 25:
                        is_clinch = True
                        clinch_reason = f"dominant {kills_b}-{kills_a} +{abs(gold):,}g @{game_min:.0f}min"

                # WS orderbook confirmation: if best_bid jumped above 80¢, smart money sees the win
                if self._ws and buy_token:
                    ws_tok = self._ws.get_price(buy_token)
                    if ws_tok and ws_tok["best_bid"] > 0.80:
                        is_clinch = True
                        clinch_reason = f"orderbook confirms win (bid={ws_tok['best_bid']:.2f})"

            if _is_series_deciding:
                bet_amount = 200.0
                logger.info(f"[EDGE] DECISIVE MAP WIN: {buy_name} wins series! Staking $200")
            elif is_clinch:
                bet_amount = 100.0  # high conviction — team about to clinch
                logger.info(f"[EDGE] CLINCH DETECTED: {buy_name} — {clinch_reason} | Staking $100")
            else:
                bet_amount = config.EDGE_BOT_BET_SIZE

            signal = TradeSignal(
                match_id=event.match_id, game=event.game, market=market,
                team=buy_team, team_name=buy_name, token_id=buy_token,
                confidence=min(abs(actual_edge) * 5, 0.95),
                our_price=model_price, market_price=buy_price, edge=actual_edge,
                latency_edge_ms=measurement.latency_edge_ms,
                bet_amount=bet_amount,
                reason=f"EDGE: {actual_edge*100:+.1f}% | {event.event_type.value} | buy@{buy_price:.3f}",
            )
            self._match_decisions[event.match_id] = {
                "action": "bet", "reason": signal.reason,
                "votes": 0, "confidence": min(abs(actual_edge) * 5, 0.95),
                "time": time.time(),
            }
            self._match_decisions[cooldown_key] = {"_last_trade": time.time()}
            logger.info(
                f"[EDGE] BUY {buy_name} ({event.game}) | "
                f"price={buy_price:.3f} → model={model_price:.3f} | "
                f"edge={actual_edge*100:+.1f}% | {event.event_type.value} | "
                f"event favours: {buy_name}"
            )
            self._emit_signal(signal)
            self._notify_state_update()

        return None

    async def _async_claude_signal(self, event, team, team_name, token_id, our_price,
                                   market_price, edge, staleness, market, measurement):
        """Call Claude AI expert panel to analyze and potentially bet."""
        try:
            # Pass existing positions so Claude knows if we're adding to a position
            existing = []
            if self._executor:
                existing = [p for p in self._executor.positions
                            if not p.resolved and p.team.lower() == team_name.lower()]
            result = await self._claude_analyst.analyze_event(event, edge, staleness, market, existing)

            if result is None:
                # Get specific error from analyst
                error_detail = ""
                if self._claude_analyst:
                    if self._claude_analyst.consecutive_errors > 0:
                        error_detail = self._claude_analyst.last_error
                    elif self._claude_analyst._match_call_counts.get(event.match_id, 0) >= self._claude_analyst._match_call_limit:
                        error_detail = f"Call limit reached ({self._claude_analyst._match_call_limit} per match)"
                    else:
                        error_detail = "Empty response from API (may be rate limited or overloaded)"
                self._match_decisions[event.match_id] = {
                    "action": "skip", "reason": f"Claude returned no result — {error_detail}" if error_detail else "Claude returned no result",
                    "votes": 0, "confidence": 0, "time": time.time(),
                }
                return

            # Store decision for dashboard (full text, not truncated)
            self._match_decisions[event.match_id] = {
                "action": result.action,
                "reason": result.reasoning,
                "votes": result.bet_votes,
                "confidence": round(result.confidence, 2),
                "dissent": result.dissent if result.dissent else "",
                "time": time.time(),
            }

            confidence = result.confidence
            if result.action == "skip" or confidence < config.MIN_CONFIDENCE:
                logger.info(f"[CLAUDE] SKIP {team_name} — {result.reasoning}")
                return

            # No declining signal block — in fast arb mode we want volume.
            # Positions auto-exit via take-profit/stop-loss/timeout.

            normalized = (confidence - config.MIN_CONFIDENCE) / (1.0 - config.MIN_CONFIDENCE)
            normalized = min(1.0, max(0.0, normalized))
            bet_amount = config.MIN_BET + (config.MAX_BET - config.MIN_BET) * (normalized ** 1.5)
            bet_amount = round(bet_amount, 2)

            # Build full analysis for storage with position
            analysis = {
                "votes": result.bet_votes,
                "confidence": confidence,
                "reasoning": result.reasoning,
                "dissent": result.dissent,
                "expert_votes": [
                    {"expert": v.get("expert","?"), "vote": v.get("vote","?"), "reason": v.get("reason","")}
                    for v in (result.votes or [])
                ],
                "edge": round(edge, 3),
                "event": event.event_type.value,
                "time": time.time(),
            }

            signal = TradeSignal(
                match_id=event.match_id, game=event.game, market=market,
                team=team, team_name=team_name, token_id=token_id,
                confidence=confidence, our_price=our_price,
                market_price=market_price, edge=edge,
                latency_edge_ms=measurement.latency_edge_ms,
                bet_amount=bet_amount,
                reason=f"Claude panel: {result.bet_votes}/11 BET | {result.reasoning} | Dissent: {result.dissent}",
            )
            signal._analysis = analysis  # attach for executor to store
            logger.info(f"[CLAUDE SIGNAL] BET on {team_name} ({event.game}) | {result.bet_votes}/11 | conf={confidence:.2f} | ${bet_amount:.2f}")
            self._emit_signal(signal)

        except Exception as e:
            logger.error(f"[CLAUDE] Signal error: {e}")
        finally:
            self._notify_state_update()

    async def _async_claude_edge(self, window, game_state, market_state,
                                  event, buy_team, buy_name, buy_token, buy_price,
                                  model_price, actual_edge, measurement, market,
                                  is_series_deciding, cooldown_key):
        """Ask Claude if we should buy based on event window + game state."""
        try:
            decision = await self._edge_analyst.should_buy(window, game_state, market_state)

            if decision is None:
                return

            self._match_decisions[event.match_id] = {
                "action": decision.action,
                "reason": f"Claude: {decision.reason}",
                "votes": 0,
                "confidence": decision.confidence,
                "time": time.time(),
            }

            # SANITY VETO: qwen sometimes inverts/hallucinates ("BRUTE has eco advantage"
            # when in fact Fire Flux did). If the rationalization claims an advantage
            # that the actual numbers contradict, veto the BUY.
            if decision.action == "buy" and decision.confidence >= 0.4:
                _reason_lower = (decision.reason or "").lower()
                _eco_a = game_state.get("economy_a", 0); _eco_b = game_state.get("economy_b", 0)
                _kills_a = game_state.get("kills_a", 0); _kills_b = game_state.get("kills_b", 0)
                _buy_eco = _eco_a if buy_team == "a" else _eco_b
                _opp_eco = _eco_b if buy_team == "a" else _eco_a
                _buy_kills = _kills_a if buy_team == "a" else _kills_b
                _opp_kills = _kills_b if buy_team == "a" else _kills_a
                _vetoed = False
                # -1. Kambi hallucination — we removed Kambi entirely; any mention is stale
                # data the model invented. B8 -$67.97 + FURIA -$68.75 both cited fake Kambi.
                if "kambi" in _reason_lower:
                    logger.info(f"[VETO] {buy_name} — reason cites Kambi (removed; hallucination)")
                    _vetoed = True
                # -0.5. Generic "$X vs $Y" reversed-framing: if reason mentions "$Xk vs $Yk"
                # and buy side has less eco, the model framed a disadvantage as advantage.
                # FURIA loss: "economy advantage ($13k vs $17k)" — $13k < $17k is a deficit.
                if _buy_eco > 0 and _opp_eco > 0 and _buy_eco < _opp_eco:
                    import re as _re
                    # Match $7,300 / $14.5k / $13k / $13850 — strip commas, handle k suffix
                    _matches = _re.findall(r"\$\s*([\d,]+(?:\.\d+)?)\s*(k?)", _reason_lower)
                    _nums = []
                    for _val, _suffix in _matches:
                        try:
                            _n = float(_val.replace(",", ""))
                            if _suffix == "k":
                                _n *= 1000
                            _nums.append(_n)
                        except Exception:
                            pass
                    if len(_nums) >= 2:
                        try:
                            _a, _b = _nums[0], _nums[1]
                            # If first number < second and reason claims advantage → VETO
                            if _a < _b and ("advantage" in _reason_lower or "lead" in _reason_lower or "ahead" in _reason_lower):
                                logger.info(f"[VETO] {buy_name} — reversed framing '${_a} vs ${_b}' claimed as advantage")
                                _vetoed = True
                        except Exception:
                            pass
                # 0. Reason acknowledges buy_team's disadvantage ("disadvantage but momentum").
                # 3 of last 5 losses had this pattern — qwen rationalizes bad eco with weak momentum.
                if ("economy disadvantage" in _reason_lower or "eco disadvantage" in _reason_lower):
                    if _buy_eco < _opp_eco:
                        logger.info(f"[VETO] {buy_name} — reason acknowledges eco disadvantage (${_buy_eco:,} < ${_opp_eco:,})")
                        _vetoed = True
                # 1. Eco advantage claim vs reality. Tightened: any deficit triggers, not just $2k+.
                if "economy advantage" in _reason_lower or "eco advantage" in _reason_lower:
                    if _eco_a == 0 and _eco_b == 0:
                        logger.info(f"[VETO] {buy_name} — qwen claimed eco advantage but no eco data (both $0)")
                        _vetoed = True
                    elif _buy_eco < _opp_eco:
                        logger.info(f"[VETO] {buy_name} — qwen claimed eco advantage but buy=${_buy_eco} < opp=${_opp_eco}")
                        _vetoed = True
                # 2. Reason explicitly names opposing team with advantage (e.g. "Spirit +$550")
                _opp_name = (game_state.get("team_b","") if buy_team=="a" else game_state.get("team_a","")).lower()
                if _opp_name and _opp_name in _reason_lower and ("advantage" in _reason_lower or "lead" in _reason_lower):
                    # Check it's the opp that has the advantage, not a comparison
                    # "Spirit +$550" or "FaZe leads" → opp actually has advantage
                    if _buy_eco <= _opp_eco and _buy_kills <= _opp_kills:
                        logger.info(f"[VETO] {buy_name} — reason names opp '{_opp_name}' with advantage/lead")
                        _vetoed = True
                # 3. Kill lead claim vs reality
                if "kill lead" in _reason_lower and _buy_kills < _opp_kills:
                    logger.info(f"[VETO] {buy_name} — qwen claimed kill lead but buy={_buy_kills} < opp={_opp_kills}")
                    _vetoed = True
                # 4. Positive momentum claim vs reality
                if "positive momentum" in _reason_lower or "positive 10s" in _reason_lower or "positive 30s" in _reason_lower:
                    _m10 = market_state.get("momentum_10", {}).get("bid_change", 0) if isinstance(market_state.get("momentum_10"), dict) else 0
                    _m30 = market_state.get("momentum_30", {}).get("bid_change", 0) if isinstance(market_state.get("momentum_30"), dict) else 0
                    if _m10 < 0 and _m30 < 0:
                        logger.info(f"[VETO] {buy_name} — qwen claimed positive momentum but m10={_m10*100:+.1f}c m30={_m30*100:+.1f}c")
                        _vetoed = True
                if _vetoed:
                    return None

            if decision.action == "buy" and decision.confidence >= 0.4:
                # AI-driven mode: qwen decides bet size AND exit targets per trade.
                # Fall back to config defaults if the model omitted a field.
                # CONFIDENCE-SCALED SIZING: override qwen's bet_size with a curve
                # tied to its confidence. At conf=0.40 (threshold) → $30 floor.
                # At conf=1.00 → $150 ceiling (pre price-cap). Quadratic so only
                # strong-conviction decisions get meaningful size.
                _conf = max(0.4, min(1.0, float(decision.confidence)))
                _norm = (_conf - 0.4) / 0.6   # 0.0 at 0.4 conf, 1.0 at 1.0 conf
                _min_bet = 30.0
                _max_bet = 150.0
                bet_amount = _min_bet + (_max_bet - _min_bet) * (_norm ** 1.5)
                bet_amount = round(bet_amount, 0)
                logger.info(f"[CONF-SIZE] {buy_name} conf={_conf:.2f} → ${bet_amount:.0f}")
                # ADAPTIVE TP/SL: scale targets with fill price and confidence.
                # - Low fill (0.30-0.50) has room to run → wider TP, wider SL (more noise).
                # - High fill (0.70+) is near ceiling → tighter TP, tighter SL.
                # - Higher confidence → widen both (ride the thesis).
                # SL widened to 12% max (user request: noise room for 3-5¢ bid swings).
                _price_mid = 1.0 - abs(buy_price - 0.50) / 0.50   # 1.0 at 0.5, 0 at 0 or 1
                # TP: fill=0.50 → 12% (lots of upside), fill=0.80 → 5% (small upside).
                #     Conf scales +/-2pp.
                _tp_base = 0.05 + 0.07 * max(0, min(1, (0.80 - buy_price) / 0.60))
                _tp_conf_bonus = 0.02 * ((_conf - 0.4) / 0.6)  # up to +2pp at conf 1.0
                tp_pct = max(0.05, min(0.14, _tp_base + _tp_conf_bonus))
                # SL: fill=0.50 → 12% (max noise room), fill=0.20/0.80 → 6%.
                #     Conf scales +/-2pp (higher conf = willing to absorb more noise).
                _sl_base = 0.06 + 0.06 * _price_mid
                _sl_conf_bonus = 0.02 * ((_conf - 0.4) / 0.6)
                sl_pct = max(0.06, min(0.12, _sl_base + _sl_conf_bonus))
                logger.info(f"[ADAPTIVE-TPSL] {buy_name} fill={buy_price:.2f} conf={_conf:.2f} → tp={tp_pct*100:.1f}% sl={sl_pct*100:.1f}%")
                # PRICE-SCALED SIZING: session data shows 100% of $-235 loss in
                # the $100-200 bet bucket came from high-price fills. Cap the
                # bet size based on fill price — cheap tokens can absorb large
                # bets (asymmetric upside), expensive tokens bleed if wrong.
                # Cap table: <0.40 → 200, 0.40-0.55 → 100, 0.55-0.70 → 150,
                # 0.70-0.80 → 30. Beyond 0.80 is rejected entirely in executor.
                # Tightened per post-fix data: 0.55-0.70 still bleeds (-$13 avg)
                # so cap cut in half. 0.70+ now blocked — we have no +EV zone there.
                if buy_price < 0.25:
                    cap = 100.0   # cheap underdogs — high variance, moderate size
                elif buy_price < 0.40:
                    cap = 150.0   # +$147 historically — best zone
                elif buy_price < 0.55:
                    cap = 40.0    # 0.40-0.55 dead zone bleeds -$18/avg, tightened from $75
                elif buy_price < 0.70:
                    cap = 75.0    # was $150 — still bleeding post-fix
                elif buy_price < 0.75:
                    cap = 30.0
                else:
                    cap = 0.0
                if cap == 0.0:
                    logger.info(f"[SIZE-CAP] SKIP {buy_name} — price {buy_price:.2f} above allowable zone")
                    return None
                if bet_amount > cap:
                    logger.info(f"[SIZE-CAP] {buy_name} @ {buy_price:.2f}: ${bet_amount:.0f} → ${cap:.0f} cap")
                    bet_amount = cap

                signal = TradeSignal(
                    match_id=event.match_id, game=event.game, market=market,
                    team=buy_team, team_name=buy_name, token_id=buy_token,
                    confidence=decision.confidence,
                    our_price=model_price, market_price=buy_price, edge=actual_edge,
                    latency_edge_ms=measurement.latency_edge_ms,
                    bet_amount=bet_amount,
                    reason=f"AI: {decision.confidence:.0%} | tp={tp_pct*100:.1f}% sl={sl_pct*100:.1f}% | ${bet_amount:.0f} | {decision.reason[:50]}",
                )
                # Attach qwen's full decision for executor + dashboard
                signal._analysis = {
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "bet_size": bet_amount,
                    "tp_pct": tp_pct,
                    "sl_pct": sl_pct,
                    "game_state": game_state,
                    "market_state": market_state,
                    "events_analyzed": len(window),
                    "event_window": [
                        {"type": e.get("type"), "team": e.get("team"), "desc": e.get("description", "")[:60]}
                        for e in window[-10:]
                    ],
                }
                # Exit targets qwen chose — executor reads these off the signal
                signal._tp_pct = tp_pct
                signal._sl_pct = sl_pct
                self._match_decisions[cooldown_key] = {"_last_trade": time.time()}
                logger.info(
                    f"[AI-EDGE] BUY {buy_name} ({event.game}) | "
                    f"conf={decision.confidence:.2f} | ${bet_amount:.0f} | "
                    f"tp={tp_pct*100:.1f}% sl={sl_pct*100:.1f}% | "
                    f"{decision.reason[:60]}"
                )
                self._emit_signal(signal)
            else:
                logger.info(
                    f"[CLAUDE-EDGE] SKIP {buy_name} ({event.game}) | "
                    f"conf={decision.confidence:.2f} | {decision.reason[:50]}"
                )

        except Exception as e:
            logger.error(f"[CLAUDE-EDGE] Error: {e}")
        finally:
            self._notify_state_update()

    def _find_market_for_match(self, event: GameEvent) -> Optional[EsportsMarket]:
        """Try to find a Polymarket moneyline market matching a game event."""
        if not event.match_state:
            return None

        team_a = event.match_state.team_a.lower()
        team_b = event.match_state.team_b.lower()

        if not team_a or not team_b:
            return None

        # Use cached markets only — never block the event loop with a fresh HTTP fetch.
        # The periodic market scanner (bot._market_scanner) repopulates the cache every 30s.
        markets = [m for m in self.market_finder._market_cache.values()
                   if m.game == event.game]

        # Prefer moneyline (BO3/BO1) markets over handicap/map/prop markets
        from team_match import teams_match as _teams_match
        candidates = []
        for market in markets:
            q = market.question
            # Match against question text AND market team_a/team_b fields
            a_found = (team_in_text(event.match_state.team_a, q) or
                       _teams_match(event.match_state.team_a, market.team_a) or
                       _teams_match(event.match_state.team_a, market.team_b))
            b_found = (team_in_text(event.match_state.team_b, q) or
                       _teams_match(event.match_state.team_b, market.team_a) or
                       _teams_match(event.match_state.team_b, market.team_b))
            if not (a_found and b_found):
                continue

            # Hard skip: markets where team-win probability doesn't apply
            skip_keywords = ["total maps", "over", "under",
                             "prop", "first blood", "total kills"]
            if any(kw in q for kw in skip_keywords):
                continue

            bo_keywords = ["bo3", "bo1", "bo5", "(bo"]
            is_moneyline = any(kw in q for kw in bo_keywords) or "winner" in q

            score = 0
            if is_moneyline:
                score += 10
            score += 5  # passed the skip filter
            score += market.volume / 100000

            candidates.append((score, market))

        if candidates:
            candidates.sort(key=lambda x: -x[0])
            best = candidates[0][1]
            self._match_to_market[event.match_id] = best
            return best

        return None

    def get_stats(self) -> dict:
        """Return latency analysis statistics."""
        if not self._measurements:
            return {
                "total_measurements": 0,
                "avg_edge": 0,
                "max_edge": 0,
                "avg_staleness": 0,
                "signals_generated": len(self._signals),
                "match_decisions": dict(self._match_decisions),
            }

        edges = [m.price_gap for m in self._measurements]
        staleness = [m.odds_staleness for m in self._measurements if m.odds_staleness < 900]

        return {
            "total_measurements": len(self._measurements),
            "avg_edge": sum(edges) / len(edges),
            "max_edge": max(edges),
            "avg_staleness": sum(staleness) / len(staleness) if staleness else 0,
            "signals_generated": len(self._signals),
            "recent_signals": [
                {
                    "game": s.game,
                    "team": s.team_name,
                    "edge": round(s.edge, 3),
                    "confidence": round(s.confidence, 2),
                    "amount": s.bet_amount,
                    "time": s.timestamp,
                }
                for s in list(self._signals)[-10:]
            ],
            "match_decisions": dict(self._match_decisions),
        }
