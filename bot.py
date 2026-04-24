"""
Core bot orchestrator — connects game feeds, latency analyzer, and trade executor.

Architecture:
  Game Feeds (CS2, LoL, Dota2, Val) → LatencyAnalyzer → TradeSignals → Executor
                                                                          ↓
                                                              Polymarket CLOB
"""
import asyncio
import logging
import time
import threading
from dataclasses import dataclass, field
from typing import Optional

import config
# from prior import combine_priors  # disabled for edge bot
from team_match import teams_match, team_in_text
# from odds_reference import OddsReference  # disabled — not needed for edge bot
from feeds.base import GameFeed, GameEvent, EventType
from feeds.cs2_bo3 import CS2Bo3Feed
from feeds.cs2_bo3_ws import CS2Bo3WebSocketFeed
from feeds.dota2_opendota import Dota2OpenDotaFeed
from feeds.lol_lolesports import LoLLolesportsFeed
from feeds.lol_livestats import LoLLiveStatsFeed
from feeds.valorant_vlr import ValorantVLRFeed
from feeds.cs2_hltv import CS2HLTVFeed
from feeds.cs2_gsi import CS2GSIFeed
from feeds.pandascore import PandaScoreFeed
# stream_ocr requires opencv (cv2) which is a heavy GUI dep. Make it optional
# so the bot can run on headless servers (VPS) without installing it.
try:
    from feeds.stream_ocr import StreamOCRFeed
    _STREAM_OCR_AVAILABLE = True
except ImportError:
    StreamOCRFeed = None  # type: ignore
    _STREAM_OCR_AVAILABLE = False
from cs2_spectator import CS2Spectator
from feeds.dota2_steam_web import Dota2SteamWebFeed
from feeds.kambi import KambiFeed
from market import MarketFinder
from latency import LatencyAnalyzer, TradeSignal
from executor import TradeExecutor
from polymarket_ws import PolymarketWebSocket
from price_impact import PriceImpactTracker
from edge_analyst import EdgeAnalyst
from match_analyzer import ContinuousMatchAnalyzer
from match_recorder import MatchRecorder

logger = logging.getLogger(__name__)


@dataclass
class BotState:
    """Shared state for dashboard updates."""
    status: str = "initializing"
    mode: str = "DRY-RUN"
    uptime_seconds: float = 0.0
    feeds_connected: dict = field(default_factory=dict)
    active_matches: list = field(default_factory=list)
    recent_events: list = field(default_factory=list)
    raw_feed_events: list = field(default_factory=list)  # ALL game events from ALL feeds
    recent_signals: list = field(default_factory=list)
    trade_stats: dict = field(default_factory=dict)
    latency_stats: dict = field(default_factory=dict)
    markets_tracked: int = 0
    claude_stats: dict = field(default_factory=dict)
    pending_orders: list = field(default_factory=list)
    price_impacts: list = field(default_factory=list)
    impact_summary: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "mode": self.mode,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "feeds_connected": self.feeds_connected,
            "active_matches": self.active_matches,
            "recent_events": self.recent_events[-50:],
            "raw_feed_events": self.raw_feed_events[-100:],
            "recent_signals": self.recent_signals[-10:],
            "trade_stats": self.trade_stats,
            "latency_stats": self.latency_stats,
            "markets_tracked": self.markets_tracked,
            "claude_stats": self.claude_stats,
            "pending_orders": self.pending_orders,
            "price_impacts": self.price_impacts,
            "impact_summary": self.impact_summary,
        }


class EsportsBot:
    """Main bot that orchestrates all components."""

    def __init__(self, dry_run: bool = True):
        self.dry_run = dry_run
        self._start_time = time.time()

        # Core components
        self.market_finder = MarketFinder()
        self.latency_analyzer = LatencyAnalyzer(self.market_finder)
        self.executor = TradeExecutor(dry_run=dry_run)
        self.executor.market_finder = self.market_finder
        self.polymarket_ws = PolymarketWebSocket()
        self.latency_analyzer._ws = self.polymarket_ws  # real-time prices for edge detection
        self.latency_analyzer._executor = self.executor  # for position checks
        # NOTE: _feeds wired in after _init_feeds() below

        # Edge Claude analyst — lightweight analysis for trading decisions
        if config.ANTHROPIC_API_KEY:
            try:
                self.edge_analyst = EdgeAnalyst(db=self.executor.db)
                self.latency_analyzer._edge_analyst = self.edge_analyst
                logger.info(f"Edge Claude analyst enabled (model: {config.EDGE_CLAUDE_MODEL})")
            except Exception as e:
                self.edge_analyst = None
                logger.warning(f"Edge Claude analyst disabled: {e}")
        else:
            self.edge_analyst = None
        self.executor._ws = self.polymarket_ws           # real-time prices for P&L
        self.price_impact = PriceImpactTracker(self.polymarket_ws)
        self.match_analyzer = ContinuousMatchAnalyzer(self)  # continuous Gemma 4 analysis
        self.recorder = MatchRecorder()  # record ALL data for backtesting

        # Team intelligence & reference odds
        self.odds_ref = None

        # Claude AI expert panel
        self._claude_analyst = None

        # Game feeds
        self.feeds: dict[str, GameFeed] = {}
        self._init_feeds()
        self.latency_analyzer._feeds = self.feeds  # access to kambi feed for cross-market odds

        # State
        self.state = BotState(mode="DRY-RUN" if dry_run else "LIVE")
        self._state_lock = threading.Lock()
        self._state_callbacks: list = []
        self._running = False

        self._unlinked_logged: set[str] = set()

        # Wire up event pipeline
        self.latency_analyzer.on_signal(self._on_trade_signal)
        self.latency_analyzer.on_state_update(self._update_state)

    def _init_feeds(self):
        """Initialize enabled game feeds."""
        feed_map = {
            "cs2": CS2Bo3Feed,              # Bo3.gg — polls every 1s (backup)
            "dota2": Dota2OpenDotaFeed,      # OpenDota — free, no auth
            "lol": LoLLiveStatsFeed,         # lolesports live stats — gold, kills, dragons, towers
            "valorant": ValorantVLRFeed,     # VLR.gg scraping — free
        }

        for game in config.ENABLED_GAMES:
            game = game.strip()
            if game in feed_map:
                feed = feed_map[game]()
                feed.on_event(self._on_game_event)
                # Wire the rich-payload recording hook if the feed supports it.
                # CS2Bo3Feed (polling) and CS2Bo3WebSocketFeed both expose this
                # via the GameFeed base class. Other feeds (dota, lol, val)
                # don't need it yet — their recording path is simpler.
                if hasattr(feed, 'on_raw_snapshot'):
                    feed.on_raw_snapshot(self._on_raw_bo3_snapshot)
                self.feeds[game] = feed
                logger.info(f"Initialized {game} feed")

        # Add Steam Web API as secondary Dota2 feed (gold, heroes, towers)
        if "dota2" in self.feeds and config.STEAM_API_KEY:
            try:
                steam_web = Dota2SteamWebFeed()
                steam_web.on_event(self._on_game_event)
                self.feeds["dota2_steam_web"] = steam_web
                logger.info("Initialized Dota2 Steam Web API feed (gold + heroes)")
            except Exception as e:
                logger.warning(f"Steam Web feed init failed: {e}")

        # Kambi sportsbook feed — cross-market odds + map scores for all enabled games.
        # The real signal is odds movement: Kambi's pro bookmaker moves before Polymarket.
        try:
            kambi = KambiFeed()
            kambi.on_event(self._on_game_event)
            self.feeds["kambi"] = kambi
            logger.info("Initialized Kambi feed (cross-market odds + map scores)")
        except Exception as e:
            logger.warning(f"Kambi feed init failed: {e}")

        # Stream OCR feeds (only for enabled games)
        # CS2 has authoritative Bo3.gg WS feed — OCR is redundant AND its left/right
        # screen-position parsing flips team_a/team_b vs Bo3, causing wrong-team analysis.
        for ocr_game in config.ENABLED_GAMES:
            ocr_game = ocr_game.strip()
            if ocr_game == "cs2":
                logger.info("Skipping Stream OCR for cs2 — Bo3.gg WS provides authoritative data")
                continue
            if not _STREAM_OCR_AVAILABLE:
                logger.info(f"Stream OCR unavailable (cv2 not installed) — skipping {ocr_game}")
                continue
            try:
                ocr_feed = StreamOCRFeed(ocr_game)
                ocr_feed.on_event(self._on_game_event)
                self.feeds[f"{ocr_game}_ocr"] = ocr_feed
                logger.info(f"Initialized Stream OCR feed for {ocr_game}")
            except Exception as e:
                logger.warning(f"Stream OCR {ocr_game} init failed: {e}")

        # PandaScore feeds (only for enabled games that need it)
        if config.PANDASCORE_TOKEN:
            for ps_game in config.ENABLED_GAMES:
                ps_game = ps_game.strip()
                if ps_game in ("cs2", "dota2"):
                    continue  # CS2 has Bo3.gg, Dota2 has OpenDota+Steam — don't need PandaScore
                try:
                    ps_feed = PandaScoreFeed(ps_game)
                    ps_feed.on_event(self._on_game_event)
                    self.feeds[f"{ps_game}_pandascore"] = ps_feed
                    logger.info(f"Initialized PandaScore {ps_game} feed")
                except Exception as e:
                    logger.warning(f"PandaScore {ps_game} init failed: {e}")

        # CS2 spectator manager (auto-cycles GOTV connections)
        self.cs2_spectator = CS2Spectator()
        if self.cs2_spectator.setup():
            logger.info("CS2 spectator ready — add GOTV addresses to start receiving data")

        # Bo3.gg WebSocket — real-time CS2 data (replaces polling)
        if "cs2" in self.feeds:
            try:
                ws_feed = CS2Bo3WebSocketFeed()
                ws_feed.on_event(self._on_game_event)
                # Record every raw bo3.gg payload so backtests have full
                # player_states / round_phase / HP / side / bomb data. The
                # recorder writes these as SNAPSHOT_MATCH_UPDATE lines that
                # match the format of our curated Test Data corpus.
                ws_feed.on_raw_snapshot(self._on_raw_bo3_snapshot)
                ws_feed.set_polling_feed(self.feeds.get("cs2"))  # share team names
                self.feeds["cs2_ws"] = ws_feed
                logger.info("Initialized Bo3.gg WebSocket feed (real-time CS2)")
            except Exception as e:
                logger.warning(f"Bo3.gg WS feed init failed: {e}")

        # CS2 GSI disabled — requires local spectator client running

        # HLTV disabled — always fails to connect, adds noise to feeds display

    @staticmethod
    def _teams_aligned(match_state, market) -> Optional[bool]:
        """Check if market.team_a corresponds to match_state.team_a (True), team_b (False), or unknown (None)."""
        a_matches_a = teams_match(match_state.team_a, market.team_a)
        a_matches_b = teams_match(match_state.team_a, market.team_b)

        if a_matches_a and not a_matches_b:
            return True  # aligned
        if a_matches_b and not a_matches_a:
            return False  # inverted
        return None  # ambiguous

    def on_state_change(self, callback):
        """Register callback for state updates (used by dashboard)."""
        self._state_callbacks.append(callback)

    def _build_market_info(self, market) -> dict:
        """
        Build a uniform dict describing one Polymarket market for the dashboard.
        Used per-match for both series moneyline and map_winner so each renders
        independently with its own bid/ask/spread/volume/liquidity/team prices.
        """
        info = {
            "market_id": market.market_id,
            "market_type": getattr(market, "_market_type", "series"),
            "question": (market.question or "")[:80],
            "volume": round(market.volume, 0),
            "liquidity": round(market.liquidity, 0),
            "price_a": round(market.price_a, 3),
            "price_b": round(market.price_b, 3),
            "bid_a": 0.0, "ask_a": 0.0, "spread": 0.0,
            "bid_b": 0.0, "ask_b": 0.0,
        }
        if market.event_slug:
            info["polymarket_url"] = f"https://polymarket.com/event/{market.event_slug}"
        # Layer in live WS prices if available
        ws_a = self.polymarket_ws.get_price(market.token_id_a) if market.token_id_a else None
        ws_b = self.polymarket_ws.get_price(market.token_id_b) if market.token_id_b else None
        if ws_a and ws_a.get("best_bid", 0) > 0:
            bid = ws_a["best_bid"]; ask = ws_a.get("best_ask", 0) or 0
            spread = ask - bid if ask > 0 else 0
            # Only trust WS prices when orderbook isn't pure dust
            if spread < 0.50 and bid > 0.05:
                info["bid_a"] = round(bid, 3)
                info["ask_a"] = round(ask, 3)
                info["spread"] = round(spread, 3)
                if ws_a.get("last_trade", 0) > 0.01:
                    info["price_a"] = round(ws_a["last_trade"], 3)
                elif ask > 0:
                    info["price_a"] = round((bid + ask) / 2, 3)
            else:
                # Dust book — still record the raw bid/ask for transparency
                info["bid_a"] = round(bid, 3) if bid > 0.05 else 0
                info["ask_a"] = round(ask, 3) if ask and ask < 0.95 else 0
                info["spread"] = round(spread, 3)
        if ws_b and ws_b.get("best_bid", 0) > 0:
            bid_b = ws_b["best_bid"]; ask_b = ws_b.get("best_ask", 0) or 0
            info["bid_b"] = round(bid_b, 3)
            info["ask_b"] = round(ask_b, 3)
            if ws_b.get("last_trade", 0) > 0.01:
                info["price_b"] = round(ws_b["last_trade"], 3)
            elif ask_b > 0:
                info["price_b"] = round((bid_b + ask_b) / 2, 3)
        # Prices should sum to ~1.0; when only one side has WS data, derive the other
        if info["price_a"] > 0 and (info["price_b"] <= 0 or abs(info["price_a"] + info["price_b"] - 1.0) > 0.15):
            info["price_b"] = round(1.0 - info["price_a"], 3)
        return info

    def _update_state(self):
        """Update shared state for dashboard."""
        with self._state_lock:
            self.state.uptime_seconds = time.time() - self._start_time
            self.state.feeds_connected = {
                game: feed.is_connected for game, feed in self.feeds.items()
            }
            self.state.active_matches = []
            linked_ids = set(self.latency_analyzer._match_to_market.keys())
            # Track seen team pairs to deduplicate across feeds
            _seen_matchups: dict[str, dict] = {}  # "teama_vs_teamb" -> best match_info
            # Also include matches with open bets (even if market link lost)
            bet_teams = {p.team.lower() for p in self.executor.positions if not p.resolved}
            now_ts = time.time()
            for feed in self.feeds.values():
                for match in feed._matches.values():
                    if not match.is_live:
                        continue
                    # Auto-expire stale matches: 3h total or 1h since last event.
                    last_event = match.last_event_time or match.started_at
                    match_age = now_ts - match.started_at
                    event_age = now_ts - last_event
                    if match_age > 10800 or event_age > 3600:
                        match.is_live = False
                        continue
                    # AI-driven mode: show ALL live matches that have a Polymarket market.
                    # Matches without a tradeable market on PM are noise — qwen can't bet on them.
                    has_market = match.match_id in linked_ids
                    has_bet = match.team_a.lower() in bet_teams or match.team_b.lower() in bet_teams
                    # Also accept matches linked by team name (cross-feed) — search PM cache
                    if not has_market and self.latency_analyzer:
                        for mkt in self.latency_analyzer._match_to_market.values():
                            if (teams_match(match.team_a, mkt.team_a) and teams_match(match.team_b, mkt.team_b)) or \
                               (teams_match(match.team_a, mkt.team_b) and teams_match(match.team_b, mkt.team_a)):
                                has_market = True
                                break
                    if not has_market and not has_bet:
                        continue
                    if True:
                        match_info = {
                            "match_id": match.match_id,
                            "game": match.game,
                            "team_a": match.team_a,
                            "team_b": match.team_b,
                            "score": f"{match.score_a}-{match.score_b}",
                            "win_prob_a": round(match.win_probability_a, 3),
                            "current_map": match.current_map,
                            "total_maps": match.total_maps,
                            "started_at": match.started_at,
                            "upcoming": match.match_id.startswith("mkt_"),
                            "market_price_a": 0,
                            "market_price_b": 0,
                            "best_bid_a": 0,
                            "best_ask_a": 0,
                            "spread": 0,
                        }
                        # Fix series score from resolved game_winner markets
                        if match.score_a == 0 and match.score_b == 0:
                            # Count resolved game winner markets for this matchup
                            from team_match import teams_match as _tm2
                            sa, sb = 0, 0
                            for mkt2 in self.market_finder._market_cache.values():
                                if mkt2.game != "dota2": continue
                                q2 = mkt2.question.lower()
                                if 'winner' not in q2 or ('game' not in q2 and 'map' not in q2): continue
                                if not (_tm2(match.team_a, mkt2.team_a) or _tm2(match.team_a, mkt2.team_b)): continue
                                if not (_tm2(match.team_b, mkt2.team_a) or _tm2(match.team_b, mkt2.team_b)): continue
                                # Check if resolved
                                if mkt2.price_a > 0.95:
                                    if _tm2(match.team_a, mkt2.team_a): sa += 1
                                    else: sb += 1
                                elif mkt2.price_b > 0.95:
                                    if _tm2(match.team_b, mkt2.team_b): sb += 1
                                    else: sa += 1
                            if sa + sb > 0:
                                match_info["score"] = f"{sa}-{sb}"

                        # Game-specific stats
                        if match.game == "cs2" and match.extra:
                            from feeds.cs2_model import classify_buy
                            match_info["round_score"] = f"{match.round_score_a}-{match.round_score_b}"
                            match_info["side"] = match.extra.get("team_a_current_side", "?")
                            match_info["money_a"] = match.extra.get("team_a_money", 0)
                            match_info["money_b"] = match.extra.get("team_b_money", 0)
                            match_info["buy_a"] = classify_buy(match.extra.get("team_a_money", 0))
                            match_info["buy_b"] = classify_buy(match.extra.get("team_b_money", 0))
                            match_info["kills"] = f"{match.extra.get('round_kills_a', 0)}-{match.extra.get('round_kills_b', 0)}"
                        elif match.game == "dota2" and match.extra:
                            match_info["kills_a"] = match.extra.get("kills_a", 0)
                            match_info["kills_b"] = match.extra.get("kills_b", 0)
                            match_info["game_minutes"] = match.extra.get("game_minutes", 0)
                            match_info["gold_lead"] = match.extra.get("gold_lead", 0)
                            match_info["towers_a"] = match.extra.get("towers_a", 0)
                            match_info["towers_b"] = match.extra.get("towers_b", 0)
                        elif match.game == "lol":
                            match_info["games_score"] = f"{match.score_a}-{match.score_b}"
                            if match.extra.get("gold_a"):
                                match_info["gold_a"] = match.extra.get("gold_a", 0)
                                match_info["gold_b"] = match.extra.get("gold_b", 0)
                                match_info["kills_a"] = match.extra.get("kills_a", 0)
                                match_info["kills_b"] = match.extra.get("kills_b", 0)
                                match_info["dragons_a"] = match.extra.get("dragons_a", 0)
                                match_info["dragons_b"] = match.extra.get("dragons_b", 0)
                                match_info["towers_a"] = match.extra.get("towers_a", 0)
                                match_info["towers_b"] = match.extra.get("towers_b", 0)
                                match_info["barons_a"] = match.extra.get("barons_a", 0)
                                match_info["barons_b"] = match.extra.get("barons_b", 0)
                                match_info["champions_a"] = match.extra.get("champions_a", [])
                                match_info["champions_b"] = match.extra.get("champions_b", [])
                        elif match.game == "valorant":
                            match_info["maps_score"] = f"{match.score_a}-{match.score_b}"
                            if match.extra.get("round_score"):
                                match_info["round_score"] = match.extra["round_score"]
                            map_stats = match.extra.get("map_stats", [])
                            if map_stats:
                                match_info["map_details"] = []
                                for ms in map_stats:
                                    t1_top = sorted(ms.get("team1_players", []), key=lambda p: -p.get("acs", 0))[:1]
                                    t2_top = sorted(ms.get("team2_players", []), key=lambda p: -p.get("acs", 0))[:1]
                                    t1_str = f"{t1_top[0]['name']}({t1_top[0]['agent']}) {t1_top[0]['kills']}/{t1_top[0]['deaths']}" if t1_top else ""
                                    t2_str = f"{t2_top[0]['name']}({t2_top[0]['agent']}) {t2_top[0]['kills']}/{t2_top[0]['deaths']}" if t2_top else ""
                                    match_info["map_details"].append(f"{ms.get('map','?')} {ms.get('score','')} | MVP: {t1_str} vs {t2_str}")
                        # Prior intelligence
                        if match.extra.get("prior_prob_a") is not None:
                            match_info["prior"] = round(match.extra["prior_prob_a"], 3)
                            match_info["prior_source"] = match.extra.get("prior_source", "")
                        # Watch links — prefer PandaScore stream, fallback to feed stream
                        if match.extra.get("stream_url"):
                            match_info["stream_url"] = match.extra["stream_url"]
                        if match.extra.get("watch_url"):
                            match_info["watch_url"] = match.extra["watch_url"]
                        if match.extra.get("bo3gg_url"):
                            match_info["bo3gg_url"] = match.extra["bo3gg_url"]
                        # Resolve BOTH series and map_winner markets for this match.
                        # This separation lets the dashboard render a parent (series) + child (map) row
                        # and prevents the two from being confused when deciding trade context.
                        def _find_market_by_teams(store, ta, tb):
                            """Look up a market in `store` by match_id first, then team-name match."""
                            # `match` is the closure — bind locally
                            for mid2, mkt2 in store.items():
                                if mid2 == match.match_id:
                                    return mkt2
                            for mid2, mkt2 in store.items():
                                if (teams_match(ta, mkt2.team_a) and teams_match(tb, mkt2.team_b)) or \
                                   (teams_match(ta, mkt2.team_b) and teams_match(tb, mkt2.team_a)):
                                    return mkt2
                            return None

                        # Primary store holds series when available; filter to series-type only
                        series_mkt = None
                        primary = self.latency_analyzer._match_to_market.get(match.match_id)
                        if primary and getattr(primary, '_market_type', 'series') == 'series':
                            series_mkt = primary
                        if not series_mkt:
                            for mid2, mkt2 in self.latency_analyzer._match_to_market.items():
                                if getattr(mkt2, '_market_type', 'series') != 'series':
                                    continue
                                if (teams_match(match.team_a, mkt2.team_a) and teams_match(match.team_b, mkt2.team_b)) or \
                                   (teams_match(match.team_a, mkt2.team_b) and teams_match(match.team_b, mkt2.team_a)):
                                    series_mkt = mkt2
                                    break
                        map_mkt = _find_market_by_teams(self.latency_analyzer._match_to_map_winner, match.team_a, match.team_b)

                        # Build dashboard info for each market we found
                        if series_mkt:
                            match_info["series"] = self._build_market_info(series_mkt)
                        if map_mkt:
                            match_info["map_winner"] = self._build_market_info(map_mkt)

                        # Legacy top-level fields (for bet matching, signals, older dashboard code).
                        # Primary is series if we have it, else map_winner, else nothing.
                        primary_info = match_info.get("series") or match_info.get("map_winner")
                        linked_market = series_mkt or map_mkt
                        if primary_info:
                            match_info["market_type"] = "series" if series_mkt else "map_winner"
                            match_info["market_price_a"] = primary_info.get("price_a", 0.5)
                            match_info["market_price_b"] = primary_info.get("price_b", 0.5)
                            match_info["best_bid_a"] = primary_info.get("bid_a", 0)
                            match_info["best_ask_a"] = primary_info.get("ask_a", 0)
                            match_info["spread"] = primary_info.get("spread", 0)
                            match_info["volume"] = primary_info.get("volume", 0)
                            match_info["liquidity"] = primary_info.get("liquidity", 0)
                            if primary_info.get("polymarket_url"):
                                match_info["polymarket_url"] = primary_info["polymarket_url"]
                        elif match.extra.get("prop_markets"):
                            # Fallback link to a prop market if no series/map available
                            first_prop = match.extra["prop_markets"][0]
                            if first_prop.get("event_slug"):
                                match_info["polymarket_url"] = f"https://polymarket.com/event/{first_prop['event_slug']}"

                        open_bets = []
                        if linked_market:
                            open_bets = [p for p in self.executor.positions
                                         if not p.resolved and p.market_id == linked_market.market_id]
                        if not open_bets:
                            # Fallback: match by team name (handles cross-session positions)
                            ta = match.team_a.lower()
                            tb = match.team_b.lower()
                            if ta and tb:
                                open_bets = [p for p in self.executor.positions
                                             if not p.resolved and (
                                                 teams_match(p.team, match.team_a)
                                                 or teams_match(p.team, match.team_b))]
                        if open_bets:
                            match_info["has_bet"] = True
                            match_info["bet_total"] = round(sum(p.amount for p in open_bets), 2)
                            match_info["bet_team"] = open_bets[0].team
                            match_info["bet_count"] = len(open_bets)
                            # Tag which market the bet is on (series moneyline vs map winner)
                            # so the dashboard can show "BET on Vitality (SERIES)" vs "(MAP)"
                            bet_market_id = open_bets[0].market_id
                            bet_mtype = "series"
                            if series_mkt and str(series_mkt.market_id) == str(bet_market_id):
                                bet_mtype = "series"
                            elif map_mkt and str(map_mkt.market_id) == str(bet_market_id):
                                bet_mtype = "map"
                            match_info["bet_market_type"] = bet_mtype
                        # Include prop markets (skip odd/even — unanalyzable)
                        props = match.extra.get("prop_markets", [])
                        if props:
                            filtered_props = [p for p in props
                                              if p["team_a"].lower() not in ("odd", "even")]
                            if filtered_props:
                                match_info["prop_markets"] = [
                                    {"question": p["question"][:50], "price_a": round(p["price_a"], 3),
                                     "price_b": round(p["price_b"], 3), "team_a": p["team_a"],
                                     "team_b": p["team_b"], "volume": p["volume"]}
                                    for p in filtered_props
                                ]
                            match_info["bets"] = [
                                {"team": p.team, "amount": round(p.amount, 2),
                                 "fill": round(p.fill_price, 3),
                                 "conf": round(p.signal_confidence, 2),
                                 "edge": round(p.our_price_at_entry - p.market_price_at_entry, 3),
                                 "analysis": p.analysis if hasattr(p, 'analysis') and p.analysis else None}
                                for p in open_bets
                            ]
                        # Last Claude decision for this match (expire after 10 min)
                        decision = self.latency_analyzer._match_decisions.get(match.match_id)
                        if decision and (now_ts - decision.get("time", 0)) < 600:
                            match_info["decision"] = decision

                        # If no stream found, use Polymarket page (embeds correct stream)
                        if not match_info.get("stream_url") and match_info.get("polymarket_url"):
                            match_info["stream_url"] = match_info["polymarket_url"]

                        # Deduplicate: same teams from multiple feeds → keep most recent + richest
                        from team_match import normalize
                        dedup_key = "_".join(sorted([normalize(match.team_a), normalize(match.team_b)]))
                        existing = _seen_matchups.get(dedup_key)
                        if existing:
                            # Prefer: most recent event time > then richest data
                            new_time = match.last_event_time or match.started_at
                            old_time = existing.get("_last_event_time", 0)
                            new_richness = (1 if match_info.get("gold_lead") else 0) + (1 if match_info.get("kills_a") else 0) + (1 if match_info.get("round_score") else 0)
                            old_richness = (1 if existing.get("gold_lead") else 0) + (1 if existing.get("kills_a") else 0) + (1 if existing.get("round_score") else 0)

                            # Most recent wins, tie-break on richness
                            replace = False
                            if new_time > old_time + 30:  # >30s newer = clearly more recent
                                replace = True
                            elif abs(new_time - old_time) <= 30 and new_richness > old_richness:
                                replace = True  # same time, richer data wins

                            if replace:
                                # Merge important fields from old entry before replacing
                                # Also keep higher series score (don't regress from 2-0 to 0-0)
                                old_score = existing.get("score", "0-0").split("-")
                                new_score = match_info.get("score", "0-0").split("-")
                                try:
                                    old_total = int(old_score[0]) + int(old_score[1])
                                    new_total = int(new_score[0]) + int(new_score[1])
                                    if old_total > new_total:
                                        match_info["score"] = existing["score"]
                                except (ValueError, IndexError):
                                    pass

                                for merge_key in ["polymarket_url", "stream_url", "watch_url", "bo3gg_url", "prior", "prior_source", "has_bet", "bet_total", "bet_team", "bet_count", "bets", "prop_markets", "decision"]:
                                    if existing.get(merge_key) and not match_info.get(merge_key):
                                        match_info[merge_key] = existing[merge_key]
                                self.state.active_matches.remove(existing)
                                match_info["_last_event_time"] = new_time
                                self.state.active_matches.append(match_info)
                                _seen_matchups[dedup_key] = match_info
                        else:
                            match_info["_last_event_time"] = match.last_event_time or match.started_at
                            self.state.active_matches.append(match_info)
                            _seen_matchups[dedup_key] = match_info

            total_feed_matches = sum(len(f._matches) for f in self.feeds.values())
            total_live = sum(1 for f in self.feeds.values() for m in f._matches.values() if m.is_live)
            if total_feed_matches != len(self.state.active_matches):
                logger.info(f"[STATE] active_matches={len(self.state.active_matches)} | feed_matches={total_feed_matches} | live={total_live} | linked={len(linked_ids)}")
            self.state.trade_stats = self.executor.get_stats()
            self.state.latency_stats = self.latency_analyzer.get_stats()
            mtm_len = len(self.latency_analyzer._match_to_market)
            self.state.markets_tracked = mtm_len
            if self.edge_analyst:
                self.state.claude_stats.update(self.edge_analyst.get_stats())
            else:
                self.state.claude_stats = {"claude_active": False}
            # Add continuous analyzer stats
            self.state.claude_stats.update(self.match_analyzer.get_stats())
            self.state.claude_stats.update(self.recorder.get_stats())
            self.state.pending_orders = self.executor.get_pending_stats()
            # Price impact data
            self.price_impact.tick()
            self.state.price_impacts = self.price_impact.get_recent_impacts(20)
            self.state.impact_summary = self.price_impact.get_summary()
            # WebSocket stats
            ws = self.polymarket_ws
            ws_age = time.time() - ws.last_message_time if ws.last_message_time > 0 else 999
            self.state.claude_stats["ws_connected"] = ws._connected
            self.state.claude_stats["ws_messages"] = ws.message_count
            self.state.claude_stats["ws_tokens"] = len(ws._subscribed_tokens)
            self.state.claude_stats["ws_fresh"] = ws_age < 10

        for cb in self._state_callbacks:
            try:
                cb(self.state)
            except Exception:
                pass

    def _on_raw_bo3_snapshot(self, match_id: str, raw_payload: dict):
        """Record the full bo3.gg payload as a SNAPSHOT_MATCH_UPDATE line.

        Handles BOTH provider shapes we consume:

          1. WebSocket payload (cs2_bo3_ws.py):
             {"id": ..., "live_updates": {...rich...}, "team1": {...}, "team2": {...}}

          2. REST snapshot (cs2_bo3.py polling feed):
             {"team_one": {...}, "team_two": {...}, "round_phase": ..., ...}
             (the rich fields are at the TOP level; no live_updates wrapper)

        Shape-compatible with the curated Test Data/ SNAPSHOT_MATCH_UPDATE
        records our backtester was built against: team_one / team_two with
        player_states, round_phase, round_time, is_bomb_planted, etc.
        """
        try:
            if not self.recorder or not match_id:
                return

            # Decide which shape we have by sniffing the live_updates key
            live = raw_payload.get("live_updates")
            if isinstance(live, dict) and live:
                # Shape 1: WebSocket. Lift live_updates to the top level + keep
                # the team1/team2 name blocks for META recovery.
                base = dict(live)
                team1 = raw_payload.get("team1") or {}
                team2 = raw_payload.get("team2") or {}
                if isinstance(team1, dict) and team1:
                    base["team_one_meta"] = team1
                if isinstance(team2, dict) and team2:
                    base["team_two_meta"] = team2
            else:
                # Shape 2: REST polling. The payload IS already the snapshot.
                # Copy it as-is (this carries team_one, team_two with
                # player_states, round_phase, is_bomb_planted, round_time_*,
                # equipment_value, match_fixture, etc).
                base = dict(raw_payload)

            # Extract team names for the recorder's filename generator
            t1 = base.get("team_one") or base.get("team_one_meta") or {}
            t2 = base.get("team_two") or base.get("team_two_meta") or {}
            ta = (t1.get("name", "") if isinstance(t1, dict) else "") or ""
            tb = (t2.get("name", "") if isinstance(t2, dict) else "") or ""

            record = base
            record["message_type"] = "SNAPSHOT_MATCH_UPDATE"
            record["match_id"] = match_id
            self.recorder._write(match_id, "SNAPSHOT_MATCH_UPDATE", record, ta, tb)
        except Exception as e:
            logger.debug(f"[RAW-SNAP] record failed: {e}")

    def _on_game_event(self, event: GameEvent):
        """Handle incoming game event from any feed."""
        # Feed ALL events into continuous analyzer AND recorder
        _raw_types = {"round_end", "map_win", "match_end", "teamfight_won", "kill_streak", "score_update", "objective_taken", "economy_shift"}
        if event.event_type.value in _raw_types and event.game in ("cs2", "dota2"):
            self.match_analyzer.add_event(event.match_id, event.event_type.value, event.description, event.team)

            # Record ALL events for training data
            ms = event.match_state
            if True:  # always record, even without match_state
                # Record meta once per match (only when first real event comes)
                market = self.latency_analyzer._match_to_market.get(event.match_id)
                ta = ms.team_a if ms else ""
                tb = ms.team_b if ms else ""
                ex = ms.extra or {} if ms else {}
                self.recorder.record_meta(
                    event.match_id, ta, tb, event.game,
                    market_slug=getattr(market, 'market_id', '') if market else '',
                    token_id_a=getattr(market, 'token_id_a', '') if market else '',
                    token_id_b=getattr(market, 'token_id_b', '') if market else '',
                )
                # Record game event with full state
                self.recorder.record_game_event(event.match_id, event.event_type.value, {
                    "event_type": event.event_type.value,
                    "team": event.team,
                    "description": event.description,
                    "team_one": {
                        "name": ta, "score": ms.round_score_a if ms else 0,
                        "match_score": ms.score_a if ms else 0,
                        "equipment_value": ex.get("team_a_money", 0),
                    },
                    "team_two": {
                        "name": tb, "score": ms.round_score_b if ms else 0,
                        "match_score": ms.score_b if ms else 0,
                        "equipment_value": ex.get("team_b_money", 0),
                    },
                    "kills_a": ex.get("kills_a", 0), "kills_b": ex.get("kills_b", 0),
                    "gold_lead": ex.get("gold_lead", 0), "game_minutes": ex.get("game_minutes", 0),
                    "map_name": ex.get("map_name", ""),
                    "side": ex.get("team_a_current_side", ""),
                }, ta, tb)
                # Record price snapshot if market exists — BOTH tokens
                # (previously only token_a was recorded — left 83% of our
                # training files with only one side's orderbook, unusable for
                # SFT generation which requires both bid+ask for both teams).
                if market and self.polymarket_ws:
                    ws_a = self.polymarket_ws.get_price(market.token_id_a)
                    ws_b = self.polymarket_ws.get_price(market.token_id_b) if market.token_id_b else None
                    if ws_a:
                        self.recorder.record_best_bid_ask(
                            event.match_id, market.token_id_a,
                            ws_a.get("best_bid", 0), ws_a.get("best_ask", 0),
                            ms.team_a, ms.team_b,
                        )
                    if ws_b:
                        self.recorder.record_best_bid_ask(
                            event.match_id, market.token_id_b,
                            ws_b.get("best_bid", 0), ws_b.get("best_ask", 0),
                            ms.team_a, ms.team_b,
                        )

        # Capture ALL raw events for live feed display
        if event.event_type.value in _raw_types and event.game in ("cs2", "dota2"):
            ms = event.match_state
            raw = {
                "game": event.game,
                "type": event.event_type.value,
                "team": event.team,
                "description": event.description[:100],
                "time": event.timestamp,
                "team_a": ms.team_a if ms else "",
                "team_b": ms.team_b if ms else "",
                "score": f"{ms.round_score_a}-{ms.round_score_b}" if ms and (ms.round_score_a or ms.round_score_b) else f"{ms.score_a}-{ms.score_b}" if ms else "",
            }
            with self._state_lock:
                self.state.raw_feed_events.append(raw)
                if len(self.state.raw_feed_events) > 200:
                    self.state.raw_feed_events = self.state.raw_feed_events[-100:]

        # Check if we have a linked market for latency comparison
        market = self.latency_analyzer._match_to_market.get(event.match_id)
        our_prob = event.match_state.win_probability_a if event.match_state else 0.5
        market_prob = 0.0
        edge = 0.0
        has_market = False
        faster = None  # True = we're faster, False = market already moved

        if market:
            has_market = True
            market_prob = market.price_a
            edge = our_prob - market_prob
            # If market hasn't moved but we detect a shift, we're faster (green)
            staleness = time.time() - market.last_price_update if market.last_price_update > 0 else 999
            if abs(edge) > 0.03 and staleness > 3:
                faster = True   # green — we have info the market doesn't
            elif abs(edge) < 0.02:
                faster = False  # red — market already priced it in

        # Check if we have an open bet on this match
        has_bet = False
        if event.match_state:
            ta = event.match_state.team_a.lower()
            tb = event.match_state.team_b.lower()
            bet_teams = {p.team.lower() for p in self.executor.positions if not p.resolved}
            has_bet = any(ta in bt or bt in ta or tb in bt or bt in tb for bt in bet_teams)

        # Only log meaningful events (skip duplicates like economy_shift which fires with round_end)
        _show_events = {"round_end", "map_win", "match_end", "teamfight_won", "kill_streak", "score_update", "objective_taken"}
        if market and not event.match_id.startswith("mkt_") and event.game in ("cs2", "dota2") and event.event_type.value in _show_events:
            with self._state_lock:
                self.state.recent_events.append({
                    "game": event.game,
                    "type": event.event_type.value,
                    "team": event.team,
                    "description": event.description,
                    "impact": round(event.impact, 3),
                    "time": event.timestamp,
                    "our_prob": round(our_prob, 3),
                    "market_prob": round(market_prob, 3),
                    "edge": round(edge, 3),
                    "has_market": has_market,
                    "has_bet": has_bet,
                    "faster": faster,
                })
                if len(self.state.recent_events) > 200:
                    self.state.recent_events = self.state.recent_events[-100:]

        # Persist match events to DB — use latency analyzer's market mapping (may be linked by now)
        _market_for_persist = market or self.latency_analyzer._match_to_market.get(event.match_id)
        if _market_for_persist and event.event_type.value in _show_events:
            try:
                ms = event.match_state
                self.executor.db.save_match_event({
                    "timestamp": event.timestamp,
                    "match_id": event.match_id,
                    "game": event.game,
                    "event_type": event.event_type.value,
                    "team": event.team,
                    "description": event.description,
                    "team_a": ms.team_a if ms else "",
                    "team_b": ms.team_b if ms else "",
                    "score_a": ms.score_a if ms else 0,
                    "score_b": ms.score_b if ms else 0,
                    "round_a": ms.round_score_a if ms else 0,
                    "round_b": ms.round_score_b if ms else 0,
                    "kills_a": ms.extra.get("kills_a", 0) if ms and ms.extra else 0,
                    "kills_b": ms.extra.get("kills_b", 0) if ms and ms.extra else 0,
                    "gold_lead": ms.extra.get("gold_lead", 0) if ms and ms.extra else 0,
                    "economy_a": ms.extra.get("team_a_money", 0) if ms and ms.extra else 0,
                    "economy_b": ms.extra.get("team_b_money", 0) if ms and ms.extra else 0,
                    "market_price": _market_for_persist.price_a if _market_for_persist else 0,
                    "had_market": True,
                })
            except Exception:
                pass

        # Record price impact for this event (track how market moves after each event)
        if market and not event.match_id.startswith("mkt_") and event.game in ("cs2", "dota2"):
            # Track impact on all tokens for this market
            if market.token_id_a:
                self.price_impact.record_event(event, market, market.token_id_a)

        # Resolve positions when a match ends
        if event.event_type == EventType.MATCH_END and event.match_state:
            winning_team_name = event.match_state.team_a if event.team == "a" else event.match_state.team_b
            # Find all markets linked to this match and resolve positions
            market = self.latency_analyzer._match_to_market.get(event.match_id)
            if market:
                self.executor.resolve_by_match(market.market_id, winning_team_name)

        # Feed to latency analyzer
        signal = self.latency_analyzer.process_event(event)

        self._update_state()

    def _log_shadow_trade(self, signal) -> None:
        """Record what the bot WOULD have done — for shadow-PnL tracking.

        Writes to data/trades.db `shadow_trades` table (created if missing).
        The Telegram admin can read this via /positions and /pnl when in
        shadow mode. When we eventually compare shadow vs live PnL during
        the Phase 5 cutover, this is the source of truth for 'what would
        have happened'.
        """
        import sqlite3
        import time
        from pathlib import Path
        try:
            analysis = getattr(signal, '_analysis', None) or {}
            db_path = Path(__file__).resolve().parent / "data" / "trades.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path))
            conn.execute("""
                CREATE TABLE IF NOT EXISTS shadow_trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts REAL NOT NULL,
                    match_id TEXT,
                    token_id TEXT,
                    team TEXT,
                    game TEXT,
                    fill_price REAL,
                    market_price REAL,
                    bet_usd REAL,
                    confidence REAL,
                    tp_pct REAL,
                    sl_pct REAL,
                    trigger TEXT,
                    llm_reason TEXT,
                    llm_model TEXT
                )
            """)
            conn.execute(
                "INSERT INTO shadow_trades (ts, match_id, token_id, team, game, "
                "fill_price, market_price, bet_usd, confidence, tp_pct, sl_pct, "
                "trigger, llm_reason, llm_model) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    time.time(),
                    getattr(signal.market, "match_id", "") if getattr(signal, "market", None) else "",
                    signal.token_id,
                    signal.team_name,
                    signal.game,
                    float(signal.our_price or 0),
                    float(signal.market_price or 0),
                    float(signal.bet_amount or 0),
                    float(signal.confidence or 0),
                    float(getattr(signal, "_tp_pct", 0) or 0),
                    float(getattr(signal, "_sl_pct", 0) or 0),
                    signal.reason[:200] if signal.reason else "",
                    (analysis.get("reason") or "")[:200],
                    analysis.get("model", ""),
                )
            )
            conn.commit()
            conn.close()
            logger.info(
                f"[SHADOW] {signal.team_name} would buy @ {signal.our_price:.3f} "
                f"bet=${signal.bet_amount:.0f} conf={signal.confidence:.2f} — logged"
            )
        except Exception as e:
            logger.warning(f"[SHADOW] log failed: {e}")

    def _on_trade_signal(self, signal: TradeSignal):
        """Handle trade signal from latency analyzer."""
        import os as _os
        from pathlib import Path as _P
        _repo = _P(__file__).resolve().parent

        # ─── SAFETY GATES (ordered: hardest stop first) ──────────────────────

        # 1. HARD_STOP.lock — permanent kill until the file is manually removed.
        #    Written by /hardstop Telegram command or by risk.py when drawdown
        #    hits HARD_STOP_PCT.
        _hs = _repo / "HARD_STOP.lock"
        if _hs.exists():
            logger.warning(f"[HARD_STOP] {signal.team_name} dropped — lock file present at {_hs}")
            return

        # 2. RECORD_ONLY — no trading at all. Feeds + recorder keep running.
        if _os.environ.get("RECORD_ONLY", "").lower() in ("1", "true", "yes"):
            return

        # 3. /pause flag — user paused via Telegram. Existing positions stay
        #    monitored (TP/SL logic runs elsewhere), new trades blocked.
        if (_repo / ".bot_paused").exists():
            logger.info(f"[PAUSED] {signal.team_name} dropped — .bot_paused flag present")
            return

        # 4. SHADOW_TRADING — log the decision but do NOT place an order.
        #    Captures what the bot WOULD have done for shadow-PnL tracking.
        _shadow = _os.environ.get("SHADOW_TRADING", "true").lower() in ("1", "true", "yes")
        if _shadow:
            self._log_shadow_trade(signal)
            return
        with self._state_lock:
            self.state.recent_signals.append({
                "game": signal.game,
                "team": signal.team_name,
                "edge": round(signal.edge, 3),
                "confidence": round(signal.confidence, 2),
                "amount": signal.bet_amount,
                "market_price": round(signal.market_price, 3),
                "our_price": round(signal.our_price, 3),
                "time": signal.timestamp,
            })

        analysis = getattr(signal, '_analysis', None) or {}

        # Enforce: all trades MUST have qwen analysis attached. Refuse any signal that
        # slipped through without going through the edge_analyst decision path.
        if not analysis:
            logger.warning(
                f"[GUARD] Refusing trade without AI analysis: {signal.team_name} ({signal.game}) | "
                f"reason={signal.reason[:80]}"
            )
            return

        # Stamp which model actually produced this decision so the dashboard can label it.
        if self.edge_analyst is not None:
            analysis["model"] = getattr(self.edge_analyst, "_ollama_model", None) \
                if getattr(self.edge_analyst, "_use_local", False) \
                else getattr(self.edge_analyst, "_model_claude", "claude")

        # AI-driven mode: max_price = 0.99 (executor's hard floor for tradeability).
        # Polymarket charges 0% on entry, 2% on net winnings — so any buy price < 1.00 can
        # be profitable with the right tp_pct. The mechanical bid<3¢/>97¢ check in latency.py
        # already blocks about-to-resolve tokens. Let qwen pick the rest.
        max_price = 0.99
        # Pre-trade price refresh + sanity gate. Always poll CLOB REST right
        # before entering, and BLOCK if the freshly-polled book shows conditions
        # that would immediately fire SL. (Bug it prevents: buying Vitality at
        # 0.64 when fresh book showed bid=0.05/ask=0.96, 92% spread — market had
        # already priced the team as losing, we ignored it, SL fired at 0.05
        # for -92% × 2 = -$138.)
        try:
            if hasattr(self, "_rest_refresh_token") and signal.token_id:
                refreshed = self._rest_refresh_token(signal.token_id)
                snap = self.polymarket_ws.get_price(signal.token_id) or {}
                fresh_bid = snap.get("best_bid", 0) or 0
                fresh_ask = snap.get("best_ask", 0) or 0
                fresh_spread = (fresh_ask - fresh_bid) if (fresh_ask > fresh_bid) else 1.0
                logger.info(f"[PRE-TRADE-POLL] {signal.team_name} tok={signal.token_id[:10]} refreshed={refreshed} bid={fresh_bid:.3f} ask={fresh_ask:.3f}")
                signal_price = getattr(signal, "market_price", 0) or 0
                # Block 1: crashed bid (market already priced this team as losing)
                if fresh_bid < 0.10:
                    logger.warning(f"[PRE-TRADE-BLOCK] {signal.team_name} fresh bid={fresh_bid:.3f} < 0.10 — market crashed, aborting")
                    return
                # Block 2: wide spread trap
                if fresh_spread > 0.20:
                    logger.warning(f"[PRE-TRADE-BLOCK] {signal.team_name} fresh spread={fresh_spread:.3f} > 0.20 — SL trap, aborting")
                    return
                # Block 3: fill-price drift — if fresh ask is materially above the signal's
                # assumed fill, we'd be filling higher than the edge implies. Reject >5% drift.
                if signal_price > 0 and fresh_ask > 0 and fresh_ask > signal_price * 1.05:
                    logger.warning(f"[PRE-TRADE-BLOCK] {signal.team_name} fresh ask={fresh_ask:.3f} drifted >5% above signal={signal_price:.3f} — aborting")
                    return
        except Exception as _e:
            logger.debug(f"[PRE-TRADE-POLL] error: {_e}")
        # Qwen-chosen exit targets (fall back to defaults if missing)
        tp_pct = getattr(signal, '_tp_pct', 0) or None
        sl_pct = getattr(signal, '_sl_pct', 0) or None
        position = self.executor.execute_trade(
            token_id=signal.token_id,
            team=signal.team_name,
            game=signal.game,
            market_id=signal.market.market_id,
            amount=signal.bet_amount,
            confidence=signal.confidence,
            latency_edge_ms=signal.latency_edge_ms,
            max_price=max_price,
            market_price=signal.market_price,
            our_price=signal.our_price,
            analysis=analysis,
            trigger_event=signal.reason,
            ai_tp_pct=tp_pct,
            ai_sl_pct=sl_pct,
        )

        if position:
            logger.info(f"Position opened: {position.order_id}")

        self._update_state()

    async def _run_feeds(self):
        """Start all game feeds concurrently."""
        tasks = []
        for game, feed in self.feeds.items():
            logger.info(f"Starting {game} feed...")
            tasks.append(asyncio.create_task(feed.connect()))

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _market_scanner(self):
        """Periodically scan for new esports markets and auto-link to matches."""
        while self._running:
            try:
                loop = asyncio.get_event_loop()
                markets = await loop.run_in_executor(None, self.market_finder.fetch_esports_markets)
                logger.info(f"Scanning markets: found {len(markets)} active esports markets")

                # Try to link markets to tracked matches
                # Prefer moneyline but allow handicap as fallback
                skip_keywords = ["total maps", "over", "under",
                                 "prop", "first blood", "total kills",
                                 "handicap", "total rounds", "total games",
                                 "odd", "even", "exact"]

                for feed_key, feed in self.feeds.items():
                    # Use the feed's actual game type, not the dict key (which may have suffixes)
                    game = feed.game
                    for match_id, match_state in feed._matches.items():
                        # Re-evaluate if current link is series but a map_winner exists
                        existing = self.latency_analyzer._match_to_market.get(match_id)
                        if existing and getattr(existing, '_market_type', '') == 'map_winner':
                            continue  # already on best type
                        # If linked to series, still check for map_winner upgrade

                        team_a = match_state.team_a.lower()
                        team_b = match_state.team_b.lower()

                        candidates = []
                        for market in markets:
                            if market.game != game:
                                continue
                            # Skip de facto resolved markets (price near 0 or 1)
                            if market.price_a < 0.02 or market.price_a > 0.98:
                                continue
                            q = market.question.lower()
                            # Match: BOTH feed teams must appear in the market
                            # Check question text first, then team fields
                            a_in_q = team_a and team_in_text(match_state.team_a, q)
                            b_in_q = team_b and team_in_text(match_state.team_b, q)

                            # Also check market team_a/team_b fields (handles abbreviations like AVE)
                            a_in_fields = team_a and (
                                teams_match(match_state.team_a, market.team_a) or
                                teams_match(match_state.team_a, market.team_b)
                            )
                            b_in_fields = team_b and (
                                teams_match(match_state.team_b, market.team_a) or
                                teams_match(match_state.team_b, market.team_b)
                            )

                            a_match = a_in_q or a_in_fields
                            b_match = b_in_q or b_in_fields

                            # BOTH teams must match — prevents matching "A vs B" to "A vs C" market
                            if not (a_match and b_match):
                                continue

                            # Extra check: if matching via team fields, make sure it's the RIGHT market
                            # Don't match "ALIS vs VP.Prodigy" to "Spirit Academy vs VP.Prodigy"
                            if a_in_fields and not a_in_q:
                                # team_a matched via field — verify it matches the CORRECT market team
                                if not (teams_match(match_state.team_a, market.team_a) or teams_match(match_state.team_a, market.team_b)):
                                    continue

                            # Hard skip: handicap/prop markets
                            if any(kw in q for kw in skip_keywords):
                                continue

                            # Score candidates — PREFER map/game winner (more volatile to events)
                            score = market.volume / 100000
                            is_map_winner = ('winner' in q and ('map' in q or 'game' in q))
                            is_series = any(kw in q for kw in ["bo3", "bo1", "bo5", "(bo"])
                            if is_map_winner:
                                score += 20  # highest priority — most reactive to round events
                            elif is_series:
                                score += 5   # fallback — less reactive but always available
                            market._market_type = "map_winner" if is_map_winner else "series" if is_series else "other"
                            candidates.append((score, market))

                        if candidates:
                            candidates.sort(key=lambda x: -x[0])
                            # Link the top candidate of EACH type (series + map_winner) so the
                            # dashboard can show both as parent/child rows and the latency analyzer
                            # can differentiate between the two. Previously we only linked the
                            # single best, which hid the other market from view.
                            best = candidates[0][1]
                            linked_types = set()
                            for _, cand in candidates:
                                ctype = getattr(cand, '_market_type', 'other')
                                if ctype in ('series', 'map_winner') and ctype not in linked_types:
                                    self.latency_analyzer.link_match_to_market(match_id, cand)
                                    logger.info(f"Linked [{ctype}]: {match_state.team_a} vs {match_state.team_b} → {cand.question[:50]}")
                                    linked_types.add(ctype)
                                    if {'series', 'map_winner'}.issubset(linked_types):
                                        break

                            # Set market price as prior — captures crowd wisdom about team strength
                            if "prior_prob_a" not in match_state.extra:
                                aligned = self._teams_aligned(match_state, best)
                                if aligned is True:
                                    prior = best.price_a if best.price_a > 0.01 else 0.5
                                elif aligned is False:
                                    prior = 1.0 - best.price_a if best.price_a < 0.99 else 0.5
                                else:
                                    prior = 0.5  # can't determine alignment
                                match_state.extra["prior_prob_a"] = max(0.10, min(0.90, prior))
                                match_state.extra["prior_source"] = "polymarket"
                                logger.info(f"Prior set: {match_state.team_a} vs {match_state.team_b} → P(A)={prior:.3f} from market")
                        elif match_id not in self._unlinked_logged:
                            self._unlinked_logged.add(match_id)
                            logger.info(f"No moneyline market found for {match_state.team_a} vs {match_state.team_b} ({game})")

                # Also link PROP markets (over/under, rampage, odd/even) to live matches
                # Only prop types where analysis helps (skip odd/even — pure coin flip)
                prop_outcomes = {"over", "under", "yes", "no"}
                for market in markets:
                    if market.team_a.lower() not in prop_outcomes:
                        continue  # not a prop market
                    if market.market_id in [m.market_id for m in self.latency_analyzer._match_to_market.values()]:
                        continue  # already linked

                    # Try to find the live match this prop belongs to
                    # Use the event slug which contains team abbreviations
                    slug = market.event_slug.lower()
                    q = market.question.lower()

                    for game_key, feed in self.feeds.items():
                        if market.game != game_key:
                            continue
                        for match_id, match_state in feed._matches.items():
                            if not match_state.is_live:
                                continue
                            # Check if slug contains team abbreviations
                            ta = match_state.team_a.lower().replace(" ", "")
                            tb = match_state.team_b.lower().replace(" ", "")
                            # Match on first 3+ chars of team names in slug
                            ta_short = ta[:4] if len(ta) >= 4 else ta
                            tb_short = tb[:4] if len(tb) >= 4 else tb
                            slug_match = (ta_short in slug or tb_short in slug) and len(ta_short) >= 3

                            # Also check if team names appear in the question
                            q_match = (team_in_text(match_state.team_a, q) or
                                       team_in_text(match_state.team_b, q))

                            if slug_match or q_match:
                                # Link this prop market to the match
                                # Store as a "prop_markets" list in match extra
                                prop_key = f"prop_{market.market_id}"
                                if prop_key not in match_state.extra:
                                    match_state.extra[prop_key] = True
                                    match_state.extra.setdefault("prop_markets", []).append({
                                        "market_id": market.market_id,
                                        "question": market.question,
                                        "team_a": market.team_a,
                                        "team_b": market.team_b,
                                        "price_a": market.price_a,
                                        "price_b": market.price_b,
                                        "volume": market.volume,
                                        "token_id_a": market.token_id_a,
                                        "token_id_b": market.token_id_b,
                                        "event_slug": market.event_slug,
                                    })
                                    # Also link in latency analyzer so events trigger analysis
                                    self.latency_analyzer._match_to_market.setdefault(
                                        f"prop_{match_id}_{market.market_id}", market)
                                    logger.info(f"[PROP] Linked: {market.question[:50]} → {match_state.team_a} vs {match_state.team_b}")
                                break

                # Auto-start OCR for bettable matches with stream URLs but no round data
                for feed_key, feed in self.feeds.items():
                    if not feed_key.endswith("_ocr"):
                        continue
                    ocr_feed = feed
                    game = ocr_feed.game
                    for other_key, other_feed in self.feeds.items():
                        if other_feed.game != game or other_key.endswith("_ocr"):
                            continue
                        for mid, ms in other_feed._matches.items():
                            if not ms.is_live or mid in ocr_feed._streams:
                                continue
                            stream = ms.extra.get("stream_url", "")
                            if not stream or not any(p in stream for p in ["twitch.tv", "youtube.com", "youtu.be", "kick.com"]):
                                continue
                            # OCR priority: always for LoL/Valorant (no direct API events)
                            # For CS2/Dota2: only as backup when no other data
                            has_market = mid in self.latency_analyzer._match_to_market
                            has_bet = any(not p.resolved and teams_match(p.team, ms.team_a) or teams_match(p.team, ms.team_b)
                                         for p in self.executor.positions)
                            has_round_data = ms.round_score_a + ms.round_score_b > 0
                            is_ocr_priority = game in ("lol", "valorant")  # primary data source
                            is_ocr_backup = game in ("cs2", "dota2") and not has_round_data  # backup only
                            if (has_market or has_bet) and (is_ocr_priority or is_ocr_backup):
                                ocr_feed.add_stream(mid, stream, ms.team_a, ms.team_b)

                # Create synthetic matches for active Polymarket markets not tracked by any feed.
                # This ensures all bettable markets appear on the dashboard for Claude analysis.
                linked_teams = set()
                for mid, mkt in self.latency_analyzer._match_to_market.items():
                    from team_match import normalize
                    key = "_".join(sorted([normalize(mkt.team_a), normalize(mkt.team_b)]))
                    linked_teams.add(key)

                # Also track teams already in feeds
                feed_teams = set()
                for feed in self.feeds.values():
                    for ms in feed._matches.values():
                        from team_match import normalize
                        key = "_".join(sorted([normalize(ms.team_a), normalize(ms.team_b)]))
                        feed_teams.add(key)

                # Skip synthetic match creation if configured
                import os as _os
                _no_synthetic = _os.environ.get("EDGE_BOT_NO_SYNTHETIC", "").lower() == "true"
                synth_count = 0
                for market in (_no_synthetic and [] or markets):
                    # Only moneyline markets
                    q = market.question.lower()
                    if not any(kw in q for kw in ["bo3", "bo1", "bo5", "(bo"]):
                        continue
                    # Skip resolved
                    if market.price_a < 0.02 or market.price_a > 0.98:
                        continue
                    from team_match import normalize
                    key = "_".join(sorted([normalize(market.team_a), normalize(market.team_b)]))
                    if key in linked_teams or key in feed_teams:
                        continue  # already tracked

                    # Create synthetic match in the appropriate PandaScore feed
                    synth_id = f"mkt_{market.market_id[:12]}"
                    ps_feed = self.feeds.get(f"{market.game}_pandascore")
                    if not ps_feed:
                        # Use any feed for this game
                        ps_feed = next((f for k, f in self.feeds.items() if f.game == market.game), None)
                    if not ps_feed:
                        continue

                    if synth_id not in ps_feed._matches:
                        from feeds.base import MatchState
                        state = MatchState(
                            match_id=synth_id, game=market.game,
                            team_a=market.team_a, team_b=market.team_b,
                            is_live=True, started_at=time.time(),
                            total_maps=3,
                            extra={"prior_prob_a": market.price_a, "prior_source": "polymarket"},
                        )
                        state.win_probability_a = market.price_a
                        ps_feed._matches[synth_id] = state
                        # Link to market
                        self.latency_analyzer.link_match_to_market(synth_id, market)
                        linked_teams.add(key)
                        synth_count += 1

                        # Emit a SCORE_UPDATE so Claude can analyze immediately
                        from feeds.base import GameEvent, EventType
                        ps_feed._emit(GameEvent(
                            event_type=EventType.SCORE_UPDATE,
                            match_id=synth_id, game=market.game, team="neutral",
                            description=f"Market: {market.team_a} vs {market.team_b} (from Polymarket)",
                            impact=0.0, match_state=state,
                        ))

                logger.info(f"[MARKETS] Synth pass: {synth_count} created | _match_to_market={len(self.latency_analyzer._match_to_market)} | no_synth={_no_synthetic}")

                # Auto-add OCR streams for LoL matches with Polymarket markets
                self._auto_add_ocr_streams()

                self._update_state()

            except Exception as e:
                logger.error(f"Market scanner error: {e}")

            await asyncio.sleep(30)  # scan every 30 seconds

    # ─── Known esports stream channels ────────────────────────────────────
    LOL_STREAMS = {
        "lck": "https://www.twitch.tv/lck",
        "lck_cl": "https://www.twitch.tv/lckclchallengers",
        "lec": "https://www.twitch.tv/lec",
        "lcs": "https://www.twitch.tv/lcs",
        "cblol": "https://www.twitch.tv/cblol",
        "lpl": "https://www.twitch.tv/lpl",
    }
    _ocr_streams_added = set()

    def _auto_add_ocr_streams(self):
        """Auto-add Twitch streams for LoL matches with Polymarket markets."""
        ocr_feed = self.feeds.get("lol_ocr")
        if not ocr_feed:
            return

        # Check linked LoL matches
        for match_id, market in self.latency_analyzer._match_to_market.items():
            if market.game != "lol":
                continue
            if match_id in self._ocr_streams_added:
                continue

            # Find match state
            state = None
            for feed in self.feeds.values():
                if match_id in feed._matches:
                    state = feed._matches[match_id]
                    break
            if not state or not state.is_live:
                continue

            # Determine which stream to use based on team/league names
            question = getattr(market, 'question', '').lower()
            stream_url = None

            if any(k in question for k in ["lck challengers", "lck cl"]):
                stream_url = self.LOL_STREAMS["lck_cl"]
            elif "lck" in question:
                stream_url = self.LOL_STREAMS["lck"]
            elif "lec" in question or "prime league" in question:
                stream_url = self.LOL_STREAMS["lec"]
            elif "lcs" in question or "north america" in question:
                stream_url = self.LOL_STREAMS["lcs"]
            elif "cblol" in question or "circuito" in question:
                stream_url = self.LOL_STREAMS["cblol"]
            elif "lpl" in question:
                stream_url = self.LOL_STREAMS["lpl"]

            if stream_url:
                ocr_feed.add_stream(match_id, stream_url, state.team_a, state.team_b)
                self._ocr_streams_added.add(match_id)
                logger.info(f"[OCR-AUTO] Added LoL stream for {state.team_a} vs {state.team_b} → {stream_url}")

    def run(self):
        """Start the bot in a background thread (used when dashboard runs in main thread)."""
        self._running = True
        self.state.status = "starting"

        if not self.executor.connect():
            if not self.dry_run:
                logger.error("Failed to connect executor — aborting")
                self.state.status = "error"
                return
            logger.warning("Executor connection failed but running in dry-run mode")

        self.state.status = "running"
        thread = threading.Thread(target=self._run_async_loop, daemon=True)
        thread.start()
        logger.info(f"Esports bot started ({'DRY-RUN' if self.dry_run else 'LIVE'} mode)")

    def run_blocking(self):
        """Run the bot's asyncio loop in the calling thread (use when dashboard runs in daemon thread)."""
        self._running = True
        self.state.status = "starting"

        if not self.executor.connect():
            if not self.dry_run:
                logger.error("Failed to connect executor — aborting")
                self.state.status = "error"
                return
            logger.warning("Executor connection failed but running in dry-run mode")

        self.state.status = "running"
        logger.info(f"Esports bot started ({'DRY-RUN' if self.dry_run else 'LIVE'} mode)")
        self._run_async_loop()

    def _run_async_loop(self):
        """Run the async event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._main_async())
        except Exception as e:
            logger.error(f"Async loop error: {e}")
            self.state.status = "error"

    async def _match_discovery_loop(self):
        """Discover live CS2 matches and auto-subscribe."""
        cs2_feed = self.feeds.get("cs2")
        if not cs2_feed:
            return

        while self._running:
            try:
                match_ids = await cs2_feed.discover_live_matches()
                for mid in match_ids:
                    if mid not in cs2_feed._subscribed_matches:
                        await cs2_feed.subscribe_match(mid)
                        logger.info(f"Auto-subscribed to CS2 match {mid}")
            except Exception as e:
                logger.error(f"Match discovery error: {e}")

            await asyncio.sleep(config.CS2_DISCOVERY_INTERVAL)

    async def _ws_subscription_loop(self):
        """Auto-subscribe to Polymarket WebSocket for all linked market tokens."""
        subscribed = set()
        while self._running:
            try:
                new_tokens = []
                for mid, mkt in self.latency_analyzer._match_to_market.items():
                    if mkt.token_id_a and mkt.token_id_a not in subscribed:
                        new_tokens.append(mkt.token_id_a)
                        subscribed.add(mkt.token_id_a)
                    if mkt.token_id_b and mkt.token_id_b not in subscribed:
                        new_tokens.append(mkt.token_id_b)
                        subscribed.add(mkt.token_id_b)
                if new_tokens:
                    await self.polymarket_ws.subscribe(new_tokens)
                    logger.info(f"[WS] Subscribed {len(new_tokens)} new tokens (total: {len(subscribed)})")
            except Exception as e:
                logger.error(f"[WS] Subscription error: {e}")
            await asyncio.sleep(10)

    async def _price_updater(self):
        """Periodically refresh CLOB prices for all linked markets (runs in thread executor)."""
        await asyncio.sleep(10)
        update_count = 0
        while self._running:
            try:
                linked = dict(self.latency_analyzer._match_to_market)
                if linked:
                    loop = asyncio.get_event_loop()
                    updated = 0
                    for match_id, market in linked.items():
                        try:
                            await loop.run_in_executor(
                                None, self.market_finder.update_market_prices, market
                            )
                            updated += 1
                        except Exception as e:
                            logger.error(f"Price update error for {market.market_id}: {e}")
                    update_count += 1
                    if update_count % 12 == 1:  # log every minute
                        logger.info(f"[PRICES] Updated {updated}/{len(linked)} linked markets")
            except Exception as e:
                logger.error(f"[PRICES] Update loop error: {e}")
            await asyncio.sleep(5)  # refresh every 5 seconds

    async def _team_intel_loop(self):
        """Disabled — edge bot doesn't use team intel."""
        pass

    async def _position_cleanup_loop(self):
        """Expire open positions whose matches are no longer live."""
        await asyncio.sleep(60)  # let feeds populate first
        while self._running:
            try:
                # Collect all currently live team names
                live_teams = set()
                for feed in self.feeds.values():
                    for match in feed._matches.values():
                        if match.is_live:
                            live_teams.add(match.team_a.lower())
                            live_teams.add(match.team_b.lower())

                # Check open positions — use Polymarket resolution as source of truth
                for p in self.executor.positions:
                    if p.resolved:
                        continue
                    still_live = any(
                        teams_match(p.team, lt) for lt in live_teams
                    )
                    age_minutes = (time.time() - p.timestamp) / 60

                    if not still_live and age_minutes > 15:
                        # Match no longer in live feed — check Polymarket for resolution
                        market = None
                        # Find the market for this position
                        for mid, mkt in self.latency_analyzer._match_to_market.items():
                            if mkt.market_id == p.market_id:
                                market = mkt
                                break

                        won = None  # None = unknown
                        if market:
                            try:
                                loop = asyncio.get_event_loop()
                                # Method 1: Check if officially closed
                                result = await loop.run_in_executor(
                                    None, self.market_finder.check_market_resolved, market
                                )
                                if result is not None:
                                    if result == "a":
                                        won = teams_match(p.team, market.team_a)
                                    else:
                                        won = teams_match(p.team, market.team_b)
                                    logger.info(f"[POLYMARKET] {market.question[:50]} resolved: team_{result} won")

                                # Method 2: If not officially closed, fetch fresh prices
                                # If price is >0.90 for one side, the outcome is de facto decided
                                if won is None:
                                    await loop.run_in_executor(
                                        None, self.market_finder.update_market_prices, market
                                    )
                                    # Use best_bid (what you can sell for) — more reliable than midpoint
                                    bid_a = market.best_bid_a if market.best_bid_a > 0 else market.price_a
                                    bid_b = market.best_bid_b if market.best_bid_b > 0 else market.price_b
                                    if bid_a > 0.90 or market.price_a > 0.95:
                                        won = teams_match(p.team, market.team_a)
                                        logger.info(f"[POLYMARKET] {market.question[:50]} price_a={market.price_a:.3f} bid_a={bid_a:.3f} — team_a de facto won")
                                    elif bid_b > 0.90 or market.price_b > 0.95:
                                        won = teams_match(p.team, market.team_b)
                                        logger.info(f"[POLYMARKET] {market.question[:50]} price_b={market.price_b:.3f} bid_b={bid_b:.3f} — team_b de facto won")
                            except Exception as e:
                                logger.debug(f"Polymarket resolution check failed: {e}")

                        if won is not None:
                            self.executor.resolve_position(p, won=won)
                            status = "WIN" if won else "LOSS"
                            logger.info(
                                f"[RESOLVED-{status}] {p.team} ({p.game}) ${p.amount:.2f} — "
                                f"confirmed by Polymarket ({age_minutes:.0f}m old)"
                            )
                        elif age_minutes > 120:
                            # 2+ hours and Polymarket hasn't resolved — keep checking, DON'T assume loss
                            # Only log once per hour to avoid spam
                            if int(age_minutes) % 60 < 3:
                                logger.warning(
                                    f"[WAITING] {p.team} ({p.game}) ${p.amount:.2f} — "
                                    f"awaiting Polymarket resolution ({age_minutes:.0f}m old)"
                                )
                        elif age_minutes > 1440:
                            # 24 hours — something is truly stuck, expire
                            self.executor.resolve_position(p, won=False)
                            logger.warning(
                                f"[EXPIRED] {p.team} ({p.game}) ${p.amount:.2f} — "
                                f"no resolution after 24h, marked as loss"
                        )

                # Persist if anything changed
                has_expired = any(p.resolved and p.pnl == -p.amount for p in self.executor.positions)
                if has_expired:
                    self.executor._persist()

            except Exception as e:
                logger.error(f"Position cleanup error: {e}")

            await asyncio.sleep(120)  # check every 2 minutes

    async def _sum_to_one_scanner(self):
        """Scan linked markets for sum-to-one arbitrage (buy both sides < $1.00)."""
        await asyncio.sleep(30)  # let markets link first
        while self._running:
            try:
                linked = dict(self.latency_analyzer._match_to_market)
                loop = asyncio.get_event_loop()
                for match_id, market in linked.items():
                    # Fetch both order books
                    book_a = await loop.run_in_executor(
                        None, self.market_finder.fetch_orderbook, market.token_id_a)
                    book_b = await loop.run_in_executor(
                        None, self.market_finder.fetch_orderbook, market.token_id_b)

                    if not book_a or not book_b:
                        continue

                    ask_sum = book_a.best_ask + book_b.best_ask
                    # Correct fee math: 2% applies to NET WINNINGS only
                    # If we buy both sides for ask_sum, winner pays out $1/share
                    # Gross profit = (1/min_ask - 1) * bet_per_side  (winning side profit)
                    # Fee = gross_profit * 0.02
                    # Net profit = (1.0 - ask_sum) * bet - fee
                    gap = 1.0 - ask_sum
                    # Approximate fee: winning side earns gap*bet, fee is 2% of that
                    fee_rate = 0.02
                    net_gap = gap * (1 - fee_rate)  # gap after fee on winnings

                    if net_gap > 0.005:  # at least 0.5% profit after fees
                        profit_pct = net_gap * 100
                        # Calculate bet size — split evenly on both sides
                        max_bet = min(50.0, self.executor.balance * 0.1)  # max 10% of balance
                        bet_per_side = max_bet / 2

                        logger.info(
                            f"[SUM-TO-ONE] ARB FOUND! {market.question[:50]} | "
                            f"Ask A={book_a.best_ask:.3f} + Ask B={book_b.best_ask:.3f} = {ask_sum:.3f} | "
                            f"Profit: {profit_pct:.1f}% after fees"
                        )

                        # Buy Team A side
                        pos_a = self.executor.execute_trade(
                            token_id=market.token_id_a, team=f"{market.team_a} [S2O]",
                            game=market.game, market_id=market.market_id,
                            amount=bet_per_side, confidence=0.99,
                            latency_edge_ms=0, max_price=book_a.best_ask + 0.01,
                            market_price=book_a.midpoint, our_price=1.0 - book_b.best_ask,
                        )
                        if pos_a:
                            pos_a.strategy = "sum_to_one"
                            self.executor.db.update_position(pos_a)

                        # Buy Team B side
                        pos_b = self.executor.execute_trade(
                            token_id=market.token_id_b, team=f"{market.team_b} [S2O]",
                            game=market.game, market_id=market.market_id,
                            amount=bet_per_side, confidence=0.99,
                            latency_edge_ms=0, max_price=book_b.best_ask + 0.01,
                            market_price=book_b.midpoint, our_price=1.0 - book_a.best_ask,
                        )
                        if pos_b:
                            pos_b.strategy = "sum_to_one"
                            self.executor.db.update_position(pos_b)

                        if pos_a and pos_b:
                            logger.info(
                                f"[SUM-TO-ONE] Executed! ${bet_per_side:.2f} each side | "
                                f"Guaranteed profit: ${max_bet * net_gap:.2f}"
                            )

                # Cross-market arb: moneyline + opposite handicap for same match
                all_markets = list(self.market_finder._market_cache.values())
                active_markets = [m for m in all_markets if 0.02 < m.price_a < 0.98]

                # Group by team pair
                from collections import defaultdict
                from team_match import normalize as _normalize
                xmarket_groups = defaultdict(list)
                for m in active_markets:
                    if m.team_a.lower() in ('over','under','yes','no','odd','even'):
                        continue
                    key = '_'.join(sorted([_normalize(m.team_a), _normalize(m.team_b)]))
                    xmarket_groups[key].append(m)

                for key, mkts in xmarket_groups.items():
                    if len(mkts) < 2:
                        continue

                    # Check all pairs for arb
                    for i in range(len(mkts)):
                        for j in range(i+1, len(mkts)):
                            m1, m2 = mkts[i], mkts[j]
                            # Try: buy m1.a + m2.b (different markets, correlated outcomes)
                            for tid1, tid2, desc in [
                                (m1.token_id_a, m2.token_id_b, f"{m1.team_a}({m1.question[:20]}) + {m2.team_b}({m2.question[:20]})"),
                                (m1.token_id_b, m2.token_id_a, f"{m1.team_b}({m1.question[:20]}) + {m2.team_a}({m2.question[:20]})"),
                            ]:
                                b1 = await loop.run_in_executor(None, self.market_finder.fetch_orderbook, tid1)
                                b2 = await loop.run_in_executor(None, self.market_finder.fetch_orderbook, tid2)
                                if not b1 or not b2:
                                    continue
                                cost = b1.best_ask + b2.best_ask
                                if cost < 0.96:  # 4%+ profit after fees
                                    profit_pct = (1.0 - cost) * 100
                                    logger.info(
                                        f"[CROSS-ARB] {desc} | cost={cost:.3f} | profit={profit_pct:.1f}%"
                                    )
                                    # Execute both sides
                                    bet_per = min(25.0, self.executor.balance * 0.05)
                                    pos1 = self.executor.execute_trade(
                                        token_id=tid1, team=f"{m1.team_a} [XARB]",
                                        game=m1.game, market_id=m1.market_id,
                                        amount=bet_per, confidence=0.99,
                                        latency_edge_ms=0, max_price=b1.best_ask + 0.01,
                                        market_price=b1.midpoint, our_price=1.0 - b2.best_ask,
                                    )
                                    pos2 = self.executor.execute_trade(
                                        token_id=tid2, team=f"{m2.team_b} [XARB]",
                                        game=m2.game, market_id=m2.market_id,
                                        amount=bet_per, confidence=0.99,
                                        latency_edge_ms=0, max_price=b2.best_ask + 0.01,
                                        market_price=b2.midpoint, our_price=1.0 - b1.best_ask,
                                    )
                                    if pos1:
                                        pos1.strategy = "cross_arb"
                                        self.executor.db.update_position(pos1)
                                    if pos2:
                                        pos2.strategy = "cross_arb"
                                        self.executor.db.update_position(pos2)
                                    if pos1 and pos2:
                                        logger.info(f"[CROSS-ARB] Executed! ${bet_per:.2f} each side | guaranteed {profit_pct:.1f}%")

            except Exception as e:
                logger.error(f"Arb scanner error: {e}")

            await asyncio.sleep(10)

    async def _position_monitor(self):
        """Auto-exit positions based on take-profit, stop-loss, or time expiry."""
        logger.info(f"[MONITOR] Position monitor started (TP={config.TAKE_PROFIT_PCT*100:.0f}% / SL={config.STOP_LOSS_PCT*100:.0f}% / timeout={config.EXIT_TIMEOUT_SECONDS:.0f}s)")
        try:
            await asyncio.sleep(1)
            logger.info(f"[MONITOR] Sleep done, entering loop (_running={self._running})")
        except Exception as e:
            logger.error(f"[MONITOR] Sleep error: {e}")
            return

        _mon_iter = 0
        while self._running:
            _mon_iter += 1
            try:
                open_positions = [p for p in self.executor.positions if not p.resolved]
                if _mon_iter % 10 == 1:
                    logger.info(f"[MONITOR] iter={_mon_iter} open={len(open_positions)}")
                if not open_positions:
                    await asyncio.sleep(config.POSITION_MONITOR_INTERVAL)
                    continue

                loop = asyncio.get_event_loop()
                for p in open_positions:
                    try:
                        # Get current market price for this position
                        linked = self.latency_analyzer._match_to_market
                        current_price = p.fill_price  # fallback
                        for mid, mkt in linked.items():
                            if str(mkt.market_id) == str(p.market_id):
                                current_price = mkt.price_a
                                break
                            # Also match by token_id
                            if p.token_id and (p.token_id == mkt.token_id_a or p.token_id == mkt.token_id_b):
                                # Use correct side price
                                current_price = mkt.price_a if p.token_id == mkt.token_id_a else mkt.price_b
                                break

                        if current_price <= 0.01:
                            age = time.time() - p.timestamp
                            if age > config.EXIT_TIMEOUT_SECONDS * 2:
                                self.executor.sell_position(p, p.fill_price, "time_exit")
                                self._update_state()
                            continue

                        reason = self.executor.check_exit_conditions(p, current_price)
                        if reason:
                            self.executor.sell_position(p, current_price, reason)
                            # Log to event feed
                            pnl_pct = (current_price - p.fill_price) / p.fill_price * 100
                            with self._state_lock:
                                self.state.recent_events.append({
                                    "game": p.game,
                                    "type": f"exit_{reason}",
                                    "team": p.team,
                                    "description": f"[{reason.upper()}] Sold {p.team} @{current_price:.3f} (bought @{p.fill_price:.3f}) P&L: {pnl_pct:+.1f}%",
                                    "impact": pnl_pct / 100,
                                    "time": time.time(),
                                    "has_market": True,
                                    "has_bet": True,
                                    "faster": pnl_pct > 0,
                                })
                            self._update_state()

                    except Exception as e:
                        logger.error(f"[MONITOR] Error checking {p.team}: {e}")

            except Exception as e:
                logger.error(f"[MONITOR] Loop error: {e}")

            await asyncio.sleep(config.POSITION_MONITOR_INTERVAL)

    async def _pending_order_monitor(self):
        """Check pending limit bids for fills or expiry."""
        logger.info("[PENDING] Pending order monitor started")
        await asyncio.sleep(5)

        while self._running:
            try:
                pending = [o for o in self.executor.pending_orders if o.status == "pending"]
                if not pending:
                    await asyncio.sleep(3)
                    continue

                loop = asyncio.get_event_loop()
                for order in pending:
                    try:
                        # Fetch current orderbook to get midpoint
                        ob = await loop.run_in_executor(
                            None, self.market_finder.fetch_orderbook, order.token_id
                        )
                        if not ob:
                            continue

                        filled = self.executor.check_limit_fill(
                            order, ob.best_ask, ob.best_bid)

                        if filled:
                            # Log to event feed
                            with self._state_lock:
                                self.state.recent_events.append({
                                    "game": order.game,
                                    "type": "limit_filled",
                                    "team": order.team,
                                    "description": f"LIMIT FILLED: {order.team} @${order.bid_price:.3f} (model ${order.model_price:.3f})",
                                    "impact": 0.1,
                                    "time": time.time(),
                                    "has_market": True,
                                    "has_bet": True,
                                    "faster": True,
                                })
                            self._update_state()

                        elif order.status == "expired":
                            with self._state_lock:
                                self.state.recent_events.append({
                                    "game": order.game,
                                    "type": "limit_expired",
                                    "team": order.team,
                                    "description": f"LIMIT EXPIRED: {order.team} bid@${order.bid_price:.3f} (no fill in {config.LIMIT_ORDER_TIMEOUT:.0f}s)",
                                    "impact": 0,
                                    "time": time.time(),
                                    "has_market": True,
                                    "has_bet": False,
                                    "faster": None,
                                })
                            self._update_state()

                    except Exception as e:
                        logger.error(f"[PENDING] Error checking {order.team}: {e}")

            except Exception as e:
                logger.error(f"[PENDING] Loop error: {e}")

            await asyncio.sleep(3)

    async def _scan_mode_loop(self):
        """Every 15s, pick top matches by |model−market| gap and synthesize a
        SCORE_UPDATE event through process_event so qwen gets a chance to vote.
        Drop-don't-queue: if qwen is busy, the should_buy call returns None and
        nothing happens. Per-match 90s cooldown prevents scan spam."""
        from feeds.base import GameEvent, EventType
        await asyncio.sleep(30)  # let feeds populate
        logger.info("[SCAN] Scan-mode sweep task started (15s cadence)")
        last_scan: dict[str, float] = {}
        while self._running:
            try:
                linked = dict(self.latency_analyzer._match_to_market)
                candidates = []
                for mid, mkt in linked.items():
                    state = None
                    for feed in self.feeds.values():
                        st = getattr(feed, "_matches", {}).get(mid)
                        if st and getattr(st, "is_live", False):
                            state = st
                            break
                    if not state:
                        continue
                    our_a = getattr(state, "win_probability_a", 0.5) or 0.5
                    mkt_a = getattr(mkt, "price_a", 0.5) or 0.5
                    gap = abs(our_a - mkt_a)
                    if gap < 0.02:
                        continue
                    if time.time() - last_scan.get(mid, 0) < 90:
                        continue
                    candidates.append((gap, mid, mkt, state, our_a))
                candidates.sort(reverse=True, key=lambda x: x[0])
                for gap, mid, mkt, state, our_a in candidates[:3]:
                    # Favor the side our model prefers
                    favored = "a" if our_a >= 0.5 else "b"
                    evt = GameEvent(
                        event_type=EventType.SCORE_UPDATE,
                        match_id=mid,
                        game=mkt.game,
                        team=favored,
                        description=f"[SCAN] gap={gap*100:.1f}% model_a={our_a:.2f} mkt_a={getattr(mkt,'price_a',0) or 0:.2f}",
                        impact=(our_a - 0.5) * 0.3,
                        match_state=state,
                    )
                    last_scan[mid] = time.time()
                    try:
                        logger.info(f"[SCAN] tick {mid[:16]} favored={favored} gap={gap*100:.1f}%")
                        self.latency_analyzer.process_event(evt)
                    except Exception as e:
                        logger.debug(f"[SCAN] process_event error: {e}")
            except Exception as e:
                logger.error(f"[SCAN] loop error: {e}")
            await asyncio.sleep(15)

    async def _main_async(self):
        """Main async entrypoint — start feeds, market scanner, and match discovery."""
        # Save main loop so thread-side callers (monitor) can schedule coroutines here
        self._async_loop = asyncio.get_event_loop()
        # Start market scanner
        scanner_task = asyncio.create_task(self._market_scanner())
        # Start qwen scan-mode sweep
        scan_task = asyncio.create_task(self._scan_mode_loop())

        # Start Polymarket WebSocket for real-time prices
        await self.polymarket_ws.connect()
        asyncio.create_task(self._ws_subscription_loop())

        # Start CS2 match discovery loop
        discovery_task = asyncio.create_task(self._match_discovery_loop())

        # Start background price updater for linked markets
        price_task = asyncio.create_task(self._price_updater())

        # Start team intelligence fetcher
        # intel_task disabled — edge bot doesn't use team intel

        # Start position cleanup (expire finished matches)
        cleanup_task = asyncio.create_task(self._position_cleanup_loop())

        # Start sum-to-one arbitrage scanner
        # arb_task disabled — edge bot only uses event-driven trading
        # arb_task = asyncio.create_task(self._sum_to_one_scanner())

        # Run price updater and position monitor in separate threads
        # (feeds block the asyncio event loop with synchronous HTTP calls)
        import threading

        def _run_price_updater():
            import time as _time
            _time.sleep(5)
            logger.info("[PRICE-THREAD] Price updater thread started (1s cadence)")
            _update_count = 0
            while self._running:
                t0 = _time.time()
                try:
                    linked = dict(self.latency_analyzer._match_to_market)
                    if linked:
                        updated = 0
                        for match_id, market in linked.items():
                            try:
                                self.market_finder.update_market_prices(market)
                                updated += 1
                            except Exception:
                                pass
                        _update_count += 1
                        if _update_count % 30 == 1:  # log once per 30s instead of per 30 cycles
                            logger.info(f"[PRICES] Updated {updated}/{len(linked)} markets")
                except Exception as e:
                    logger.error(f"[PRICES] Error: {e}")
                # Target 1s loop — adjust sleep so total iteration is ≈1s, never 0
                elapsed = _time.time() - t0
                _time.sleep(max(0.1, 1.0 - elapsed))

        threading.Thread(target=_run_price_updater, daemon=True).start()

        # REST fallback poller — WS is primary. Only polls a token if its WS
        # data is stale >10s (connection lost or missed updates). Also exposed
        # as self._rest_refresh_token() for on-demand pre-trade refresh.
        def _refresh_token(tid: str) -> bool:
            try:
                book = self.market_finder.fetch_orderbook(tid)
                if not book:
                    return False
                if book.best_bid <= 0.01 and book.best_ask >= 0.99:
                    return False
                cur = self.polymarket_ws.prices.get(tid) or {
                    "best_bid": 0, "best_ask": 0, "last_trade": 0, "timestamp": 0
                }
                cur["best_bid"] = book.best_bid
                cur["best_ask"] = book.best_ask
                cur["timestamp"] = time.time()
                self.polymarket_ws.prices[tid] = cur
                return True
            except Exception:
                return False
        self._rest_refresh_token = _refresh_token

        def _run_rest_poller():
            import time as _time
            _time.sleep(5)
            logger.info("[REST-POLL] WS-gap fallback poller started (polls on >10s WS silence)")
            while self._running:
                t0 = _time.time()
                try:
                    tokens: set = set()
                    for p in self.executor.positions:
                        if not p.resolved and p.token_id:
                            tokens.add(p.token_id)
                    for mkt in self.latency_analyzer._match_to_market.values():
                        for attr in ("token_id_a", "token_id_b"):
                            tid = getattr(mkt, attr, None)
                            if tid:
                                tokens.add(tid)
                    now = _time.time()
                    polled = 0
                    for tid in tokens:
                        cur = self.polymarket_ws.prices.get(tid)
                        age = (now - cur["timestamp"]) if (cur and cur.get("timestamp")) else 999
                        if age > 10:
                            if _refresh_token(tid):
                                polled += 1
                    if polled:
                        logger.info(f"[REST-POLL] WS silent >10s — refreshed {polled} tokens via REST")
                except Exception as e:
                    logger.debug(f"[REST-POLL] error: {e}")
                elapsed = _time.time() - t0
                _time.sleep(max(0.5, 2.0 - elapsed))

        threading.Thread(target=_run_rest_poller, daemon=True).start()

        # Monitor heartbeat + watchdog — if position monitor hangs (past bug: DB lock
        # in sell_position wedges the whole thread, leaving positions stuck 20+ min
        # past their stop-loss), this watchdog kills the process so the wrapper restarts.
        self._monitor_heartbeat = time.time()
        def _monitor_watchdog():
            import time as _w
            import os as _os
            import signal as _signal
            _w.sleep(60)  # grace period for startup
            while self._running:
                age = _w.time() - self._monitor_heartbeat
                if age > 120:  # no monitor tick in 2 minutes = stalled
                    logger.error(f"[WATCHDOG] Monitor hung for {age:.0f}s — killing process for wrapper restart")
                    _os.kill(_os.getpid(), _signal.SIGKILL)
                    return
                _w.sleep(15)
        threading.Thread(target=_monitor_watchdog, daemon=True).start()

        def _run_monitor():
            import time as _time
            logger.info("[MONITOR-THREAD] Position monitor thread started")
            _time.sleep(10)  # let feeds start
            while self._running:
                self._monitor_heartbeat = _time.time()  # feed watchdog
                try:
                    open_positions = [p for p in self.executor.positions if not p.resolved]
                    if not open_positions:
                        _time.sleep(config.POSITION_MONITOR_INTERVAL)
                        continue

                    for p in open_positions:
                        try:
                            # Get real-time price — prefer last trade (realistic), fall back to bid.
                            # STALE-PRICE PROTECTION: if we have no fresh WS data, SKIP this tick
                            # entirely rather than SL off a stale/default price. (Past bug: during
                            # restart, WS wasn't reconnected → fell back to price_a=0.50 → auto-SL'd
                            # 4 winning Vitality trades at fill 0.65 → forced-sell at 0.50.)
                            ws_data = self.polymarket_ws.get_price(p.token_id)
                            has_fresh_ws = bool(ws_data and (ws_data.get("last_trade", 0) > 0.01 or ws_data.get("best_bid", 0) > 0.01))
                            if has_fresh_ws:
                                if ws_data.get("last_trade", 0) > 0.01:
                                    current_price = ws_data["last_trade"]
                                else:
                                    current_price = ws_data["best_bid"]
                            else:
                                # No fresh price — skip exit checks for this position this tick.
                                logger.debug(f"[MONITOR] skip {p.team} — no fresh WS price")
                                continue

                            if current_price <= 0.01:
                                # Price collapsed — force exit
                                self.executor.sell_position(p, 0.01, "stop_loss")
                                logger.info(f"[MONITOR] FORCE EXIT: {p.team} — price collapsed to {current_price:.3f}")
                                continue

                            reason = self.executor.check_exit_conditions(p, current_price)

                            # AI-CUT pre-check: if position is at >=50% of SL distance and qwen idle,
                            # ask if thesis is broken. Drop-don't-queue; throttled 30s per position.
                            if (not reason) and self.edge_analyst is not None and getattr(self, "_async_loop", None):
                                try:
                                    sl_pct = p.effective_sl_pct or config.STOP_LOSS_PCT
                                    loss_frac = (p.fill_price - current_price) / max(p.fill_price, 0.01)
                                    if sl_pct > 0 and loss_frac >= 0.5 * sl_pct:
                                        if not hasattr(self, "_exit_check_last"):
                                            self._exit_check_last = {}
                                        last = self._exit_check_last.get(p.token_id, 0)
                                        if _time.time() - last >= 30:
                                            self._exit_check_last[p.token_id] = _time.time()
                                            # Build context
                                            pos_info = {
                                                "team": p.team,
                                                "fill_price": p.fill_price,
                                                "pnl_pct": -loss_frac * 100,
                                                "age_s": _time.time() - p.timestamp,
                                                "sl_target_pct": sl_pct,
                                            }
                                            ws_snap = self.polymarket_ws.get_price(p.token_id) or {}
                                            mkt_state = {
                                                "bid": ws_snap.get("best_bid", 0) or 0,
                                                "ask": ws_snap.get("best_ask", 0) or 0,
                                                "spread": (ws_snap.get("best_ask", 0) or 0) - (ws_snap.get("best_bid", 0) or 0),
                                            }
                                            gs = {"game": p.game, "team_a": p.team, "team_b": "opponent"}
                                            # Find match state for richer context
                                            for mid, mkt in self.latency_analyzer._match_to_market.items():
                                                if str(mkt.market_id) == str(p.market_id):
                                                    for feed in self.feeds.values():
                                                        st = getattr(feed, "_matches", {}).get(mid)
                                                        if st:
                                                            gs.update({
                                                                "team_a": getattr(st, "team_a", p.team),
                                                                "team_b": getattr(st, "team_b", "opponent"),
                                                                "score_a": getattr(st, "score_a", 0),
                                                                "score_b": getattr(st, "score_b", 0),
                                                                "round_a": getattr(st, "round_a", 0),
                                                                "round_b": getattr(st, "round_b", 0),
                                                                "economy_a": getattr(st, "economy_a", 0),
                                                                "economy_b": getattr(st, "economy_b", 0),
                                                                "round_kills_a": getattr(st, "round_kills_a", 0),
                                                                "round_kills_b": getattr(st, "round_kills_b", 0),
                                                                "side": getattr(st, "side", "?"),
                                                            })
                                                        break
                                                    break
                                            recent = list(self.state.recent_events)[-5:]
                                            fut = asyncio.run_coroutine_threadsafe(
                                                self.edge_analyst.should_exit(pos_info, gs, mkt_state, recent),
                                                self._async_loop,
                                            )
                                            try:
                                                result = fut.result(timeout=5.0)
                                            except Exception:
                                                result = None
                                            if result and result.get("action") == "cut" and result.get("confidence", 0) > 0.6:
                                                reason = "ai_cut"
                                                logger.info(f"[AI-CUT] {p.team} @{current_price:.3f} (fill {p.fill_price:.3f}, {-loss_frac*100:+.1f}%) conf={result.get('confidence'):.2f} — {result.get('reason','')}")
                                except Exception as _e:
                                    logger.debug(f"[AI-CUT] error: {_e}")

                            if reason:
                                # STAMP COOLDOWN *BEFORE* selling — prevents race where a
                                # concurrent qwen call opens a new position between sell and stamp.
                                # (Real bug: SIM-769 SL at 22:18:03 → SIM-770 opened 22:19:18 →
                                # SL-STAMP at 22:19:19, 1s too late.)
                                if reason in ("stop_loss", "ai_cut"):
                                    _now = _time.time()
                                    for mid, mkt in self.latency_analyzer._match_to_market.items():
                                        if str(mkt.market_id) == str(p.market_id):
                                            self.latency_analyzer._match_decisions[f"{mid}_{p.token_id}_stopped"] = {"stopped": True, "stopped_at": _now}
                                    self.latency_analyzer._match_decisions[f"mkt_{p.market_id}_{p.token_id}_stopped"] = {"stopped": True, "stopped_at": _now}
                                    if not hasattr(self.latency_analyzer, "_sl_token_cooldowns"):
                                        self.latency_analyzer._sl_token_cooldowns = {}
                                    self.latency_analyzer._sl_token_cooldowns[p.token_id] = _now
                                    logger.info(f"[SL-STAMP] token={p.token_id[:12]} market={p.market_id} cooled 10min (pre-sell, reason={reason})")
                                self.executor.sell_position(p, current_price, reason)
                                pnl_pct = (current_price - p.fill_price) / p.fill_price * 100
                                logger.info(f"[MONITOR] {reason.upper()}: {p.team} | buy@{p.fill_price:.3f} sell@{current_price:.3f} | {pnl_pct:+.1f}%")
                                if reason == "stop_loss":
                                    self._log_loss_analysis(p, current_price, pnl_pct)
                                with self._state_lock:
                                    self.state.recent_events.append({
                                        "game": p.game, "type": f"exit_{reason}", "team": p.team,
                                        "description": f"[{reason.upper()}] {p.team} @{current_price:.3f} (bought @{p.fill_price:.3f}) {pnl_pct:+.1f}%",
                                        "impact": pnl_pct / 100, "time": _time.time(),
                                        "has_market": True, "has_bet": True, "faster": pnl_pct > 0,
                                    })
                        except Exception as e:
                            logger.error(f"[MONITOR] Error: {e}")
                except Exception as e:
                    logger.error(f"[MONITOR] Loop error: {e}")
                # Update dashboard state every tick for real-time display
                try:
                    self._update_state()
                except Exception:
                    pass
                _time.sleep(config.POSITION_MONITOR_INTERVAL)

        threading.Thread(target=_run_monitor, daemon=True).start()

        # Continuous snapshot recorder — capture price + state every 5s for training data
        def _run_snapshot_recorder():
            import time as _time
            _time.sleep(30)  # let feeds + markets link first
            logger.info("[SNAPSHOT-RECORDER] Started — capturing every 5s for all linked matches")
            while self._running:
                try:
                    linked = dict(self.latency_analyzer._match_to_market)
                    for match_id, market in linked.items():
                        # Find match state
                        state = None
                        for feed in self.feeds.values():
                            if match_id in feed._matches:
                                state = feed._matches[match_id]
                                break
                        if not state or not state.is_live:
                            continue
                        # Get WS prices for BOTH tokens (training pipeline
                        # needs both sides' orderbook to compute the buy-side
                        # hindsight against the opposite side's pricing).
                        ws_a = self.polymarket_ws.get_price(market.token_id_a) if self.polymarket_ws else None
                        ws_b = self.polymarket_ws.get_price(market.token_id_b) if (self.polymarket_ws and market.token_id_b) else None
                        if not ws_a or ws_a.get("best_bid", 0) <= 0:
                            continue
                        # Record price snapshot — BOTH tokens
                        self.recorder.record_best_bid_ask(
                            match_id, market.token_id_a,
                            ws_a.get("best_bid", 0), ws_a.get("best_ask", 0),
                            state.team_a, state.team_b,
                        )
                        if ws_b and ws_b.get("best_bid", 0) > 0:
                            self.recorder.record_best_bid_ask(
                                match_id, market.token_id_b,
                                ws_b.get("best_bid", 0), ws_b.get("best_ask", 0),
                                state.team_a, state.team_b,
                            )
                        # (thin game-state snapshot DISABLED — the bo3.gg raw-payload
                        # handler _on_raw_bo3_snapshot writes rich SNAPSHOT_MATCH_UPDATE
                        # lines on every provider tick. Writing a thin one here too
                        # would just poison the training corpus with 8-field records.)
                except Exception as e:
                    pass  # don't crash the thread
                _time.sleep(5)

        threading.Thread(target=_run_snapshot_recorder, daemon=True).start()

        # Start all feeds (don't await — they run forever)
        feeds_task = asyncio.create_task(self._run_feeds())

        # Match analyzer disabled — was burning ~15s/call doing dashboard decoration only.
        logger.info("[ANALYZER] DISABLED — freeing GPU/VRAM for edge_analyst trade decisions")

        # ─── CONTINUOUS TICKER: re-evaluate every active match every 20s ─────
        # Even when no game event fires, qwen sees the current price/momentum and can
        # decide to enter (or stay out). This keeps qwen analyzing constantly.
        async def _continuous_ticker():
            await asyncio.sleep(15)  # wait for feeds to subscribe
            from feeds.base import GameEvent, EventType
            import time as _time
            logger.info("[TICKER] Continuous match ticker started (60s cadence — re-evaluates every linked match)")
            while self._running:
                try:
                    # Snapshot of currently linked CS2 matches with real (non-synthetic) IDs
                    linked = list(self.latency_analyzer._match_to_market.items())
                    fired = 0
                    for match_id, market in linked:
                        if market.game != "cs2":
                            continue
                        # Find the live MatchState for this match across all feeds
                        ms = None
                        for feed in self.feeds.values():
                            if match_id in feed._matches and feed._matches[match_id].is_live:
                                ms = feed._matches[match_id]
                                break
                        if not ms:
                            continue
                        # Skip if we already have an open position on this market
                        has_open = any(not p.resolved and p.market_id == market.market_id
                                       for p in self.executor.positions)
                        if has_open:
                            continue
                        # Fire a TICKER event — looks like a SCORE_UPDATE so the analyzer
                        # processes it through the normal qwen path with full context.
                        evt = GameEvent(
                            event_type=EventType.SCORE_UPDATE,
                            match_id=match_id, game="cs2",
                            team="a" if ms.win_probability_a > 0.5 else "b",
                            description=f"[TICKER] periodic re-evaluation @ {_time.strftime('%H:%M:%S')}",
                            impact=0.0, match_state=ms,
                        )
                        try:
                            self.latency_analyzer.process_event(evt)
                            fired += 1
                        except Exception as e:
                            pass  # individual match failure shouldn't kill the ticker
                    if fired:
                        logger.debug(f"[TICKER] Re-evaluated {fired} matches")
                except Exception as e:
                    logger.error(f"[TICKER] Error: {e}")
                await asyncio.sleep(60)

        # Ticker disabled — it was flooding qwen's serial queue causing 60s+ latencies.
        # Real game events (firehose mode) already trigger qwen on every round/kill/score change.
        # ticker_task = asyncio.create_task(_continuous_ticker())

        # Keep running
        while self._running:
            await asyncio.sleep(1)

    def _log_loss_analysis(self, position, exit_price: float, pnl_pct: float):
        """Auto-categorize and persist every stop loss for learning."""
        import time as _time
        age = _time.time() - position.timestamp
        spread_at_entry = abs(position.fill_price - position.market_price_at_entry)

        # Categorize the loss
        if exit_price < 0.05 and position.fill_price > 0.20:
            category = "resolved_token"
            details = f"Token crashed to {exit_price:.3f} — likely resolved/wrong map"
        elif age < 20 and pnl_pct < -10:
            category = "spread_loss"
            details = f"Instant loss in {age:.0f}s — bid-ask spread wider than expected"
        elif pnl_pct < -40:
            category = "crash"
            details = f"Massive {pnl_pct:.0f}% drop — market resolved or flash crash"
        elif position.fill_price > 0.80:
            category = "high_price_entry"
            details = f"Bought at {position.fill_price:.2f}¢ — limited upside, high risk"
        elif position.our_price_at_entry < position.market_price_at_entry - 0.10:
            category = "market_ahead"
            details = f"Market was ahead: our={position.our_price_at_entry:.2f} mkt={position.market_price_at_entry:.2f}"
        elif position.game == "dota2":
            category = "dota2_swing"
            details = f"Dota2 game swung against us despite initial advantage"
        else:
            category = "market_move"
            details = f"Market moved against thesis: {pnl_pct:+.1f}% in {age:.0f}s"

        analysis = {
            "timestamp": _time.time(),
            "order_id": position.order_id,
            "team": position.team,
            "game": position.game,
            "entry_price": position.fill_price,
            "exit_price": exit_price,
            "pnl": position.pnl,
            "pnl_pct": pnl_pct,
            "age_seconds": age,
            "spread_at_entry": spread_at_entry,
            "category": category,
            "details": details,
            "market_type": getattr(position, 'strategy', ''),
            "market_volume": 0,
            "our_price": position.our_price_at_entry,
            "market_price": position.market_price_at_entry,
        }

        self.executor.db.save_loss_analysis(analysis)
        logger.info(f"[LOSS-ANALYSIS] {category}: {position.team} ({position.game}) | {details}")

    def stop(self):
        """Stop the bot."""
        self._running = False
        self.state.status = "stopped"
        self.recorder.close()
        self.executor.db.flush_events()
        self.executor.db.close()
        logger.info("Esports bot stopped")
