"""
Polymarket market discovery and monitoring for esports events.
Finds active esports markets, tracks odds in real-time, and detects staleness.
"""
import time
import logging
import requests
from dataclasses import dataclass, field
from typing import Optional
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

import config

logger = logging.getLogger(__name__)


@dataclass
class EsportsMarket:
    """Represents a single esports betting market on Polymarket."""
    market_id: str
    condition_id: str
    question: str
    game: str  # cs2, lol, dota2, valorant
    team_a: str
    team_b: str
    token_id_a: str  # token for team A win
    token_id_b: str  # token for team B win
    price_a: float = 0.5
    price_b: float = 0.5
    best_bid_a: float = 0.0
    best_ask_a: float = 0.0
    best_bid_b: float = 0.0
    best_ask_b: float = 0.0
    volume: float = 0.0
    liquidity: float = 0.0
    active: bool = True
    last_price_update: float = 0.0  # timestamp of last price change
    event_slug: str = ""


@dataclass
class OrderBookSnapshot:
    """Snapshot of order book state for latency measurement."""
    token_id: str
    best_bid: float
    best_ask: float
    bid_depth: float  # total $ on bid side (top 5 levels)
    ask_depth: float
    spread: float
    midpoint: float
    timestamp: float = field(default_factory=time.time)


class MarketFinder:
    """Discovers and monitors esports markets on Polymarket."""

    GAME_KEYWORDS = {
        "cs2": ["counter-strike", "cs2", "csgo", "cs:go"],
        "lol": ["league-of-legends", "lol", "league of legends"],
        "dota2": ["dota-2", "dota2", "dota 2", "the-international"],
        "valorant": ["valorant", "vct"],
    }

    # Polymarket sports tag IDs (from /sports endpoint)
    # These are used with tag_id param on /events endpoint
    GAME_TAG_IDS = {
        "cs2": "100780",     # Counter-Strike 2
        "dota2": "102366",   # Dota 2
        "lol": "65",         # League of Legends
        "valorant": "101672", # Valorant
    }

    def __init__(self):
        self._session = self._create_session()
        self._market_cache: dict[str, EsportsMarket] = {}
        self._cache_expiry: float = 0.0

    def _create_session(self) -> requests.Session:
        session = requests.Session()
        retry = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=4)
        session.mount("https://", adapter)
        return session

    def fetch_esports_markets(self, game_filter: Optional[str] = None) -> list[EsportsMarket]:
        """Fetch all active esports markets from Gamma API."""
        now = time.time()
        if now < self._cache_expiry and self._market_cache:
            markets = list(self._market_cache.values())
            if game_filter:
                markets = [m for m in markets if m.game == game_filter]
            return markets

        markets = []
        try:
            # Use Polymarket sports tag IDs to fetch esports markets
            games_to_fetch = [game_filter] if game_filter else list(config.ENABLED_GAMES)

            for game in games_to_fetch:
                game = game.strip()
                tag_id = self.GAME_TAG_IDS.get(game)
                if not tag_id:
                    continue

                params = {
                    "tag_id": tag_id,
                    "active": "true",
                    "closed": "false",
                    "limit": 500,
                }

                resp = self._session.get(
                    f"{config.GAMMA_API_URL}/events",
                    params=params,
                    timeout=10,
                )
                if resp.status_code != 200:
                    logger.warning(f"Gamma API returned {resp.status_code} for {game} (tag_id={tag_id})")
                    continue

                events = resp.json()
                if not isinstance(events, list):
                    continue

                for event in events:
                    parsed = self._parse_all_markets(event, game_hint=game)
                    markets.extend(parsed)

            # Deduplicate by market_id
            seen = set()
            unique = []
            for m in markets:
                if m.market_id not in seen:
                    seen.add(m.market_id)
                    unique.append(m)
                    self._market_cache[m.market_id] = m
            markets = unique

            self._cache_expiry = now + 30  # cache for 30 seconds
            logger.info(f"Found {len(markets)} active esports markets")

        except Exception as e:
            logger.error(f"Failed to fetch esports markets: {e}")

        if game_filter:
            markets = [m for m in markets if m.game == game_filter]

        return markets

    def check_market_resolved(self, market: EsportsMarket) -> Optional[str]:
        """Check if a market has resolved on Polymarket.
        Returns 'a' if team_a won, 'b' if team_b won, None if still open."""
        try:
            resp = self._session.get(
                f"{config.GAMMA_API_URL}/markets/{market.condition_id}",
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            closed = data.get("closed", False)
            if not closed:
                return None

            prices = data.get("outcomePrices", "")
            if isinstance(prices, str):
                import json as _json
                prices = _json.loads(prices)

            if not prices or len(prices) < 2:
                return None

            # outcomePrices ["1","0"] = token A won, ["0","1"] = token B won
            if str(prices[0]) == "1":
                return "a"
            elif str(prices[1]) == "1":
                return "b"

            return None
        except Exception as e:
            logger.debug(f"Market resolution check failed for {market.market_id}: {e}")
            return None

    def _parse_all_markets(self, event: dict, game_hint: Optional[str] = None) -> list[EsportsMarket]:
        """Parse ALL sub-markets from an event (series winner, map winners, etc.)."""
        results = []
        sub_markets = event.get("markets", [])
        slug = event.get("slug", "")
        event_volume = float(event.get("volume", 0) or 0)  # total volume across all sub-markets
        title = event.get("title", "")

        for market in sub_markets:
            try:
                question = market.get("question", "")
                if not market.get("active") or market.get("closed"):
                    continue

                game = game_hint or self._detect_game(question, slug, event.get("tags", []))
                if not game:
                    continue

                outcomes = market.get("outcomes", [])
                if isinstance(outcomes, str):
                    import json as _json
                    outcomes = _json.loads(outcomes)

                clob_token_ids = market.get("clobTokenIds", [])
                if isinstance(clob_token_ids, str):
                    import json as _json
                    clob_token_ids = _json.loads(clob_token_ids)

                outcome_prices = market.get("outcomePrices", [])
                if isinstance(outcome_prices, str):
                    import json as _json
                    outcome_prices = _json.loads(outcome_prices)

                if len(outcomes) < 2 or len(clob_token_ids) < 2:
                    continue

                try:
                    price_a = float(outcome_prices[0]) if outcome_prices else 0.5
                    price_b = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
                except (ValueError, IndexError):
                    price_a = price_b = 0.5

                team_a = outcomes[0]
                team_b = outcomes[1]

                # Skip non-match markets (Yes/No, Over/Under, Odd/Even)
                skip_outcomes = {"yes", "no", "over", "under", "odd", "even"}
                if team_a.lower() in skip_outcomes or team_b.lower() in skip_outcomes:
                    continue

                # Use the specific market's volume, NOT the event total. The event total
                # lumps every sub-market together (series + all map winners + props + handicaps)
                # which massively overstates liquidity of any single market we'd actually trade.
                market_vol = float(market.get("volume", 0) or 0)
                market_liq = float(market.get("liquidity", 0) or 0)
                results.append(EsportsMarket(
                    market_id=str(market.get("id", "")),
                    condition_id=market.get("conditionId", ""),
                    question=question or title,
                    game=game,
                    team_a=team_a,
                    team_b=team_b,
                    token_id_a=clob_token_ids[0],
                    token_id_b=clob_token_ids[1],
                    price_a=price_a,
                    price_b=price_b,
                    volume=market_vol,
                    liquidity=market_liq,
                    active=True,
                    event_slug=slug,
                ))
            except Exception:
                continue

        return results

    def _parse_event(self, event: dict, game_hint: Optional[str] = None) -> Optional[EsportsMarket]:
        """Parse a Gamma API event into an EsportsMarket."""
        try:
            markets = event.get("markets", [])
            if not markets:
                return None

            market = markets[0]  # primary market (match winner)
            question = market.get("question", "")
            slug = event.get("slug", "")
            title = event.get("title", "")

            # Only match-winner markets (must have "vs" in title)
            if " vs " not in title and " vs " not in question:
                return None

            # Use game hint from tag-based fetch, or detect from text
            game = game_hint or self._detect_game(question, slug, event.get("tags", []))
            if not game:
                return None

            # Extract outcomes — may be a JSON string or list
            outcomes = market.get("outcomes", [])
            if isinstance(outcomes, str):
                import json as _json
                outcomes = _json.loads(outcomes)

            # Extract token IDs — Gamma API returns these as JSON strings
            clob_token_ids = market.get("clobTokenIds", [])
            if isinstance(clob_token_ids, str):
                import json as _json
                clob_token_ids = _json.loads(clob_token_ids)

            # Extract prices — also may be JSON strings
            outcome_prices = market.get("outcomePrices", [])
            if isinstance(outcome_prices, str):
                import json as _json
                outcome_prices = _json.loads(outcome_prices)

            if len(outcomes) < 2 or len(clob_token_ids) < 2:
                return None

            # Parse prices safely
            try:
                price_a = float(outcome_prices[0]) if outcome_prices else 0.5
                price_b = float(outcome_prices[1]) if len(outcome_prices) > 1 else 0.5
            except (ValueError, IndexError):
                price_a = price_b = 0.5

            # Extract team names from outcomes or title
            team_a = outcomes[0]
            team_b = outcomes[1]

            return EsportsMarket(
                market_id=str(market.get("id", "")),
                condition_id=market.get("conditionId", ""),
                question=question or title,
                game=game,
                team_a=team_a,
                team_b=team_b,
                token_id_a=clob_token_ids[0],
                token_id_b=clob_token_ids[1],
                price_a=price_a,
                price_b=price_b,
                volume=float(market.get("volume", 0) or 0),
                liquidity=float(market.get("liquidity", 0) or 0),
                active=market.get("active", True),
                event_slug=slug,
            )
        except Exception as e:
            logger.debug(f"Failed to parse event: {e}")
            return None
            return None

    def _detect_game(self, question: str, slug: str, tags: list) -> Optional[str]:
        """Detect which game an event belongs to."""
        text = f"{question} {slug} {' '.join(str(t) for t in tags)}".lower()
        for game, keywords in self.GAME_KEYWORDS.items():
            if game in config.ENABLED_GAMES:
                for kw in keywords:
                    if kw in text:
                        return game
        return None

    def fetch_orderbook(self, token_id: str) -> Optional[OrderBookSnapshot]:
        """Fetch current order book for a token from CLOB API."""
        try:
            resp = self._session.get(
                f"{config.CLOB_API_URL}/book",
                params={"token_id": token_id},
                timeout=10,
            )
            if resp.status_code != 200:
                return None

            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])

            if not bids and not asks:
                return None  # empty book = no market, don't trade

            # Polymarket CLOB returns bids sorted ASCENDING and asks sorted DESCENDING,
            # so the BEST prices are at the END of each array (bids[-1] = highest bid,
            # asks[-1] = lowest ask). Reading [0] gave the WORST price on each side →
            # every PRE-TRADE-BLOCK thought the book was crashed when it wasn't.
            # (Bug: Vitality vs Falcons book reported bid=0.04 ask=0.96 when real
            # book was bid=0.33 ask=0.34 with $47k depth.)
            real_bids = [b for b in bids if float(b["price"]) > 0.03]
            real_asks = [a for a in asks if float(a["price"]) < 0.97]

            best_bid = float(real_bids[-1]["price"]) if real_bids else (float(bids[-1]["price"]) if bids else 0.0)
            best_ask = float(real_asks[-1]["price"]) if real_asks else (float(asks[-1]["price"]) if asks else 1.0)

            # Sum top 5 levels of real depth — the END of each array is the best
            # side, so take the last 5 entries (or all, if fewer).
            depth_bids = real_bids[-5:] if real_bids else bids[-5:]
            depth_asks = real_asks[-5:] if real_asks else asks[-5:]
            bid_depth = sum(float(b.get("size", 0)) * float(b.get("price", 0)) for b in depth_bids)
            ask_depth = sum(float(a.get("size", 0)) * float(a.get("price", 0)) for a in depth_asks)

            spread = best_ask - best_bid if best_ask > best_bid else 0.0
            midpoint = (best_bid + best_ask) / 2

            return OrderBookSnapshot(
                token_id=token_id,
                best_bid=best_bid,
                best_ask=best_ask,
                bid_depth=bid_depth,
                ask_depth=ask_depth,
                spread=spread,
                midpoint=midpoint,
            )
        except Exception as e:
            logger.error(f"Failed to fetch orderbook for {token_id}: {e}")
            return None

    def update_market_prices(self, market: EsportsMarket) -> EsportsMarket:
        """Update a market's prices from the order book."""
        book_a = self.fetch_orderbook(market.token_id_a)
        book_b = self.fetch_orderbook(market.token_id_b)

        if book_a:
            old_mid = (market.best_bid_a + market.best_ask_a) / 2 if market.best_bid_a else 0
            new_mid = book_a.midpoint

            market.best_bid_a = book_a.best_bid
            market.best_ask_a = book_a.best_ask
            market.price_a = book_a.midpoint
            market.liquidity = book_a.bid_depth + book_a.ask_depth

            # Track when price actually changes (for staleness detection)
            if abs(new_mid - old_mid) > 0.001:
                market.last_price_update = time.time()

        if book_b:
            market.best_bid_b = book_b.best_bid
            market.best_ask_b = book_b.best_ask
            market.price_b = book_b.midpoint

        return market
