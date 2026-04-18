"""
Reference odds from external bookmakers via the-odds-api.com.

Provides Pinnacle, Betway, DraftKings etc. odds as a sanity check
against Polymarket prices. If Polymarket deviates significantly from
bookmaker consensus, either we have an edge or Polymarket knows something.

Free tier: 500 requests/month (~16/day). One call per live match is enough.
Sign up at https://the-odds-api.com for a free API key.
"""
import time
import logging
from typing import Optional

import aiohttp

import config

logger = logging.getLogger(__name__)

# the-odds-api esports sport keys
SPORT_KEYS = {
    "cs2": "esports_csgo",      # CS2/CSGO
    "dota2": "esports_dota2",
    "lol": "esports_lol",
    "valorant": "esports_valorant",
}


class OddsReference:
    """Fetches reference odds from traditional bookmakers."""

    CACHE_TTL = 300  # 5 minutes — odds don't change that fast

    def __init__(self):
        self._api_key = getattr(config, "ODDS_API_KEY", "")
        self._session: Optional[aiohttp.ClientSession] = None
        self._cache: dict[str, tuple[float, dict]] = {}  # sport_key -> (timestamp, odds_data)
        self._enabled = bool(self._api_key)
        self._requests_used = 0

        if self._enabled:
            logger.info("Reference odds enabled (the-odds-api)")
        else:
            logger.info("Reference odds disabled — set ODDS_API_KEY in .env for bookmaker odds")

    async def _ensure_session(self):
        if not self._session or self._session.closed:
            self._session = aiohttp.ClientSession()

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def get_odds(self, game: str, team_a: str, team_b: str) -> Optional[dict]:
        """Get bookmaker odds for a matchup. Returns dict with bookmaker prices or None."""
        if not self._enabled:
            return None

        sport_key = SPORT_KEYS.get(game)
        if not sport_key:
            return None

        # Check cache
        cache_key = f"{sport_key}"
        if cache_key in self._cache:
            ts, data = self._cache[cache_key]
            if time.time() - ts < self.CACHE_TTL:
                return self._find_match(data, team_a, team_b)

        # Fetch fresh odds
        await self._ensure_session()
        try:
            async with self._session.get(
                f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
                params={
                    "apiKey": self._api_key,
                    "regions": "eu,us",
                    "markets": "h2h",
                    "oddsFormat": "decimal",
                },
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                self._requests_used += 1
                if resp.status != 200:
                    logger.warning(f"Odds API returned {resp.status}")
                    return None

                events = await resp.json()
                self._cache[cache_key] = (time.time(), events)

                remaining = resp.headers.get("x-requests-remaining", "?")
                logger.info(f"[ODDS] Fetched {len(events)} {game} events (remaining: {remaining})")

                return self._find_match(events, team_a, team_b)

        except Exception as e:
            logger.error(f"Odds API error: {e}")
            return None

    def _find_match(self, events: list, team_a: str, team_b: str) -> Optional[dict]:
        """Find a specific match in the odds data and extract consensus."""
        from team_match import teams_match

        for event in events:
            home = event.get("home_team", "")
            away = event.get("away_team", "")

            if (teams_match(team_a, home) and teams_match(team_b, away)) or \
               (teams_match(team_a, away) and teams_match(team_b, home)):

                # Extract odds from each bookmaker
                bookmaker_odds = {}
                for bm in event.get("bookmakers", []):
                    name = bm.get("title", "?")
                    for market in bm.get("markets", []):
                        if market.get("key") == "h2h":
                            outcomes = market.get("outcomes", [])
                            for o in outcomes:
                                if teams_match(o.get("name", ""), team_a):
                                    bookmaker_odds[name] = {
                                        "prob_a": round(1.0 / o.get("price", 2.0), 3),
                                        "decimal_a": o.get("price", 2.0),
                                    }

                if not bookmaker_odds:
                    return None

                # Calculate consensus
                probs = [v["prob_a"] for v in bookmaker_odds.values()]
                consensus = sum(probs) / len(probs)

                # Find Pinnacle specifically (sharpest book)
                pinnacle = bookmaker_odds.get("Pinnacle", {}).get("prob_a")

                result = {
                    "consensus_prob_a": round(consensus, 3),
                    "pinnacle_prob_a": pinnacle,
                    "num_bookmakers": len(bookmaker_odds),
                    "bookmakers": bookmaker_odds,
                }

                logger.info(
                    f"[ODDS] {team_a} vs {team_b}: consensus={consensus:.1%}, "
                    f"pinnacle={pinnacle:.1%} ({len(bookmaker_odds)} books)"
                    if pinnacle else
                    f"[ODDS] {team_a} vs {team_b}: consensus={consensus:.1%} ({len(bookmaker_odds)} books)"
                )
                return result

        return None

    def format_for_prompt(self, odds: dict) -> str:
        """Format odds data for Claude prompt."""
        if not odds:
            return "No external bookmaker odds available."

        lines = [f"BOOKMAKER CONSENSUS: {odds['consensus_prob_a']:.1%} for team A ({odds['num_bookmakers']} bookmakers)"]
        if odds.get("pinnacle_prob_a"):
            lines.append(f"PINNACLE (sharpest book): {odds['pinnacle_prob_a']:.1%}")

        # Show top 3 bookmakers
        for name, data in list(odds.get("bookmakers", {}).items())[:3]:
            lines.append(f"  {name}: {data['prob_a']:.1%} (decimal {data['decimal_a']:.2f})")

        return "\n".join(lines)
