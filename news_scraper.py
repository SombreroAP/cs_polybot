"""
Esports news & social media scraper for the Insider Intel expert.

Fetches recent team/player news from free sources:
- HLTV (CS2): news headlines, roster changes
- Reddit (all games): r/GlobalOffensive, r/DotA2, r/leagueoflegends, r/ValorantCompetitive
- Liquipedia (all games): roster pages for stand-in detection

All data is cached to avoid repeated scraping. Rate-limited to be respectful.
"""
import asyncio
import time
import logging
import re
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import requests as sync_requests

logger = logging.getLogger(__name__)

# Reddit JSON API (no auth needed, just add .json to any page)
REDDIT_SUBS = {
    "cs2": "GlobalOffensive",
    "dota2": "DotA2",
    "lol": "leagueoflegends",
    "valorant": "ValorantCompetitive",
}

HLTV_NEWS_URL = "https://www.hltv.org/news/archive"


class NewsScraper:
    """Fetches recent esports news for team matchups."""

    CACHE_TTL = 600  # 10 minutes — news doesn't change that fast

    def __init__(self):
        self._session = sync_requests.Session()
        self._session.headers.update({
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html,application/json",
        })
        self._cache: dict[str, tuple[float, list[str]]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)

    async def close(self):
        self._executor.shutdown(wait=False)

    async def get_match_news(self, game: str, team_a: str, team_b: str) -> list[str]:
        """Get recent news headlines relevant to a matchup. Returns list of headline strings."""
        cache_key = f"{game}:{team_a.lower()}:{team_b.lower()}"
        if cache_key in self._cache:
            ts, headlines = self._cache[cache_key]
            if time.time() - ts < self.CACHE_TTL:
                return headlines

        headlines = []

        # Run fetches in thread executor to avoid blocking asyncio
        loop = asyncio.get_event_loop()

        try:
            reddit_news = await loop.run_in_executor(
                self._executor, self._fetch_reddit_sync, game, team_a, team_b)
            headlines.extend(reddit_news)
        except Exception as e:
            logger.debug(f"Reddit fetch failed for {team_a} vs {team_b}: {e}")

        if game == "cs2":
            try:
                hltv_news = await loop.run_in_executor(
                    self._executor, self._fetch_hltv_news_sync, team_a, team_b)
                headlines.extend(hltv_news)
            except Exception as e:
                logger.debug(f"HLTV news fetch failed: {e}")

        # Deduplicate and limit
        seen = set()
        unique = []
        for h in headlines:
            h_lower = h.lower().strip()
            if h_lower not in seen and len(h) > 10:
                seen.add(h_lower)
                unique.append(h)

        unique = unique[:10]  # max 10 headlines
        self._cache[cache_key] = (time.time(), unique)

        if unique:
            logger.info(f"[NEWS] {game} {team_a} vs {team_b}: {len(unique)} headlines found")

        return unique

    def _fetch_reddit_sync(self, game: str, team_a: str, team_b: str) -> list[str]:
        """Search Reddit for recent posts about these teams (sync, runs in executor)."""
        sub = REDDIT_SUBS.get(game)
        if not sub:
            return []

        headlines = []
        searches = [f"{team_a} vs {team_b}", team_a, team_b]
        for query in searches:
            url = f"https://www.reddit.com/r/{sub}/search.json"
            params = {
                "q": query,
                "sort": "relevance",
                "t": "week",
                "restrict_sr": "on",
                "limit": 5,
            }
            try:
                resp = self._session.get(url, params=params, timeout=8)
                if resp.status_code != 200:
                    continue
                data = resp.json()
                posts = data.get("data", {}).get("children", [])
                for post in posts:
                    d = post.get("data", {})
                    title = d.get("title", "")
                    score = d.get("score", 0)
                    if score >= 2 and title:
                        headlines.append(f"[Reddit r/{sub}, {score}pts] {title}")
                import time as _time
                _time.sleep(1)
            except Exception:
                continue

        return headlines

    def _fetch_hltv_news_sync(self, team_a: str, team_b: str) -> list[str]:
        """Fetch recent HLTV news mentioning these teams (sync, runs in executor)."""
        headlines = []
        try:
            resp = self._session.get("https://www.hltv.org/rss/news", timeout=8)
            if resp.status_code != 200:
                return []
            xml = resp.text

            titles = re.findall(r'<title><!\[CDATA\[(.*?)\]\]></title>', xml)
            if not titles:
                titles = re.findall(r'<title>(.*?)</title>', xml)

            team_a_lower = team_a.lower()
            team_b_lower = team_b.lower()
            for title in titles:
                t_lower = title.lower()
                if team_a_lower in t_lower or team_b_lower in t_lower:
                    headlines.append(f"[HLTV News] {title}")
        except Exception:
            pass

        return headlines
