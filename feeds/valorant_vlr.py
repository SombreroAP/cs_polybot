"""
Valorant live match feed via VLR.gg scraping.

VLR.gg's hosted API (vlrggapi.vercel.app) is paywalled.
We scrape VLR.gg's matches page directly for live scores.

Provides map-level scores in a series. Round-level data
requires scraping individual match pages (slower).

Coverage: VCT Americas, EMEA, Pacific, China, Challengers.
"""
import asyncio
import logging
import re
import time
from bs4 import BeautifulSoup
from typing import Optional

import aiohttp

from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

VLR_MATCHES_URL = "https://www.vlr.gg/matches"


class ValorantVLRFeed(GameFeed):
    """Valorant live match data via VLR.gg scraping."""

    POLL_INTERVAL = 10  # seconds for match list
    ROUND_POLL_INTERVAL = 8  # seconds for live round data

    def __init__(self):
        super().__init__("valorant")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        self._round_poll_task = None
        self._subscribed_matches: set[str] = set()
        self._initialized: set[str] = set()
        self._last_round_scores: dict[str, dict] = {}  # match_id -> {map_id: (score_a, score_b)}

    async def connect(self):
        self._session = aiohttp.ClientSession(headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept": "text/html",
        })
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="valorant", team="neutral",
            description="Connected to Valorant VLR.gg feed", impact=0.0,
        ))
        logger.info("Valorant VLR.gg feed connected")
        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "val_poll"))
        self._round_poll_task = asyncio.create_task(self._safe_run(self._round_poll_loop(), "val_rounds"))

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
        if self._session:
            await self._session.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._subscribed_matches.add(match_id)
        if match_id not in self._matches:
            self._matches[match_id] = MatchState(
                match_id=match_id, game="valorant",
                team_a="", team_b="",
                is_live=True, started_at=time.time(),
                total_maps=3, extra={},
            )

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)
        self._matches.pop(match_id, None)

    async def get_live_matches(self) -> list[MatchState]:
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self) -> list[str]:
        return list(self._subscribed_matches)

    def estimate_win_probability(self, state: MatchState) -> float:
        maps_needed = (state.total_maps // 2) + 1
        if state.score_a >= maps_needed:
            return 0.99
        if state.score_b >= maps_needed:
            return 0.01
        a_needs = maps_needed - state.score_a
        b_needs = maps_needed - state.score_b
        raw = max(0.01, min(0.99, b_needs / (a_needs + b_needs)))
        prior = state.extra.get("prior_prob_a")
        if prior is not None:
            from prior import apply_prior
            return apply_prior(raw, prior)
        return raw

    async def _poll_loop(self):
        """Poll VLR.gg for live Valorant matches."""
        logger.info("Valorant poll loop started")
        while self._running:
            try:
                matches = await self._fetch_live_matches()
                if matches is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                live_ids = set()
                for m in matches:
                    mid = m["id"]
                    live_ids.add(mid)

                    if mid not in self._subscribed_matches:
                        await self.subscribe_match(mid)
                        state = self._matches[mid]
                        state.team_a = m["team1"]
                        state.team_b = m["team2"]
                        state.score_a = m["score1"]
                        state.score_b = m["score2"]
                        state.total_maps = 3  # default Bo3
                        state.extra["match_url"] = m.get("url", "")
                        state.extra["stream_url"] = m.get("url", "")  # VLR page as fallback

                        logger.info(f"Subscribed: {m['team1']} vs {m['team2']} (valorant) [{mid}]")
                        self._emit(GameEvent(
                            event_type=EventType.MATCH_START,
                            match_id=mid, game="valorant", team="neutral",
                            description=f"{m['team1']} vs {m['team2']} LIVE",
                            impact=0.0, match_state=state,
                        ))
                        self._initialized.add(mid)
                        continue

                    # Check score changes
                    state = self._matches.get(mid)
                    if not state:
                        continue

                    old_a = state.score_a
                    old_b = state.score_b
                    new_a = m["score1"]
                    new_b = m["score2"]

                    if new_a != old_a or new_b != old_b:
                        winner = "a" if new_a > old_a else "b"
                        state.score_a = new_a
                        state.score_b = new_b

                        # Scrape match page for agent picks + player stats
                        match_url = state.extra.get("match_url", "")
                        if match_url:
                            try:
                                map_stats = await self._scrape_match_stats(match_url)
                                if map_stats:
                                    state.extra["map_stats"] = map_stats
                                    logger.info(f"[VAL STATS] Got {len(map_stats)} maps of stats for {state.team_a} vs {state.team_b}")
                            except Exception as e:
                                logger.debug(f"Match stats scrape failed: {e}")

                        old_prob = state.win_probability_a
                        state.win_probability_a = self.estimate_win_probability(state)
                        impact = state.win_probability_a - old_prob

                        winner_name = state.team_a if winner == "a" else state.team_b
                        logger.info(f"[MAP] {winner_name} wins map! {state.team_a} {new_a}-{new_b} {state.team_b}")

                        self._emit(GameEvent(
                            event_type=EventType.MAP_WIN,
                            match_id=mid, game="valorant", team=winner,
                            description=f"Map to {winner_name} ({new_a}-{new_b})",
                            impact=impact, match_state=state,
                        ))

                        maps_needed = (state.total_maps // 2) + 1
                        if new_a >= maps_needed or new_b >= maps_needed:
                            state.is_live = False
                            self._emit(GameEvent(
                                event_type=EventType.MATCH_END,
                                match_id=mid, game="valorant", team=winner,
                                description=f"Match won by {winner_name} ({new_a}-{new_b})",
                                impact=1.0 if winner == "a" else -1.0,
                                match_state=state,
                            ))

                    state.last_event_time = time.time()

                for mid in list(self._subscribed_matches):
                    if mid not in live_ids and mid in self._matches:
                        self._matches[mid].is_live = False

                live_count = sum(1 for m in self._matches.values() if m.is_live)
                logger.info(f"Valorant: {len(matches)} live, {live_count} tracked")

                await asyncio.sleep(self.POLL_INTERVAL)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Valorant poll error: {e}")
                await asyncio.sleep(self.POLL_INTERVAL)

    async def _round_poll_loop(self):
        """Poll individual match pages for live round scores."""
        logger.info("Valorant round poll loop started")
        await asyncio.sleep(30)  # let matches discover first
        while self._running:
            try:
                for mid in list(self._subscribed_matches):
                    state = self._matches.get(mid)
                    if not state or not state.is_live:
                        continue

                    match_url = state.extra.get("match_url", "")
                    if not match_url:
                        continue

                    try:
                        async with self._session.get(
                            match_url, timeout=aiohttp.ClientTimeout(total=10),
                        ) as resp:
                            if resp.status != 200:
                                continue
                            html = await resp.text()

                        self._process_round_data(mid, html)

                    except Exception as e:
                        logger.debug(f"Round poll failed for {mid}: {e}")

                    await asyncio.sleep(2)  # stagger requests

            except Exception as e:
                logger.error(f"Valorant round poll error: {e}")

            await asyncio.sleep(self.ROUND_POLL_INTERVAL)

    def _process_round_data(self, match_id: str, html: str):
        """Extract live round scores from VLR.gg match page."""
        state = self._matches.get(match_id)
        if not state:
            return

        soup = BeautifulSoup(html, "html.parser")

        for game_div in soup.find_all(class_="vm-stats-game"):
            game_id = game_div.get("data-game-id", "")
            if not game_id or game_id == "all":
                continue

            scores = game_div.find_all(class_="score")
            if len(scores) < 2:
                continue

            s1 = scores[0].get_text(strip=True)
            s2 = scores[1].get_text(strip=True)
            if not s1.isdigit() or not s2.isdigit():
                continue

            new_a, new_b = int(s1), int(s2)

            # Check if round score changed
            prev = self._last_round_scores.get(match_id, {})
            prev_a, prev_b = prev.get(game_id, (0, 0))

            if (new_a, new_b) != (prev_a, prev_b) and (new_a + new_b) > 0:
                self._last_round_scores.setdefault(match_id, {})[game_id] = (new_a, new_b)

                # Store round score in state
                state.round_score_a = new_a
                state.round_score_b = new_b
                state.extra["round_score"] = f"{new_a}-{new_b}"

                if prev_a + prev_b == 0:
                    # First round data for this map
                    logger.info(f"[VAL ROUND] {state.team_a} vs {state.team_b} | Map rounds: {new_a}-{new_b}")
                    return

                # Round changed — emit event
                winning_team = "a" if new_a > prev_a else "b"
                winner_name = state.team_a if winning_team == "a" else state.team_b

                old_prob = state.win_probability_a
                state.win_probability_a = self.estimate_win_probability(state)
                impact = state.win_probability_a - old_prob

                logger.info(f"[VAL ROUND] {state.team_a} {new_a}-{new_b} {state.team_b}")

                self._emit(GameEvent(
                    event_type=EventType.ROUND_END,
                    match_id=match_id, game="valorant", team=winning_team,
                    description=f"Round to {winner_name} ({new_a}-{new_b})",
                    impact=impact, match_state=state,
                ))

    async def _fetch_live_matches(self) -> Optional[list]:
        """Scrape VLR.gg matches page for live match data."""
        try:
            async with self._session.get(
                VLR_MATCHES_URL,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    logger.warning(f"VLR.gg returned {resp.status}")
                    return None
                html = await resp.text()

            return self._parse_live_matches(html)

        except Exception as e:
            logger.error(f"VLR.gg fetch failed: {e}")
            return None

    def _parse_live_matches(self, html: str) -> list:
        """Parse live matches from VLR.gg HTML using BeautifulSoup."""
        matches = []
        try:
            soup = BeautifulSoup(html, "html.parser")

            # VLR.gg groups matches under "wf-label" headers.
            # The "LIVE" section has class "mod-live" on the label.
            # Walk the DOM: after a mod-live label, collect match items
            # until the next label (end of live section).
            in_live_section = False
            live_ids: set[str] = set()

            for elem in soup.find_all(["div", "a"]):
                classes = elem.get("class") or []

                # Detect section label
                if "wf-label" in classes:
                    in_live_section = "mod-live" in classes
                    continue

                # Collect live match cards
                if in_live_section and "match-item" in classes and elem.name == "a":
                    self._parse_match_card(elem, matches, live_ids)

            # Fallback: if no live section found, grab match cards with numeric scores
            # (live matches show digits; upcoming show "–")
            if not matches:
                for a in soup.find_all("a", class_="match-item"):
                    self._parse_match_card(a, matches, live_ids)
                # Filter: only keep matches with numeric scores (live indicator)
                matches = [m for m in matches if m["score1"] >= 0 and m["score2"] >= 0
                           and (m["score1"] > 0 or m["score2"] > 0)]

        except Exception as e:
            logger.error(f"VLR.gg parse error: {e}")

        return matches

    def _parse_match_card(self, tag, matches: list, seen_ids: set) -> None:
        """Extract match data from a single VLR.gg match card element."""
        href = tag.get("href", "")
        id_match = re.match(r"/(\d+)/", href)
        if not id_match:
            return
        mid = id_match.group(1)
        if mid in seen_ids:
            return

        team_name_tags = tag.find_all(class_="match-item-vs-team-name")
        score_tags = tag.find_all(class_="match-item-vs-team-score")

        if len(team_name_tags) < 2 or len(score_tags) < 2:
            return

        t1 = team_name_tags[0].get_text(strip=True)
        t2 = team_name_tags[1].get_text(strip=True)
        s1_text = score_tags[0].get_text(strip=True)
        s2_text = score_tags[1].get_text(strip=True)

        if not t1 or not t2:
            return
        if not s1_text.isdigit() or not s2_text.isdigit():
            return  # "–" = not started yet

        seen_ids.add(mid)
        matches.append({
            "id": mid,
            "url": f"https://www.vlr.gg{href}",
            "team1": t1,
            "team2": t2,
            "score1": int(s1_text),
            "score2": int(s2_text),
        })

    async def _scrape_match_stats(self, url: str) -> list:
        """Scrape VLR.gg match page for per-map agent picks and player stats."""
        try:
            async with self._session.get(
                url, timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status != 200:
                    return []
                html = await resp.text()

            soup = BeautifulSoup(html, "html.parser")
            maps_data = []

            for game_div in soup.find_all(class_="vm-stats-game"):
                map_info = {"map": "", "score": "", "team1_players": [], "team2_players": []}

                # Map name
                map_el = game_div.find(class_="map")
                if map_el:
                    # Extract just the map name (first text node)
                    map_name = ""
                    for child in map_el.children:
                        if isinstance(child, str):
                            cleaned = child.strip()
                            if cleaned and cleaned not in ("PICK", "-"):
                                map_name = cleaned
                                break
                    map_info["map"] = map_name

                # Map score
                scores = game_div.find_all(class_="score")
                if len(scores) >= 2:
                    s1 = scores[0].get_text(strip=True)
                    s2 = scores[1].get_text(strip=True)
                    if s1.isdigit() and s2.isdigit():
                        map_info["score"] = f"{s1}-{s2}"

                # Player stats — 2 tables (one per team)
                tables = game_div.find_all("tbody")
                for team_idx, tbody in enumerate(tables[:2]):
                    players = []
                    for tr in tbody.find_all("tr"):
                        tds = tr.find_all("td")
                        if len(tds) < 6:
                            continue

                        name = tds[0].get_text(strip=True)
                        # Clean up name (remove team tag)
                        name = re.sub(r'[A-Z]{2,5}$', '', name).strip()

                        agent_img = tds[1].find("img")
                        agent = agent_img.get("title", "?") if agent_img else "?"

                        # Parse K/D/A from combined text
                        stats_text = tds[3].get_text(strip=True)
                        kda_match = re.match(r"(\d+)\D+(\d+)\D+(\d+)", stats_text)
                        if kda_match:
                            k, d, a = kda_match.groups()
                        else:
                            k, d, a = "0", "0", "0"

                        acs_text = tds[2].get_text(strip=True)
                        acs = re.search(r"[\d.]+", acs_text)
                        acs = acs.group() if acs else "0"

                        players.append({
                            "name": name, "agent": agent,
                            "kills": int(k), "deaths": int(d), "assists": int(a),
                            "acs": float(acs),
                        })

                    key = "team1_players" if team_idx == 0 else "team2_players"
                    map_info[key] = players

                if map_info["score"]:  # only add maps with data
                    maps_data.append(map_info)

            return maps_data

        except Exception as e:
            logger.debug(f"Match stats scrape error: {e}")
            return []
