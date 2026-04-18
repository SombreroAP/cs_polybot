"""
Bo3.gg WebSocket Feed — real-time CS2 match data.

Connects to wss://updates.bo3.gg/ws and receives instant updates:
- Round scores, economy, equipment values
- Side (CT/T), map name, round phase
- Series score, game number
- Per-player stats when available

This replaces the 1-second polling with zero-delay push data.
"""
import asyncio
import json
import logging
import time
from typing import Optional

import aiohttp

from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

WS_URL = "wss://updates.bo3.gg/ws"
SUBSCRIBE_MSG = {"type": "subscribe", "payload": {"topics": [{"key": "/matches"}]}}


class CS2Bo3WebSocketFeed(GameFeed):
    """Real-time CS2 data via Bo3.gg WebSocket — zero polling delay."""

    def __init__(self):
        super().__init__("cs2")
        self._session: Optional[aiohttp.ClientSession] = None
        self._ws = None
        self._last_states: dict[int, dict] = {}  # match_id -> last known state
        self._team_names_cache: dict[int, tuple] = {}  # match_id -> (team1_name, team2_name)

    async def connect(self):
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="cs2", team="neutral",
            description="Bo3.gg WebSocket feed connected", impact=0.0,
        ))

        self._session = aiohttp.ClientSession()
        asyncio.create_task(self._ws_loop())
        logger.info("Bo3.gg WebSocket feed starting")

    async def _ws_loop(self):
        """Maintain persistent WebSocket connection with auto-reconnect."""
        reconnect_delay = 1

        while self._running:
            try:
                async with self._session.ws_connect(WS_URL, headers={
                    'User-Agent': 'Mozilla/5.0',
                    'Origin': 'https://bo3.gg',
                }, heartbeat=30) as ws:
                    self._ws = ws
                    reconnect_delay = 1
                    logger.info("[BO3-WS] Connected to Bo3.gg WebSocket")

                    # Subscribe to matches channel
                    await ws.send_json(SUBSCRIBE_MSG)

                    async for msg in ws:
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self._process_message(json.loads(msg.data))
                            except Exception as e:
                                logger.debug(f"[BO3-WS] Parse error: {e}")
                        elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSED):
                            break

            except Exception as e:
                logger.warning(f"[BO3-WS] Connection error: {e}")

            self._ws = None
            if self._running:
                logger.info(f"[BO3-WS] Reconnecting in {reconnect_delay}s...")
                await asyncio.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 2, 30)

    def _process_message(self, data: dict):
        """Process a WebSocket message from Bo3.gg."""
        if data.get("key") == "system":
            return  # pong, subscribed, etc

        if data.get("key") != "/matches" or data.get("action") != "updated":
            return

        payload = data.get("payload", {})
        if not payload:
            return

        match_id = payload.get("id")
        if not match_id:
            return

        live = payload.get("live_updates", {})
        if not live:
            return

        mid = str(match_id)

        # Emit the FULL bo3.gg payload for recording. This preserves player
        # states, HP, round_phase, is_bomb_planted, sides, equipment, and all
        # the other rich fields the backtester's trigger engine needs. The
        # synthesized MatchState we build downstream only holds scores + money.
        try:
            self._emit_raw_snapshot(mid, payload)
        except Exception:
            pass

        # Extract team names from full payload (sent occasionally)
        # WS team1/team2 names are AUTHORITATIVE — they correspond to WS team_1/team_2 scores
        if "team1" in payload and isinstance(payload["team1"], dict):
            t1_name = payload["team1"].get("name", "").strip()
            t2_name = payload.get("team2", {}).get("name", "").strip() if isinstance(payload.get("team2"), dict) else ""
            if t1_name and mid in self._matches:
                old_a = self._matches[mid].team_a
                if old_a and old_a != t1_name and old_a == t2_name:
                    # Names were swapped from polling feed — fix them
                    logger.warning(f"[BO3-WS] TEAM SWAP detected for {mid}: {old_a} was team_a but WS says team1={t1_name}")
                self._matches[mid].team_a = t1_name
            if t2_name and mid in self._matches:
                self._matches[mid].team_b = t2_name
            # Cache for new match creation
            self._team_names_cache[match_id] = (t1_name, t2_name)

        team1 = live.get("team_1", {})
        team2 = live.get("team_2", {})

        new_score_a = team1.get("game_score", 0)
        new_score_b = team2.get("game_score", 0)
        series_a = team1.get("match_score", 0)
        series_b = team2.get("match_score", 0)
        round_phase = live.get("round_phase", "")
        round_number = live.get("round_number", 0)
        game_number = live.get("game_number", 1)
        map_name = live.get("map_name", "")
        game_ended = live.get("game_ended", False)
        equip_a = team1.get("equipment_value", 0)
        equip_b = team2.get("equipment_value", 0)
        side_a = team1.get("side", "")

        # Get or create match state
        if mid not in self._matches:
            # Get team names from cache, payload, or REST API
            cached_names = self._team_names_cache.get(match_id)
            if cached_names:
                team_a_name, team_b_name = cached_names
            elif "team1" in payload and isinstance(payload["team1"], dict):
                team_a_name = payload["team1"].get("name", "").strip()
                team_b_name = payload.get("team2", {}).get("name", "").strip() if isinstance(payload.get("team2"), dict) else ""
            else:
                # Fetch from REST API
                team_a_name, team_b_name = self._fetch_team_names(match_id)
                self._team_names_cache[match_id] = (team_a_name, team_b_name)

            state = MatchState(
                match_id=mid, game="cs2",
                team_a=team_a_name, team_b=team_b_name,
                is_live=True, started_at=time.time(),
                total_maps=3,
                extra={},
            )
            self._matches[mid] = state
            self._subscribed_matches.add(mid)
            logger.info(f"[BO3-WS] New match: {team_a_name} vs {team_b_name} (id={match_id})")
            # Tell polling feed to suppress round events for this match (WS is authoritative)
            if hasattr(self, '_polling_feed') and self._polling_feed:
                self._polling_feed._ws_tracked_matches.add(mid)

        state = self._matches[mid]
        prev = self._last_states.get(match_id, {})

        # Retry team name resolution if still placeholder (throttle to once per 15s)
        if state.team_a.startswith("Team1_") or state.team_b.startswith("Team2_"):
            _resolve_key = f"_resolve_{match_id}"
            last_try = self._team_names_cache.get(_resolve_key, 0)
            if not isinstance(last_try, (int, float)):
                last_try = 0
            import time as _time
            if _time.time() - last_try > 15:
                self._team_names_cache[_resolve_key] = _time.time()
                t1, t2 = self._fetch_team_names(match_id)
                if t1 and not t1.startswith("Team1_"):
                    state.team_a = t1
                    state.team_b = t2
                    self._team_names_cache[match_id] = (t1, t2)
                    logger.info(f"[BO3-WS] Resolved: {t1} vs {t2} (id={match_id})")

        # Update state
        state.round_score_a = new_score_a
        state.round_score_b = new_score_b
        state.score_a = series_a
        state.score_b = series_b
        state.current_map = game_number
        state.extra["team_a_current_side"] = side_a.lower() if side_a else "?"
        state.extra["team_a_money"] = equip_a
        state.extra["team_b_money"] = equip_b
        state.extra["map_name"] = map_name
        state.extra["round_phase"] = round_phase

        # Calculate win probability
        state.win_probability_a = self.estimate_win_probability(state)

        # Detect events by comparing to previous state
        prev_score_a = prev.get("score_a", 0)
        prev_score_b = prev.get("score_b", 0)
        prev_series_a = prev.get("series_a", 0)
        prev_series_b = prev.get("series_b", 0)
        prev_phase = prev.get("round_phase", "")

        # Round end: score changed
        if (new_score_a != prev_score_a or new_score_b != prev_score_b) and (new_score_a + new_score_b > 0):
            if new_score_a > prev_score_a:
                winner = "a"
                winner_name = state.team_a
            else:
                winner = "b"
                winner_name = state.team_b

            impact = state.win_probability_a - prev.get("prob_a", 0.5)
            state.last_event_time = time.time()

            self._emit(GameEvent(
                event_type=EventType.ROUND_END,
                match_id=mid, game="cs2", team=winner,
                description=f"Round to {winner_name} ({new_score_a}-{new_score_b}) on {map_name}",
                impact=impact, match_state=state,
            ))
            logger.info(f"[BO3-WS] ROUND: {state.team_a} {new_score_a}-{new_score_b} {state.team_b} | {map_name}")

        # Map end
        if game_ended and not prev.get("game_ended", False):
            winner = "a" if new_score_a > new_score_b else "b"
            winner_name = state.team_a if winner == "a" else state.team_b

            self._emit(GameEvent(
                event_type=EventType.MAP_WIN,
                match_id=mid, game="cs2", team=winner,
                description=f"Map {game_number} to {winner_name} (Series: {series_a}-{series_b})",
                impact=1.0 if winner == "a" else -1.0, match_state=state,
            ))
            logger.info(f"[BO3-WS] MAP WIN: {winner_name} | Series {series_a}-{series_b}")

            # Check series end
            maps_needed = (state.total_maps // 2) + 1
            if series_a >= maps_needed or series_b >= maps_needed:
                state.is_live = False
                self._emit(GameEvent(
                    event_type=EventType.MATCH_END,
                    match_id=mid, game="cs2", team=winner,
                    description=f"Match won by {winner_name} ({series_a}-{series_b})",
                    impact=1.0 if winner == "a" else -1.0, match_state=state,
                ))

        # Save state for next comparison
        self._last_states[match_id] = {
            "score_a": new_score_a,
            "score_b": new_score_b,
            "series_a": series_a,
            "series_b": series_b,
            "round_phase": round_phase,
            "game_ended": game_ended,
            "prob_a": state.win_probability_a,
        }

    def set_polling_feed(self, polling_feed):
        """Link to the polling feed to resolve team names."""
        self._polling_feed = polling_feed

    def _fetch_team_names(self, match_id: int) -> tuple:
        """Get team names from the polling feed, then scrape Bo3.gg.

        IMPORTANT: The WS team_1/team_2 ordering may differ from the polling
        feed's team_one/team_two. We return names in WS order (team_1, team_2)
        and only use the polling feed as a name source, not for ordering.
        """
        mid = str(match_id)
        # Check polling feed for names
        if hasattr(self, '_polling_feed') and self._polling_feed:
            for pmid, pstate in self._polling_feed._matches.items():
                if pmid == mid and pstate.team_a and not pstate.team_a.startswith("Team"):
                    return (pstate.team_a, pstate.team_b)
        # Try Bo3.gg REST API (fast, reliable)
        try:
            import urllib.request, urllib.parse, json as _json
            # Use filter[matches.id][eq]=X — the direct /matches/{id} route returns
            # "not found" for most IDs; the filtered list query does work.
            qs = urllib.parse.urlencode({
                "filter[matches.id][eq]": match_id,
                "with": "teams",
            })
            url = f"https://api.bo3.gg/api/v1/matches?{qs}"
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            resp = urllib.request.urlopen(req, timeout=5)
            data = _json.loads(resp.read())
            results = data.get("results") or []
            if results:
                row = results[0]
                # Prefer bet_updates.team_N.name (short display name) over slug split.
                bu = row.get("bet_updates") or {}
                t1 = (bu.get("team_1") or {}).get("name", "").strip()
                t2 = (bu.get("team_2") or {}).get("name", "").strip()
                # Fallback: parse slug "team-a-vs-team-b-DD-MM-YYYY"
                if not (t1 and t2):
                    slug = row.get("slug", "")
                    import re as _re
                    m = _re.match(r"^(.+?)-vs-(.+?)-\d{2}-\d{2}-\d{4}$", slug)
                    if m:
                        t1 = m.group(1).replace("-", " ").title()
                        t2 = m.group(2).replace("-", " ").title()
                if t1 and t2:
                    logger.info(f"[BO3-WS] Resolved via API filter: {t1} vs {t2} (id={match_id})")
                    return (t1, t2)
        except Exception as e:
            logger.debug(f"[BO3-WS] API filter failed for {match_id}: {e}")
        # Try scraping Bo3.gg match page
        try:
            import urllib.request, re
            url = f"https://bo3.gg/matches/{match_id}"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"
            })
            resp = urllib.request.urlopen(req, timeout=5)
            html = resp.read().decode("utf-8", errors="replace")
            # Look for __NEXT_DATA__ JSON with match teams
            m = re.search(r'<script id="__NEXT_DATA__".*?>(.*?)</script>', html)
            if m:
                import json
                d = json.loads(m.group(1))
                page = d.get("props", {}).get("pageProps", {})
                match = page.get("match", page.get("initialMatch", {}))
                t1 = match.get("team1", {}).get("name", "")
                t2 = match.get("team2", {}).get("name", "")
                if t1 and t2:
                    logger.info(f"[BO3-WS] Resolved via scrape: {t1} vs {t2}")
                    return (t1, t2)
            # Fallback: parse <title> tag "Team1 vs Team2"
            title_m = re.search(r"<title>(.*?)\s+vs\s+(.*?)[\s|-]", html)
            if title_m:
                t1 = title_m.group(1).strip()
                t2 = title_m.group(2).strip()
                if t1 and t2:
                    logger.info(f"[BO3-WS] Resolved via title: {t1} vs {t2}")
                    return (t1, t2)
        except Exception as e:
            logger.debug(f"[BO3-WS] Scrape failed for {match_id}: {e}")
        return (f"Team1_{match_id}", f"Team2_{match_id}")

    def estimate_win_probability(self, state: MatchState) -> float:
        """Simple CS2 win probability from round + series score."""
        # Series component
        maps_needed = (state.total_maps // 2) + 1
        sa = state.score_a
        sb = state.score_b
        a_needs = maps_needed - sa
        b_needs = maps_needed - sb

        if a_needs <= 0:
            return 0.99
        if b_needs <= 0:
            return 0.01

        series_prob = max(0.01, min(0.99, b_needs / (a_needs + b_needs)))

        # Map component — rounds needed to win (13 in regulation)
        ra = state.round_score_a
        rb = state.round_score_b
        if ra + rb > 0:
            # Approximate map win probability from round score
            a_rounds_need = max(1, 13 - ra)
            b_rounds_need = max(1, 13 - rb)
            map_prob = max(0.01, min(0.99, b_rounds_need / (a_rounds_need + b_rounds_need)))
            # Blend series and map probability
            return series_prob * 0.4 + map_prob * 0.6
        return series_prob

    async def disconnect(self):
        self._running = False
        if self._ws:
            await self._ws.close()
        if self._session:
            await self._session.close()
        self._connected = False

    async def subscribe_match(self, match_id: str):
        self._subscribed_matches.add(match_id)

    async def unsubscribe_match(self, match_id: str):
        self._subscribed_matches.discard(match_id)

    async def get_live_matches(self):
        return [m for m in self._matches.values() if m.is_live]

    async def discover_live_matches(self):
        return list(self._subscribed_matches)
