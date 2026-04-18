"""
Kambi sportsbook feed — live map scores + pro bookmaker odds via the public
offering CDN. Covers CS2, Dota2, Valorant, LoL under a single endpoint.

WHY THIS FEED MATTERS:
  1. Kambi serves odds to regulated sportsbooks worldwide. Their odds reflect
     professional trader action and move BEFORE retail markets like Polymarket.
  2. Map score changes are reported within ~3s of real-world events — we emit
     these as MAP_WIN events so they feed the same decision pipeline.
  3. Listing is reachable without auth; no WebSocket needed.

WHAT IT DOES NOT GIVE US:
  Round-by-round timeline / kill feed. That data is served by widget.abiosgaming.com
  in an encrypted binary format and is out of reach without JS reverse engineering.

ENDPOINT:
  https://eu-offering-api.kambicdn.com/offering/v2018/kambi/listView/esports/all/all/all/in-play.json

SHAPE:
  {
    "events": [
      {
        "event": { "id": 1026994998, "name": "TYLOO - JD Gaming", "state": "STARTED",
                   "sport": "VALORANT", "group": "VCT: CN", "homeName": "TYLOO", ... },
        "liveData": { "score": {"home": "0", "away": "1", "who": "AWAY"}, "liveStatistics": [] },
        "betOffers": [{ "outcomes": [{"label": "TYLOO", "odds": 10000, ...}] }]
      }
    ]
  }

  odds are in milli-decimal: 10000 = 10.00 decimal odds.
"""
import asyncio
import logging
import time
from typing import Optional

import aiohttp

import config
from feeds.base import GameFeed, GameEvent, MatchState, EventType

logger = logging.getLogger(__name__)

LISTVIEW_URL = (
    "https://eu-offering-api.kambicdn.com/offering/v2018/kambi/"
    "listView/esports/all/all/all/in-play.json"
)
PARAMS = {"lang": "en_GB", "market": "GB", "includeParticipants": "true"}
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0",
    "Origin": "https://play.kambi.com",
    "Referer": "https://play.kambi.com/",
    "Accept": "application/json",
}

# Map Kambi's sport tag to our internal game key
SPORT_TO_GAME = {
    "COUNTER_STRIKE": "cs2",
    "CS2": "cs2",
    "DOTA_2": "dota2",
    "DOTA2": "dota2",
    "VALORANT": "valorant",
    "LEAGUE_OF_LEGENDS": "lol",
    "LOL": "lol",
}


class KambiFeed(GameFeed):
    """Polling feed for Kambi live scores + odds across all esports."""

    POLL_INTERVAL = 3  # Kambi CDN is cache-friendly at 3s; faster risks rate limit

    def __init__(self):
        # Game field is multi; we tag per-event with the actual game
        super().__init__("kambi")
        self._session: Optional[aiohttp.ClientSession] = None
        self._poll_task = None
        # Per-event cached state for diff detection
        self._last_state: dict[int, dict] = {}  # kambi_id -> {score_home, score_away, odds_home, odds_away}
        self._subscribed: set[int] = set()

    async def connect(self):
        self._session = aiohttp.ClientSession()
        self._connected = True
        self._connect_time = time.time()
        self._running = True

        self._emit(GameEvent(
            event_type=EventType.FEED_CONNECTED,
            match_id="", game="kambi", team="neutral",
            description="Connected to Kambi offering feed",
            impact=0.0,
        ))
        logger.info("Kambi feed connected")

        self._poll_task = asyncio.create_task(self._safe_run(self._poll_loop(), "kambi_poll"))

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
        # All live matches are auto-discovered via listView. Keep the signature for protocol compliance.
        try:
            self._subscribed.add(int(match_id))
        except ValueError:
            pass

    async def unsubscribe_match(self, match_id: str):
        try:
            self._subscribed.discard(int(match_id))
        except ValueError:
            pass

    async def get_live_matches(self) -> list[MatchState]:
        return list(self._matches.values())

    async def _fetch(self) -> Optional[list]:
        try:
            async with self._session.get(LISTVIEW_URL, params=PARAMS, headers=HEADERS,
                                          timeout=aiohttp.ClientTimeout(total=6)) as resp:
                if resp.status != 200:
                    logger.warning(f"Kambi listView HTTP {resp.status}")
                    return None
                data = await resp.json()
                return data.get("events", [])
        except Exception as e:
            logger.warning(f"Kambi fetch failed: {e}")
            return None

    async def _poll_loop(self):
        logger.info("Kambi poll loop started")
        while self._running:
            try:
                events = await self._fetch()
                if events is None:
                    await asyncio.sleep(self.POLL_INTERVAL)
                    continue

                live_ids = set()
                for wrapper in events:
                    ev = wrapper.get("event", {})
                    if ev.get("state") != "STARTED":
                        continue

                    sport = ev.get("sport", "").upper()
                    game = SPORT_TO_GAME.get(sport)
                    if not game or game not in config.ENABLED_GAMES:
                        continue

                    kid = ev.get("id")
                    if not kid:
                        continue
                    live_ids.add(kid)
                    self._process_event(wrapper, ev, game, kid)

                # Clean up stopped matches
                dropped = set(self._last_state.keys()) - live_ids
                for kid in dropped:
                    self._last_state.pop(kid, None)

            except Exception as e:
                logger.error(f"Kambi poll error: {e}", exc_info=True)

            await asyncio.sleep(self.POLL_INTERVAL)

    def _process_event(self, wrapper: dict, ev: dict, game: str, kid: int):
        """Detect changes in scores and odds, emit events."""
        live_data = wrapper.get("liveData", {})
        score = live_data.get("score", {}) or {}
        try:
            home_score = int(score.get("home", 0))
            away_score = int(score.get("away", 0))
        except (TypeError, ValueError):
            return

        home_name = ev.get("homeName", "?")
        away_name = ev.get("awayName", "?")
        match_id = f"kambi_{kid}"

        # Extract match winner odds from the primary betOffer
        odds_home = 0.0
        odds_away = 0.0
        for bo in wrapper.get("betOffers", []) or []:
            outcomes = bo.get("outcomes", []) or []
            if len(outcomes) != 2:
                continue
            # Milli-decimal → decimal
            for oc in outcomes:
                label = (oc.get("label") or oc.get("participant") or "").strip()
                odds_dec = (oc.get("odds") or 0) / 1000.0
                if odds_dec <= 0:
                    continue
                if label == home_name:
                    odds_home = odds_dec
                elif label == away_name:
                    odds_away = odds_dec
            if odds_home > 0 and odds_away > 0:
                break

        # First time seeing this match — initialize state, emit MATCH_START.
        # Wait for valid odds before locking in baseline state so later diffs work.
        if kid not in self._last_state:
            if odds_home <= 0 or odds_away <= 0:
                # Skip initial poll if no odds — we'll retry next tick.
                # Still create the MatchState so the match shows up as tracked.
                state = MatchState(
                    match_id=match_id, game=game,
                    team_a=home_name, team_b=away_name,
                    score_a=home_score, score_b=away_score,
                    is_live=True, started_at=time.time(),
                )
                state.extra["kambi_id"] = kid
                self._matches[match_id] = state
                return
            self._last_state[kid] = {
                "score_home": home_score, "score_away": away_score,
                "odds_home": odds_home, "odds_away": odds_away,
            }
            state = MatchState(
                match_id=match_id, game=game,
                team_a=home_name, team_b=away_name,
                score_a=home_score, score_b=away_score,
                is_live=True, started_at=time.time(),
            )
            state.extra["kambi_id"] = kid
            state.extra["odds_a"] = odds_home
            state.extra["odds_b"] = odds_away
            state.extra["kambi_prob_a"] = (1.0 / odds_home) if odds_home > 0 else 0.5
            self._matches[match_id] = state
            logger.info(
                f"[KAMBI] INIT {game} {home_name} vs {away_name} | "
                f"score={home_score}-{away_score} | odds={odds_home:.2f}/{odds_away:.2f}"
            )
            self._emit(GameEvent(
                event_type=EventType.MATCH_START,
                match_id=match_id, game=game, team="neutral",
                description=f"{home_name} vs {away_name} LIVE (Kambi)",
                impact=0.0, match_state=state,
            ))
            return

        last = self._last_state[kid]
        state = self._matches.get(match_id)
        if state is None:
            return
        state.score_a = home_score
        state.score_b = away_score
        state.extra["odds_a"] = odds_home
        state.extra["odds_b"] = odds_away
        state.extra["kambi_prob_a"] = (1.0 / odds_home) if odds_home > 0 else 0.5

        # Score change → MAP_WIN event
        if (home_score, away_score) != (last["score_home"], last["score_away"]):
            winner = "a" if home_score > last["score_home"] else "b"
            winner_name = home_name if winner == "a" else away_name
            logger.info(
                f"[KAMBI] MAP_WIN {game} {home_name} vs {away_name} | "
                f"{last['score_home']}-{last['score_away']} → {home_score}-{away_score} "
                f"({winner_name})"
            )
            # Derive impact from odds change
            old_prob = 1.0 / last["odds_home"] if last["odds_home"] > 0 else 0.5
            new_prob = 1.0 / odds_home if odds_home > 0 else old_prob
            self._emit(GameEvent(
                event_type=EventType.MAP_WIN,
                match_id=match_id, game=game, team=winner,
                description=f"Kambi: {winner_name} wins map (series {home_score}-{away_score})",
                impact=(new_prob - old_prob),
                match_state=state,
            ))

        # Significant odds move (≥10% probability shift) → SCORE_UPDATE as signal
        if last["odds_home"] > 0 and odds_home > 0:
            old_prob = 1.0 / last["odds_home"]
            new_prob = 1.0 / odds_home
            if abs(new_prob - old_prob) >= 0.05:  # 5% prob shift on the book
                direction = "a" if new_prob > old_prob else "b"
                logger.info(
                    f"[KAMBI] ODDS MOVE {game} {home_name}: "
                    f"{last['odds_home']:.2f} → {odds_home:.2f} "
                    f"(prob {old_prob*100:.0f}¢ → {new_prob*100:.0f}¢)"
                )
                self._emit(GameEvent(
                    event_type=EventType.SCORE_UPDATE,
                    match_id=match_id, game=game, team=direction,
                    description=(
                        f"Kambi odds moved: {home_name} "
                        f"{last['odds_home']:.2f}→{odds_home:.2f} "
                        f"(book {old_prob*100:.0f}¢→{new_prob*100:.0f}¢)"
                    ),
                    impact=(new_prob - old_prob),
                    match_state=state,
                ))

        # Update cache
        self._last_state[kid] = {
            "score_home": home_score, "score_away": away_score,
            "odds_home": odds_home, "odds_away": odds_away,
        }
