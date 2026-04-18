"""
Continuous Match Analyzer — feeds ALL game data to Gemma 4 on RTX 5090.

Runs every 5 seconds, analyzing ALL active matches with Polymarket markets.
Sends rich game state, event history, market data, and historical patterns.
Uses the local LLM for fast ($0) inference on every match continuously.

This supplements the event-driven edge trading with constant monitoring.
"""
import asyncio
import aiohttp
import json
import logging
import os
import time
from typing import Optional

import config

logger = logging.getLogger(__name__)

ANALYZER_PROMPT = """You are an elite esports betting analyst running on a dedicated RTX 5090 GPU.
You continuously monitor live matches and identify profitable betting opportunities.

Decide if there's a tradeable edge. CS2: economy+rounds matter. DOTA2: gold>kills. We buy at ASK, sell at BID.
JSON: {"action":"buy_a/buy_b/monitor","confidence":0-1,"team":"name","reason":"brief","urgency":"now/wait/no"}"""


class ContinuousMatchAnalyzer:
    """Runs on a loop, feeding ALL match data to local LLM for continuous analysis."""

    def __init__(self, bot):
        self._bot = bot
        self._running = False
        self._ollama_url = config.OLLAMA_URL
        self._model = config.OLLAMA_MODEL
        self._interval = 8  # seconds between full scans
        self._match_history: dict[str, list] = {}  # match_id -> event history
        self._last_analysis: dict[str, dict] = {}  # match_id -> last analysis result
        self._analysis_count = 0

    async def run(self):
        """Main loop — analyze all active matches continuously."""
        if not self._ollama_url:
            logger.info("[ANALYZER] No OLLAMA_URL configured, skipping continuous analysis")
            return

        self._running = True
        self._pause_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".pause_analyzer")
        self._last_keepalive = 0
        logger.info(f"[ANALYZER] Continuous match analyzer started — {self._model} @ {self._ollama_url}")
        await asyncio.sleep(15)

        while self._running:
            try:
                if os.path.exists(self._pause_file):
                    await asyncio.sleep(5)
                    continue

                # Keepalive ping every 2 min — prevents Ollama from unloading model
                if time.time() - self._last_keepalive > 120:
                    await self._keepalive_ping()
                    self._last_keepalive = time.time()

                await self._analyze_all_matches()
            except Exception as e:
                logger.error(f"[ANALYZER] Error: {e}")
            await asyncio.sleep(self._interval)

    async def _keepalive_ping(self):
        """Ping Ollama to keep the model loaded in VRAM."""
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self._model,
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": False,
                    "think": False,
                    "keep_alive": -1,
                    "options": {"num_predict": 1},
                }
                async with session.post(
                    f"{self._ollama_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    await resp.json()
                logger.info("[ANALYZER] Keepalive OK — model stays in VRAM")
        except Exception as e:
            logger.warning(f"[ANALYZER] Keepalive failed: {e}")

    async def _analyze_all_matches(self):
        """Analyze every active match with a Polymarket market."""
        linked = dict(self._bot.latency_analyzer._match_to_market)
        if not linked:
            return

        # Get all active matches with markets and real game data
        analyzable = []
        for match_id, market in linked.items():
            # Find match state from any feed
            state = None
            for feed in self._bot.feeds.values():
                if match_id in feed._matches:
                    state = feed._matches[match_id]
                    break

            if not state or not state.is_live:
                continue

            # Skip resolved markets
            if self._bot.polymarket_ws:
                ws = self._bot.polymarket_ws.get_price(market.token_id_a)
                if ws and (ws.get("best_bid", 0) > 0.95 or ws.get("best_bid", 0) < 0.05):
                    continue

            analyzable.append((match_id, state, market))

        if not analyzable:
            return

        # Prioritize matches with recent events (active gameplay)
        def sort_key(item):
            mid, st, mkt = item
            events = self._match_history.get(mid, [])
            if not events:
                return 100  # still analyze — market state alone is useful
            last_event_age = time.time() - events[-1].get("time", 0)
            if last_event_age < 60:
                return 1000 - last_event_age
            elif last_event_age < 300:
                return 500 - last_event_age
            return 50

        analyzable.sort(key=sort_key, reverse=True)

        # Analyze ALL live matches (esports-trader is fast enough)
        for match_id, state, market in analyzable[:5]:
            try:
                await self._analyze_match(match_id, state, market)
            except Exception as e:
                logger.error(f"[ANALYZER] Match error {match_id}: {e}")

    async def _analyze_match(self, match_id: str, state, market):
        """Send full match context to Gemma for analysis."""
        # Build rich context
        ws_a = self._bot.polymarket_ws.get_price(market.token_id_a) if self._bot.polymarket_ws else None
        ws_b = self._bot.polymarket_ws.get_price(market.token_id_b) if self._bot.polymarket_ws and market.token_id_b else None

        bid_a = ws_a["best_bid"] if ws_a and ws_a.get("best_bid", 0) > 0 else 0
        ask_a = ws_a["best_ask"] if ws_a and ws_a.get("best_ask", 0) > 0 else 0
        bid_b = ws_b["best_bid"] if ws_b and ws_b.get("best_bid", 0) > 0 else 0
        ask_b = ws_b["best_ask"] if ws_b and ws_b.get("best_ask", 0) > 0 else 0

        spread = ask_a - bid_a if ask_a > 0 and bid_a > 0 else 0

        # Skip if no orderbook
        if bid_a <= 0 and ask_a <= 0:
            return

        # Skip if spread is extreme
        if spread > 0.30:
            return

        # Build prompt — team alignment is CRITICAL
        # feed.team_a may differ from market.team_a — make it explicit
        lines = []
        lines.append(f"GAME: {state.game.upper()}")
        lines.append(f"FEED TEAMS: team_a={state.team_a} | team_b={state.team_b}")
        lines.append(f"MARKET: token_a={market.team_a} | token_b={market.team_b}")
        lines.append(f"Series: {state.score_a}-{state.score_b} | Round: {state.round_score_a}-{state.round_score_b}")
        lines.append(f"IMPORTANT: buy_a means buy {state.team_a}. buy_b means buy {state.team_b}.")

        # Game-specific data
        ex = state.extra or {}
        if state.game == "cs2":
            side = ex.get("team_a_current_side", "?")
            eco_a = ex.get("team_a_money", 0)
            eco_b = ex.get("team_b_money", 0)
            kills_a = ex.get("round_kills_a", 0)
            kills_b = ex.get("round_kills_b", 0)
            lines.append(f"Side: {state.team_a} on {side}")
            if eco_a or eco_b:
                lines.append(f"Economy: {state.team_a}=${eco_a:,} | {state.team_b}=${eco_b:,}")
            if kills_a or kills_b:
                lines.append(f"Round kills: {kills_a}-{kills_b}")
        elif state.game in ("dota2", "lol"):
            kills_a = ex.get("kills_a", 0)
            kills_b = ex.get("kills_b", 0)
            gold = ex.get("gold_lead", 0)
            mins = ex.get("game_minutes", 0)
            lines.append(f"Kills: {state.team_a} {kills_a} - {kills_b} {state.team_b} | {mins:.0f}min")
            if gold != 0:
                gold_team = state.team_a if gold > 0 else state.team_b
                lines.append(f"GOLD LEAD: {gold_team} +{abs(gold):,} gold")

        # Market data — clearly labeled per team
        mtype = getattr(market, '_market_type', 'series')
        lines.append(f"\nMARKET ({mtype}) Vol: ${market.volume:,.0f}")
        lines.append(f"  {market.team_a} (token_a): bid={bid_a*100:.0f}c ask={ask_a*100:.0f}c")
        if bid_b > 0:
            lines.append(f"  {market.team_b} (token_b): bid={bid_b*100:.0f}c ask={ask_b*100:.0f}c")
        if ask_a > 0 and spread > 0:
            lines.append(f"  Spread: {spread*100:.0f}c ({spread/ask_a*100:.0f}%)")

        # Orderbook momentum
        if self._bot.polymarket_ws and market.token_id_a:
            mom = self._bot.polymarket_ws.get_momentum(market.token_id_a, window=30)
            stale = self._bot.polymarket_ws.get_staleness(market.token_id_a)
            if mom.get("trend", "unknown") != "unknown":
                bc = mom.get("bid_change", 0)
                lines.append(f"  Momentum: {mom['first_bid']*100:.0f}c→{mom['last_bid']*100:.0f}c ({bc*100:+.1f}c) {mom['trend'].upper()} | {mom['trades']} updates")
            if stale > 30:
                lines.append(f"  ⚠️ STALE: {stale:.0f}s since last update")

        # Event history — keep compact for speed
        events = self._match_history.get(match_id, [])
        if events:
            lines.append(f"\nEVENTS ({len(events)}):")
            for e in events[-6:]:
                lines.append(f"  [{e.get('type','?')}] {e.get('desc','')[:50]}")

        lines.append(f"\n{market.question}")
        lines.append(f"\nAnalyze this match. Is there a tradeable edge RIGHT NOW?")

        prompt = "\n".join(lines)

        # Call Gemma
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": ANALYZER_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "think": False,
            "format": "json",   # force valid JSON
            "keep_alive": -1,
            "options": {"temperature": 0, "num_predict": 500},
        }

        t0 = time.time()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self._ollama_url}/api/chat",
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    data = await resp.json()

            msg = data.get("message", {})
            response_text = msg.get("content", "") or msg.get("thinking", "")
            latency = (time.time() - t0) * 1000
            self._analysis_count += 1

            # Parse response
            text = response_text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            try:
                result = json.loads(text)
            except json.JSONDecodeError:
                return

            self._last_analysis[match_id] = result

            action = result.get("action", "monitor")
            conf = result.get("confidence", 0)
            reason = result.get("reason", "")
            urgency = result.get("urgency", "no")
            team = result.get("team", "")

            # Log all analysis
            if action in ("buy_a", "buy_b") and conf >= 0.7:
                logger.info(
                    f"[ANALYZER] {urgency.upper()} BUY {team} ({state.game}) | "
                    f"conf={conf:.0%} | {reason[:60]} | {latency:.0f}ms"
                )
                # If urgent and high confidence, emit a trade signal
                if urgency == "now" and conf >= 0.8:
                    self._emit_trade_signal(match_id, state, market, result, bid_a, ask_a, bid_b, ask_b)
            elif action == "monitor":
                logger.info(f"[ANALYZER] WATCH {state.team_a} vs {state.team_b} ({state.game}) | {reason[:60]} | {latency:.0f}ms")
            else:
                pass  # skip, don't log noise

        except asyncio.TimeoutError:
            logger.warning(f"[ANALYZER] LLM timeout for {state.team_a} vs {state.team_b}")
        except Exception as e:
            logger.error(f"[ANALYZER] LLM error for {state.team_a} vs {state.team_b}: {type(e).__name__}: {e}")

    def _emit_trade_signal(self, match_id, state, market, analysis, bid_a, ask_a, bid_b, ask_b):
        """Convert analyzer BUY signal into a trade signal for the executor."""
        from latency import TradeSignal
        from team_match import teams_match

        action = analysis.get("action", "")
        team_name = analysis.get("team", "").strip()
        conf = analysis.get("confidence", 0)
        reason = analysis.get("reason", "")

        # ─── TEAM ALIGNMENT ──────────────────────────────────────────
        # The LLM says "buy_a" meaning feed's team_a. But we need to figure out
        # which POLYMARKET TOKEN corresponds to that team.
        # feed.team_a might be "The goal of all life" but market.team_a might be "Roar Gaming"
        # We must match by team NAME, not by a/b index.

        # First: determine which feed team the LLM wants to buy
        if action == "buy_a":
            feed_buy_team = state.team_a
        elif action == "buy_b":
            feed_buy_team = state.team_b
        else:
            return

        # If LLM returned a team name, verify it matches
        if team_name and not teams_match(team_name, feed_buy_team):
            # LLM team name doesn't match the action — check if it matches the OTHER team
            other_team = state.team_b if action == "buy_a" else state.team_a
            if teams_match(team_name, other_team):
                # LLM said buy_a but named team_b — FLIP the action
                logger.warning(f"[ANALYZER] TEAM FLIP: LLM said {action} but named {team_name} = flipping")
                action = "buy_b" if action == "buy_a" else "buy_a"
                feed_buy_team = other_team
            else:
                logger.warning(f"[ANALYZER] TEAM MISMATCH: LLM named '{team_name}' doesn't match {state.team_a} or {state.team_b}")
                return

        # Now map feed team → market token
        # Check if feed's buy team matches market's team_a or team_b
        if teams_match(feed_buy_team, market.team_a):
            buy_team = "a"
            buy_name = market.team_a
            token_id = market.token_id_a
            buy_price = ask_a if ask_a > 0 else market.price_a
            buy_bid = bid_a
        elif teams_match(feed_buy_team, market.team_b):
            buy_team = "b"
            buy_name = market.team_b
            token_id = market.token_id_b or market.token_id_a
            buy_price = ask_b if ask_b > 0 else market.price_b
            buy_bid = bid_b
        else:
            logger.warning(f"[ANALYZER] CANT MATCH: feed team '{feed_buy_team}' to market teams '{market.team_a}'/'{market.team_b}'")
            return

        buy_spread = buy_price - buy_bid if buy_bid > 0 else 0

        # ─── SAFETY FILTERS (same as edge system) ────────────────────
        mtype = getattr(market, '_market_type', 'series')

        # Block ALL map_winner for Dota2/LoL/Valorant
        if mtype == 'map_winner' and state.game in ('dota2', 'lol', 'valorant'):
            logger.info(f"[ANALYZER] BLOCKED — {state.game} map_winner")
            return
        # Block map_winner above 80c
        if mtype == 'map_winner' and buy_price > 0.80:
            logger.info(f"[ANALYZER] BLOCKED — map_winner at {buy_price*100:.0f}c")
            return
        # Price range 15-85c
        if buy_price <= 0.15 or buy_price >= 0.85:
            logger.info(f"[ANALYZER] BLOCKED — price {buy_price*100:.0f}c outside 15-85c")
            return
        # Token resolved
        if buy_bid < 0.05 and buy_price > 0.20:
            logger.info(f"[ANALYZER] BLOCKED — resolved (bid={buy_bid:.3f})")
            return
        # Spread > 20%
        if buy_spread > 0.20:
            logger.info(f"[ANALYZER] BLOCKED — spread {buy_spread*100:.0f}%")
            return
        # Max 2 open positions per team
        executor = self._bot.executor
        team_exposure = sum(p.amount for p in executor.positions
                          if not p.resolved and p.team.lower() == buy_name.lower())
        if team_exposure >= 50:
            logger.info(f"[ANALYZER] BLOCKED — max exposure on {buy_name} (${team_exposure:.0f})")
            return

        bet_amount = config.EDGE_BOT_BET_SIZE
        if conf >= 0.9:
            bet_amount *= 2

        signal = TradeSignal(
            match_id=match_id, game=state.game, market=market,
            team=buy_team, team_name=buy_name, token_id=token_id,
            confidence=conf,
            our_price=state.win_probability_a if buy_team == "a" else 1 - state.win_probability_a,
            market_price=buy_price,
            edge=0.05,
            latency_edge_ms=0,
            bet_amount=bet_amount,
            reason=f"ANALYZER: {conf:.0%} | {reason[:50]} | buy@{buy_price:.3f}",
        )
        signal._analysis = {
            "confidence": conf,
            "reason": reason,
            "game_state": {
                "game": state.game, "team_a": state.team_a, "team_b": state.team_b,
                "score_a": state.score_a, "score_b": state.score_b,
                "round_a": state.round_score_a, "round_b": state.round_score_b,
            },
            "market_state": {
                "price": buy_price, "bid": bid_a, "ask": ask_a, "spread": ask_a - bid_a,
                "volume": market.volume, "market_type": getattr(market, '_market_type', 'series'),
                "question": market.question[:50],
            },
            "source": "continuous_analyzer",
        }

        self._bot._on_trade_signal(signal)
        logger.info(f"[ANALYZER] SIGNAL EMITTED: {buy_name} ({state.game}) @ {buy_price:.3f}")

    def add_event(self, match_id: str, event_type: str, description: str, team: str = ""):
        """Track event for match history."""
        if match_id not in self._match_history:
            self._match_history[match_id] = []
        self._match_history[match_id].append({
            "type": event_type,
            "desc": description[:80],
            "team": team,
            "time": time.time(),
        })
        # Keep last 20 events per match
        if len(self._match_history[match_id]) > 20:
            self._match_history[match_id] = self._match_history[match_id][-20:]

    def get_stats(self) -> dict:
        return {
            "analyzer_calls": self._analysis_count,
            "matches_tracked": len(self._match_history),
            "active_analyses": len(self._last_analysis),
        }
