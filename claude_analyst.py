"""
Claude AI Expert Panel for esports betting decisions.

Replaces mathematical confidence scoring with a panel of 10 AI esports experts
who vote on each betting opportunity. Uses Claude Sonnet for cost-effective,
high-quality analysis.

Cost: ~$0.008 per call, max 30 calls per match = ~$0.24/match.
"""
import json
import time
import hashlib
import logging
from dataclasses import dataclass
from typing import Optional

import anthropic

import config
from news_scraper import NewsScraper

logger = logging.getLogger(__name__)

EXPERT_SYSTEM_PROMPT = """You are a panel of 11 esports betting experts analyzing a LIVE match for a latency arbitrage opportunity. Our data feed is 2-15 seconds ahead of public broadcasts, so the market hasn't reacted to the latest event yet.

Each expert has a specialty:
1. Game Specialist - Deep knowledge of this game's mechanics, meta, and team tendencies
2. Momentum Analyst - Reads momentum shifts, tilt, and psychological pressure
3. Economy Expert - Analyzes resource management (money in CS2, gold/XP in MOBAs)
4. Map/Draft Specialist - Understands map pools, agent/hero picks, side advantages
5. Statistical Modeler - Pure probability and expected value calculations
6. Market Analyst - Reads betting market movements and liquidity patterns
7. Risk Manager - Evaluates downside scenarios and position sizing
8. Timing Expert - Assesses whether the market will correct before we can act
9. Series Strategist - Understands Bo3/Bo5 dynamics, how momentum carries across maps
10. Contrarian - Actively looks for reasons the obvious bet is WRONG
11. Insider Intel - Has their ear to the ground on social media, player/team/coach announcements, roster changes, injuries, travel issues, bootcamp info, and community buzz. Cross-references any RECENT NEWS provided to assess whether the market has priced in this information or if we have an informational edge.

Each expert votes BET or SKIP with a one-sentence reason.
Then provide a consensus confidence score from 0.0 to 1.0.

IMPORTANT: Only vote BET if the edge is REAL and ACTIONABLE. Consider:
- Is this edge likely to persist for the next 5-10 seconds? (we need time to execute)
- Could the market already be pricing in information we don't have?
- Is the edge large enough to overcome the ~2% spread + fees?
- Is the data quality high enough to trust our probability estimate?
- Are there any recent news, roster changes, or player issues that affect this match?

Respond ONLY with valid JSON in this exact format:
{
  "votes": [
    {"expert": "Game Specialist", "vote": "BET", "reason": "..."},
    {"expert": "Momentum Analyst", "vote": "SKIP", "reason": "..."},
    {"expert": "Economy Expert", "vote": "BET", "reason": "..."},
    {"expert": "Map/Draft Specialist", "vote": "BET", "reason": "..."},
    {"expert": "Statistical Modeler", "vote": "BET", "reason": "..."},
    {"expert": "Market Analyst", "vote": "SKIP", "reason": "..."},
    {"expert": "Risk Manager", "vote": "SKIP", "reason": "..."},
    {"expert": "Timing Expert", "vote": "BET", "reason": "..."},
    {"expert": "Series Strategist", "vote": "BET", "reason": "..."},
    {"expert": "Contrarian", "vote": "SKIP", "reason": "..."},
    {"expert": "Insider Intel", "vote": "BET", "reason": "..."}
  ],
  "bet_votes": 7,
  "skip_votes": 4,
  "confidence": 0.65,
  "action": "bet",
  "reasoning": "One sentence consensus summary",
  "dissent": "Strongest counter-argument against betting"
}"""


@dataclass
class AnalysisResult:
    """Result from the Claude expert panel."""
    confidence: float           # 0.0-1.0 consensus confidence
    action: str                 # "bet" or "skip"
    bet_votes: int              # how many of 11 voted to bet
    reasoning: str              # one-line consensus summary
    dissent: str                # strongest counter-argument
    votes: list = None          # individual expert votes [{expert, vote, reason}]
    raw_response: str = ""      # full response for debugging


class ClaudeAnalyst:
    """Wraps Claude Sonnet API for esports betting analysis."""

    def __init__(self):
        api_key = config.ANTHROPIC_API_KEY
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self._client = anthropic.AsyncAnthropic(api_key=api_key)
        self._model = getattr(config, "CLAUDE_MODEL", "claude-sonnet-4-6")
        self._news = NewsScraper()
        self._cache: dict[str, tuple[float, AnalysisResult]] = {}
        self._cache_ttl = 120.0  # 2 minutes — avoid re-analyzing same match state
        self._match_call_counts: dict[str, int] = {}
        self._match_call_limit = getattr(config, "CLAUDE_MAX_CALLS_PER_MATCH", 30)
        self._news_cache: dict[str, list[str]] = {}  # match_id -> headlines

        # Stats
        self.call_count = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self.consecutive_errors = 0
        self.last_error = ""
        self.recent_analyses: list[dict] = []  # last 10 full analyses for dashboard

    def _cache_key(self, event) -> str:
        """Generate cache key from game state — same state within TTL returns cached result."""
        s = event.match_state
        raw = f"{event.match_id}:{event.event_type.value}:{s.score_a}-{s.score_b}:{s.round_score_a}-{s.round_score_b}:{s.current_map}"
        return hashlib.md5(raw.encode()).hexdigest()[:12]

    def _format_news(self, news: list) -> str:
        if not news:
            return "No recent news found. The Insider Intel expert should note the ABSENCE of news — no roster changes or issues detected, which is itself informative (teams are likely at full strength with standard preparation)."
        lines = ["Recent headlines (treat as unverified — cross-reference and assess credibility):"]
        for h in news:
            lines.append(f"- {h}")
        lines.append("NOTE: Assess each headline critically. Reddit posts may be rumors or speculation. HLTV/official sources are more reliable. Only factor in news you judge to be credible and relevant.")
        return "\n".join(lines)

    def _format_ref_odds(self, event) -> str:
        """Format reference bookmaker odds for the prompt."""
        extra = event.match_state.extra if event.match_state else {}
        odds = extra.get("ref_odds")
        if not odds:
            return "No external bookmaker odds available for this match."

        lines = []
        consensus = odds.get("consensus_prob_a", 0.5)
        pinnacle = odds.get("pinnacle_prob_a")
        num_books = odds.get("num_bookmakers", 0)

        lines.append(f"Bookmaker consensus: {consensus:.1%} for {event.match_state.team_a} ({num_books} bookmakers)")
        if pinnacle:
            lines.append(f"Pinnacle (sharpest): {pinnacle:.1%} for {event.match_state.team_a}")

        # Compare with Polymarket
        poly_price = extra.get("prior_prob_a", 0.5)
        diff = poly_price - consensus
        if abs(diff) > 0.05:
            direction = "HIGHER" if diff > 0 else "LOWER"
            lines.append(f"WARNING: Polymarket ({poly_price:.1%}) is {abs(diff):.1%} {direction} than bookmaker consensus ({consensus:.1%})")

        for name, data in list(odds.get("bookmakers", {}).items())[:3]:
            lines.append(f"  {name}: {data['prob_a']:.1%}")

        return "\n".join(lines)

    def _format_team_intel(self, event) -> str:
        """Format team intelligence data for the prompt."""
        extra = event.match_state.extra if event.match_state else {}
        intel = extra.get("match_intel")
        if not intel:
            return "No detailed team intelligence available."

        lines = []
        for label, prof in [("TEAM A", intel.team_a), ("TEAM B", intel.team_b)]:
            name = prof.name
            parts = [f"{label}: {name}"]
            if prof.rating > 0:
                parts.append(f"Rating: {prof.rating:.0f}")
            if prof.ranking > 0:
                parts.append(f"Rank: #{prof.ranking}")
            if prof.recent_matches > 0:
                w = int(prof.recent_form * prof.recent_matches)
                l = prof.recent_matches - w
                parts.append(f"Form: {w}W-{l}L last {prof.recent_matches}")
            if prof.win_streak != 0:
                if prof.win_streak > 0:
                    parts.append(f"{prof.win_streak}-game WIN streak")
                else:
                    parts.append(f"{abs(prof.win_streak)}-game LOSS streak")
            if prof.schedule_strength > 0:
                parts.append(f"Avg opponent: {prof.schedule_strength:.0f} rating")
            if prof.hero_pool:
                heroes = ", ".join(f"{h['hero']} ({h['wr']*100:.0f}% wr, {h['picks']} picks)"
                                   for h in prof.hero_pool[:5])
                parts.append(f"Hero Pool: {heroes}")
            lines.append(" | ".join(parts))

        # H2H
        h2h_a = intel.h2h_wins_a
        h2h_b = intel.h2h_wins_b
        if h2h_a + h2h_b > 0:
            days = intel.h2h_last_played_days
            lines.append(f"HEAD-TO-HEAD: {intel.team_a.name} {h2h_a}-{h2h_b} {intel.team_b.name} (last played {days} days ago)")
        else:
            lines.append("HEAD-TO-HEAD: No recent meetings found")

        return "\n".join(lines)

    def _format_positions(self, positions: list) -> str:
        if not positions:
            return "None — this would be a NEW position."
        lines = []
        for p in positions:
            lines.append(f"- ${p.amount:.2f} on {p.team} @{p.fill_price:.3f} (conf={p.signal_confidence:.0%})")
        total = sum(p.amount for p in positions)
        lines.append(f"Total exposure: ${total:.2f} ({len(positions)} position{'s' if len(positions) > 1 else ''})")
        return "\n".join(lines)

    def _build_prompt(self, event, edge: float, staleness: float, market, existing_positions: list = None, news: list = None) -> str:
        """Build the user message with all match context."""
        s = event.match_state
        game = event.game.upper()

        bet_team = s.team_a if edge > 0 else s.team_b
        other_team = s.team_b if edge > 0 else s.team_a
        bet_side = "undervalued" if edge > 0 else "overvalued"

        # Game-specific context
        extra_ctx = ""
        if event.game == "cs2" and s.extra:
            money_a = s.extra.get("team_a_money", 0)
            money_b = s.extra.get("team_b_money", 0)
            side = s.extra.get("team_a_current_side", "?").upper()
            kills_a = s.extra.get("round_kills_a", 0)
            kills_b = s.extra.get("round_kills_b", 0)
            from feeds.cs2_model import classify_buy
            buy_a = classify_buy(money_a)
            buy_b = classify_buy(money_b)
            extra_ctx = f"""CS2 DETAILS:
- {s.team_a} side: {side} | Economy: ${money_a:,} ({buy_a})
- {s.team_b} Economy: ${money_b:,} ({buy_b})
- Round kills: {kills_a}-{kills_b}
- MR12 format (first to 13 rounds, OT at 12-12)"""

        elif event.game == "dota2" and s.extra:
            kills_a = s.extra.get("kills_a", 0)
            kills_b = s.extra.get("kills_b", 0)
            mins = s.extra.get("game_minutes", 0)
            gold_lead = s.extra.get("gold_lead", 0)
            towers_a = s.extra.get("towers_a", 0)
            towers_b = s.extra.get("towers_b", 0)
            gold_str = f"{abs(gold_lead)/1000:.1f}k for {s.team_a if gold_lead > 0 else s.team_b}" if gold_lead else "even"
            extra_ctx = f"""DOTA 2 DETAILS:
- Kill score: {s.team_a} {kills_a} - {kills_b} {s.team_b}
- Game time: {mins:.0f} minutes
- Kill lead: {kills_a - kills_b:+d} for {s.team_a if kills_a > kills_b else s.team_b}
- Gold lead: {gold_str} ({gold_lead:+,} net worth difference)
- Towers standing: {s.team_a} {towers_a} - {towers_b} {s.team_b}
- Gold context: {'MASSIVE advantage' if abs(gold_lead) > 15000 else 'significant lead' if abs(gold_lead) > 8000 else 'moderate lead' if abs(gold_lead) > 3000 else 'close game'}"""
            # Add hero picks if available
            heroes_a = s.extra.get("heroes_a", [])
            heroes_b = s.extra.get("heroes_b", [])
            if heroes_a:
                # Hero IDs — Claude knows Dota2 heroes
                extra_ctx += f"""
- {s.team_a} heroes (IDs): {heroes_a}
- {s.team_b} heroes (IDs): {heroes_b}
- Analyze team compositions: Are these teamfight-heavy drafts (AoE ultimates like Enigma, Tidehunter, Magnus)?
  Or farming/split-push drafts (Anti-Mage, Nature's Prophet, Terrorblade)?
  Teamfight drafts peak mid-game (20-35min), farming drafts scale late (35min+).
  This affects kill pace predictions for Over/Under props."""

        elif event.game == "lol":
            gold_a = s.extra.get("gold_a", 0)
            gold_b = s.extra.get("gold_b", 0)
            kills_a = s.extra.get("kills_a", 0)
            kills_b = s.extra.get("kills_b", 0)
            dragons_a = s.extra.get("dragons_a", 0)
            dragons_b = s.extra.get("dragons_b", 0)
            towers_a = s.extra.get("towers_a", 0)
            towers_b = s.extra.get("towers_b", 0)
            barons_a = s.extra.get("barons_a", 0)
            barons_b = s.extra.get("barons_b", 0)
            champs_a = s.extra.get("champions_a", [])
            champs_b = s.extra.get("champions_b", [])
            gold_diff = gold_a - gold_b
            extra_ctx = f"""LEAGUE OF LEGENDS DETAILS:
- Series score: {s.team_a} {s.score_a} - {s.score_b} {s.team_b}
- Gold: {s.team_a} {gold_a:,} vs {s.team_b} {gold_b:,} (lead: {gold_diff:+,} for {s.team_a if gold_diff > 0 else s.team_b})
- Kills: {kills_a} - {kills_b}
- Dragons: {dragons_a} - {dragons_b}
- Towers: {towers_a} - {towers_b}
- Barons: {barons_a} - {barons_b}
- {s.team_a} comp: {', '.join(champs_a) if champs_a else 'unknown'}
- {s.team_b} comp: {', '.join(champs_b) if champs_b else 'unknown'}
- Gold lead analysis: {'MASSIVE advantage' if abs(gold_diff) > 8000 else 'significant lead' if abs(gold_diff) > 4000 else 'slight lead' if abs(gold_diff) > 1500 else 'even'}"""

        elif event.game == "valorant":
            map_stats = s.extra.get("map_stats", [])
            extra_ctx = f"""VALORANT DETAILS:
- Map score: {s.team_a} {s.score_a} - {s.score_b} {s.team_b}"""
            if map_stats:
                for i, ms in enumerate(map_stats):
                    extra_ctx += f"\n  Map {i+1} ({ms.get('map','?')}): {ms.get('score','?')}"
                    for label, key in [("  " + s.team_a, "team1_players"), ("  " + s.team_b, "team2_players")]:
                        players = ms.get(key, [])
                        if players:
                            top = sorted(players, key=lambda p: -p.get("acs", 0))[:3]
                            pstr = ", ".join(f"{p['name']}({p['agent']}) {p['kills']}/{p['deaths']}/{p['assists']} ACS={p['acs']}" for p in top)
                            extra_ctx += f"\n  {label}: {pstr}"

        # Prior info
        prior = s.extra.get("prior_prob_a")
        prior_src = s.extra.get("prior_source", "")
        prior_ctx = ""
        if prior is not None:
            prior_ctx = f"\nPRE-MATCH PRIOR: {prior:.1%} for {s.team_a} (source: {prior_src})"

        # Detect market type
        mq = market.question.lower()
        is_handicap = "handicap" in mq
        is_map_winner = "map" in mq and "winner" in mq
        is_prop = market.team_a.lower() in ("over", "under", "yes", "no", "odd", "even")
        is_moneyline = not is_handicap and not is_map_winner and not is_prop

        market_type_ctx = ""
        if is_handicap:
            market_type_ctx = f"""
MARKET TYPE: HANDICAP
- This is a HANDICAP market: {market.question}
- Token A = "{market.team_a}" covers the handicap
- Token B = "{market.team_b}" covers the handicap
- IMPORTANT: Our model gives MATCH WIN probability, NOT handicap probability.
  For -1.5 handicap in Bo3: team must win 2-0 to cover.
  For +1.5 handicap: team just needs to win 1 map.
  Use the series score and game state to estimate handicap probability INDEPENDENTLY of our model.
  Current series: {s.score_a}-{s.score_b}. If favored team leads 1-0 and is winning current map,
  -1.5 handicap probability is HIGH. If series is 1-1, -1.5 is essentially dead."""
        elif is_prop:
            # Extract the prop type
            if "over" in mq or "under" in mq:
                import re as _re
                threshold = _re.search(r'(\d+\.?\d*)', market.question)
                threshold_str = threshold.group(1) if threshold else "?"
                market_type_ctx = f"""
MARKET TYPE: PROP BET — OVER/UNDER
- Question: {market.question}
- Token A = "{market.team_a}" (price: {market.price_a:.1%})
- Token B = "{market.team_b}" (price: {market.price_b:.1%})
- IMPORTANT: This is NOT a team-win bet. This is a statistical prop.
  You must estimate whether the stat will go OVER or UNDER {threshold_str}.
  Use the current game state (kills, game time, pace) to project the final total.
  Example: If 37 kills at 26 min, pace = 1.42 kills/min, projected ~55 kills in 40min game → OVER 47.5."""
            elif "rampage" in mq or "penta" in mq:
                market_type_ctx = f"""
MARKET TYPE: PROP BET — PLAYER ACHIEVEMENT
- Question: {market.question}
- Token A = "{market.team_a}" (Yes, price: {market.price_a:.1%})
- Token B = "{market.team_b}" (No, price: {market.price_b:.1%})
- IMPORTANT: Rampages (5-kill streak) are RARE (~5-10% of games).
  Consider: hero picks (AoE ult heroes more likely), game state (stomps more likely),
  and current kill pace. Market at {market.price_a:.1%} for Yes."""
            else:
                market_type_ctx = f"""
MARKET TYPE: PROP BET
- Question: {market.question}
- Token A = "{market.team_a}" (price: {market.price_a:.1%})
- Token B = "{market.team_b}" (price: {market.price_b:.1%})
- Analyze using current game state and statistics."""
        elif is_map_winner:
            market_type_ctx = f"""
MARKET TYPE: MAP WINNER
- This market is for the CURRENT MAP winner, not the series.
- Use round score ({s.round_score_a}-{s.round_score_b}) and economy to judge map winner probability."""

        return f"""LIVE {game} MATCH: {s.team_a} vs {s.team_b}
Series: {s.score_a}-{s.score_b} (Bo{s.total_maps})
Current Map {s.current_map}: {s.round_score_a}-{s.round_score_b}
Our model P({s.team_a} wins series): {s.win_probability_a:.1%}
{prior_ctx}
{market_type_ctx}

EVENT JUST OCCURRED: {event.event_type.value} — {event.description}
Estimated impact on win probability: {event.impact:+.3f}

MARKET STATE:
- Market: {market.question}
- Polymarket price: {market.price_a:.1%} for {market.team_a}, {market.price_b:.1%} for {market.team_b}
- Detected edge: {abs(edge):.1%} ({bet_team} appears {bet_side})
- Market staleness: {staleness:.0f}s since last price movement
- Market volume: ${market.volume:,.0f}

{extra_ctx}

TEAM INTELLIGENCE (for Game Specialist, Momentum Analyst, Contrarian):
{self._format_team_intel(event)}

BOOKMAKER REFERENCE ODDS (for Market Analyst):
{self._format_ref_odds(event)}

EXISTING POSITIONS ON THIS MATCH:
{self._format_positions(existing_positions)}

RECENT NEWS & SOCIAL MEDIA (for Insider Intel expert):
{self._format_news(news)}

PROPOSED BET: {bet_team} (our model says they are {bet_side})
Our model: {bet_team} at {abs(edge)+market.price_a if edge > 0 else 1-market.price_a+abs(edge):.1%} vs market at {market.price_a if edge > 0 else market.price_b:.1%}

IMPORTANT: If our model's probability seems WRONG given the game state (e.g., model says 3% but
team is clearly losing), the proposed bet may be on the WRONG SIDE. In that case:
- Vote SKIP if {bet_team} is clearly losing
- State in your reason that {other_team} is the correct side if applicable

For HANDICAP markets: ignore our model's match-win probability. Instead, estimate the handicap
probability yourself based on series score and game state. For example:
- Series 1-0, winning map 2 by 11-5 → -1.5 handicap (2-0 win) very likely (~75-85%)
- Series 1-1 → -1.5 handicap is dead (0%), +1.5 is guaranteed (100%)
- Series 0-0, map 1 close → -1.5 is speculative (~40-50%)

You may also recommend a CONVICTION BET even with zero market edge — if team intel, H2H,
momentum, and game state give you 80%+ confidence, that IS the edge.

If we already have a position, should we ADD or HOLD?"""

    async def analyze_event(self, event, edge: float, staleness: float,
                            market, existing_positions: list = None) -> Optional[AnalysisResult]:
        """Call Claude to analyze a betting opportunity. Returns None on failure."""
        # Cache check
        key = self._cache_key(event)
        if key in self._cache:
            cached_time, cached_result = self._cache[key]
            if time.time() - cached_time < self._cache_ttl:
                self.cache_hits += 1
                return cached_result

        # Per-match call limit
        mid = event.match_id
        count = self._match_call_counts.get(mid, 0)
        if count >= self._match_call_limit:
            logger.debug(f"[CLAUDE] Call limit reached for match {mid} ({count}/{self._match_call_limit})")
            return None

        # Fetch news for Insider Intel expert (cached per match)
        mid = event.match_id
        if mid not in self._news_cache:
            try:
                s = event.match_state
                self._news_cache[mid] = await self._news.get_match_news(
                    event.game, s.team_a, s.team_b
                )
            except Exception as e:
                logger.debug(f"News fetch failed: {e}")
                self._news_cache[mid] = []

        # Build prompt
        news = self._news_cache.get(mid, [])
        user_msg = self._build_prompt(event, edge, staleness, market, existing_positions, news)

        try:
            self._match_call_counts[mid] = count + 1
            self.call_count += 1

            response = await self._client.messages.create(
                model=self._model,
                max_tokens=1200,
                temperature=0,
                system=EXPERT_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )

            # Track cost (approximate)
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = (input_tokens * 3.0 + output_tokens * 15.0) / 1_000_000
            self.total_cost += cost

            # Parse response
            text = response.content[0].text.strip()
            # Handle markdown code blocks
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

            data = json.loads(text)

            result = AnalysisResult(
                confidence=max(0.0, min(1.0, float(data.get("confidence", 0.5)))),
                action=data.get("action", "skip"),
                bet_votes=int(data.get("bet_votes", 0)),
                reasoning=data.get("reasoning", ""),
                dissent=data.get("dissent", ""),
                votes=data.get("votes", []),
                raw_response=text,
            )

            # Sanity check: if fewer than 6 experts (of 11) voted BET but confidence > 0.7, cap it
            if result.bet_votes < 6 and result.confidence > 0.7:
                result.confidence = result.bet_votes / 11.0

            # Cache
            self._cache[key] = (time.time(), result)

            logger.info(
                f"[CLAUDE] {event.game} {event.match_state.team_a} vs {event.match_state.team_b} | "
                f"{result.bet_votes}/10 voted BET | conf={result.confidence:.2f} | "
                f"action={result.action} | ${cost:.4f} | "
                f"calls={self.call_count} total=${self.total_cost:.3f}"
            )
            logger.info(f"[CLAUDE] Consensus: {result.reasoning}")
            if result.dissent:
                logger.info(f"[CLAUDE] Dissent: {result.dissent}")

            self.consecutive_errors = 0

            # Store for dashboard live view
            self.recent_analyses.append({
                "time": time.time(),
                "game": event.game,
                "team_a": event.match_state.team_a,
                "team_b": event.match_state.team_b,
                "event": event.event_type.value,
                "action": result.action,
                "confidence": result.confidence,
                "bet_votes": result.bet_votes,
                "reasoning": result.reasoning[:120],
                "dissent": result.dissent[:100] if result.dissent else "",
                "votes": [
                    {"expert": v.get("expert", "?"), "vote": v.get("vote", "?"), "reason": v.get("reason", "")[:60]}
                    for v in (result.votes or [])
                ],
                "edge": round(edge, 3),
                "cost": round(cost, 4),
            })
            if len(self.recent_analyses) > 10:
                self.recent_analyses = self.recent_analyses[-10:]

            return result

        except json.JSONDecodeError as e:
            self.consecutive_errors += 1
            self.last_error = f"JSON parse: {e}"
            logger.error(f"[CLAUDE] JSON parse error: {e}")
            return None
        except anthropic.APIError as e:
            self.consecutive_errors += 1
            self.last_error = str(e)[:100]
            if "insufficient" in str(e).lower() or "credit" in str(e).lower() or "balance" in str(e).lower():
                logger.error(f"[CLAUDE] *** API BALANCE DEPLETED *** — add credits at console.anthropic.com")
                self.last_error = "API BALANCE DEPLETED — add credits"
            else:
                logger.error(f"[CLAUDE] API error: {e}")
            return None
        except Exception as e:
            self.consecutive_errors += 1
            self.last_error = str(e)[:100]
            logger.error(f"[CLAUDE] Unexpected error: {e}")
            return None

    def get_stats(self) -> dict:
        return {
            "claude_calls": self.call_count,
            "claude_cost": round(self.total_cost, 4),
            "claude_cache_hits": self.cache_hits,
            "claude_errors": self.consecutive_errors,
            "claude_last_error": self.last_error if self.consecutive_errors > 0 else "",
            "claude_analyses": self.recent_analyses,
            "claude_active": True,
        }
