"""
Lightweight Claude analyst for edge trading decisions.

Fast, cheap, focused — not the 11-expert panel.
Uses Haiku for <1s response at $0.0003/call.
Analyzes a window of recent events + game state to decide: BUY or SKIP.
"""
import json
import time
import hashlib
import logging
from typing import Optional
from dataclasses import dataclass

import anthropic

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """Autonomous CS2 arb trader on Polymarket. Output JSON only:
{"action":"buy"|"skip","confidence":0-1,"bet_size":5-200,"tp_pct":0.01-0.30,"sl_pct":0.05-0.40,"reason":"brief"}

YOUR JOB: find BUY opportunities. Default to BUY if game state supports and price is in sweet zone. Only SKIP for clear red flags.

BUY when ask is 20¢-75¢ AND ANY of:
- economy advantage (buy side has more money)
- kill lead or recent round win
- alive-count advantage mid-round (e.g. 4v2, 5v3) — buy side has more players alive
- HP/damage advantage (buy side dealt more damage this round)
- buy side has AWP and opponent doesn't
- buy side has full_buy / half_buy vs opponent eco / pistol
- favorable map-side (CT on defense-favored maps like mirage/inferno)
- positive 10s/30s bid momentum (+1¢ or more)
- open_kill event just fired for buy side (first frag of round → 60-70% round win)
- clutch_win event for buy side

SKIP only for:
- ask >75¢ (asymmetric ceiling risk — BLOCKED)
- ask=0¢ OR bid=0¢ (untradable)
- staleness >300s (stale data) — staleness ≤180s is FRESH, do NOT skip for 6s/30s/60s/100s — only skip at 300s+
- buy-side economy disadvantage AND trailing score AND negative momentum (ALL three)

BET SIZING — stake a percentage of CURRENT BALANCE (shown in prompt):
- Default stake = 5% of balance (e.g. balance $1000 → ~$50)
- High-conviction setup (economy + kill lead + momentum all agree): stake 8% (~$80 of $1000)
- Dead-zone ask 0.40-0.55: half stake (2.5%, ~$25 of $1000)
- Tight ceiling ask 0.70-0.75: 3% max (~$30 of $1000)
- HARD MINIMUM $30 — anything smaller won't matter after fees
- HARD MAXIMUM 10% of balance — never risk more than that per trade
- Ignore your own confidence — high conf is anti-signal in this session

TP/SL — keep them tight:
- TP: 5-10% typical (aim for quick exits, not home runs)
- SL: 5-8% max (losses have been too large)
- Series market: TP 5-8%, SL 5%
- Map winner: TP 8-12%, SL 7%

Quote specific numbers (¢, %, round scores, momentum) in reason. Skip adds no bet.

CRITICAL DATA HONESTY RULES (violations auto-VETO your BUY):
- NEVER mention "Kambi" or any external bookmaker — not in the prompt, not available. Any mention = hallucination.
- When citing economy, write the team you're buying FIRST with its value, then the opponent, using the actual team name (e.g. "FaZe $18,000 vs NaVi $4,200"). NEVER write the literal string "BUY_TEAM" — substitute the real team name. If the buy side has LESS money than the opponent, that is a DISADVANTAGE, never call it an "advantage".
- Do not invent numbers. Only cite values shown above (eco $, kills, rounds, ¢, momentum).
- If buy side has less economy AND fewer kills AND is trailing → SKIP, do not rationalize.

MOMENTUM SANITY (critical — we burn money on noise):
- If 10s momentum shows ±15¢ or more, OR 60s momentum shows ±25¢ or more, that is almost certainly a SINGLE SMALL-SIZE PRINT on an illiquid token, NOT real momentum. Treat as NOISE, not signal.
- A momentum value of exactly 0.0¢ is NOT "positive momentum" — it is flat. Do not label it positive.
- Real momentum is gradual: +1¢ to +5¢ over 30-60s with multiple trades. Spikes > 10¢ in 10s are artifacts.
- When you see a spike-looking momentum, explicitly acknowledge it as noise in your reason and lean toward SKIP."""


@dataclass
class EdgeDecision:
    action: str  # "buy" or "skip"
    confidence: float  # 0.0-1.0
    reason: str
    bet_size: float = 0.0   # USD stake — qwen decides per trade
    tp_pct: float = 0.0     # take-profit as fraction (e.g. 0.05 = 5%)
    sl_pct: float = 0.0     # stop-loss as fraction


class EdgeAnalyst:
    """Fast edge analysis — uses local Ollama (RTX 5090) or Claude API fallback."""

    def __init__(self, db=None):
        # Local LLM via Ollama (fast, free, ~300ms)
        self._ollama_url = getattr(config, 'OLLAMA_URL', None)
        self._ollama_model = getattr(config, 'OLLAMA_MODEL', 'qwen2.5-coder:7b')
        self._use_local = bool(self._ollama_url)
        # Global concurrency limiter — Ollama serves requests SEQUENTIALLY (single GPU).
        # With concurrent async callers, requests queue on the server side invisibly and each
        # call sees huge latency (30-70s). A semaphore of 1 serializes at OUR side so the
        # server never queues more than one. Pending callers wait in asyncio land (fast to cancel)
        # instead of blocking in HTTP requests.
        # Lazy per-loop semaphore. Creating at __init__ binds to whatever loop
        # happens to exist at import time (often None or a one-off loop) which
        # causes "attached to a different loop" when called from real tasks.
        self._qwen_sem = None
        self._qwen_sem_loop = None
        # Shared aiohttp session — avoids opening a new TCP connection per call.
        self._http_session = None

        # Claude API fallback
        self._client = anthropic.AsyncAnthropic(
            api_key=config.ANTHROPIC_API_KEY
        )
        self._model = getattr(config, 'EDGE_CLAUDE_MODEL', 'claude-haiku-3-5')
        self._cache: dict[str, tuple[float, EdgeDecision]] = {}
        self._cache_ttl = 10.0 if self._use_local else 30.0  # shorter cache for fast local LLM
        self._match_calls: dict[str, int] = {}
        self._max_calls_per_match = getattr(config, 'EDGE_CLAUDE_MAX_CALLS', 10)
        self._db = db

        if self._use_local:
            logger.info(f"[EDGE-ANALYST] Using LOCAL LLM: {self._ollama_model} @ {self._ollama_url}")
        else:
            logger.info(f"[EDGE-ANALYST] Using Claude API: {self._model}")

        # Stats
        self.call_count = 0
        self.total_cost = 0.0
        self.cache_hits = 0
        self._model_claude = getattr(config, 'EDGE_CLAUDE_MODEL', 'claude-haiku-4-5-20251001')
        self.last_decisions: list[dict] = []  # for dashboard (recent 50)
        self.buy_decisions: list[dict] = []   # ALL buy decisions (never pruned)

    async def should_buy(self, events_window: list[dict], game_state: dict,
                         market_state: dict) -> Optional[EdgeDecision]:
        """Analyze recent events and decide: BUY or SKIP.

        Args:
            events_window: list of recent events [{type, team, description, timestamp}, ...]
            game_state: {game, team_a, team_b, score_a, score_b, round_a, round_b,
                        kills_a, kills_b, gold_lead, economy_a, economy_b, ...}
            market_state: {price, bid, ask, spread, volume, market_type, question}
        """
        if not config.ANTHROPIC_API_KEY:
            return None

        # Cache check
        cache_key = self._make_cache_key(game_state, events_window)
        cached = self._cache.get(cache_key)
        if cached and time.time() - cached[0] < self._cache_ttl:
            self.cache_hits += 1
            return cached[1]

        # Per-match call limit
        match_key = f"{game_state.get('team_a', '')}_{game_state.get('team_b', '')}"
        calls = self._match_calls.get(match_key, 0)
        if calls >= self._max_calls_per_match:
            return None

        # Build prompt
        prompt = self._build_prompt(events_window, game_state, market_state)

        try:
            self._match_calls[match_key] = calls + 1
            self.call_count += 1

            if self._use_local:
                # ─── LOCAL LLM (Ollama on RTX 5090) ──────────────
                # Using /api/chat + think=True for qwen3:30b-a3b-think.
                # Thinking cannot co-exist with format=json, so we parse JSON out of
                # the assistant content ourselves (regex picks first object w/ "action").
                import aiohttp
                _use_think = "think" in (self._ollama_model or "").lower()
                payload = {
                    "model": self._ollama_model,
                    "stream": False,
                    "keep_alive": -1,      # keep model loaded in VRAM indefinitely
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    "options": {
                        "temperature": 0,
                        "num_predict": 2500 if _use_think else 250,
                    },
                }
                if _use_think:
                    payload["think"] = True
                else:
                    payload["think"] = False
                    payload["format"] = "json"
                    payload["options"]["stop"] = ["<think>", "</think>", "<start_of_turn>", "<end_of_turn>"]
                t0 = time.time()
                # SERIALIZE: qwen can't process requests in parallel on one GPU.
                # The semaphore ensures only one in flight at a time from our side,
                # avoiding invisible server-side queue that caused 60s+ latencies.
                import asyncio as _aio
                cur_loop = _aio.get_event_loop()
                # Rebuild semaphore if loop changed
                if self._qwen_sem is None or self._qwen_sem_loop is not cur_loop:
                    self._qwen_sem = _aio.Semaphore(1)
                    self._qwen_sem_loop = cur_loop
                # BACKPRESSURE: if too many callers already waiting, drop this one
                # instead of letting the queue grow to minutes. Stale decisions on
                # stale context aren't useful — we'd rather skip and wait for a
                # fresher event on this match.
                # Semaphore._waiters is a private deque; bool-check is O(1).
                # Strict: if ANY caller is waiting AND semaphore is locked, drop.
                # 1 in flight + 0 waiting is the only allowed state. Aggressive
                # but necessary — qwen does ~1-2s per call and events fire
                # faster than that across 30+ matches.
                pending = len(getattr(self._qwen_sem, "_waiters", []) or [])
                locked = self._qwen_sem.locked()
                if locked and pending >= 1:
                    self._match_calls[match_key] = calls  # refund the count
                    logger.info(f"[LOCAL-LLM] DROP — queue busy ({pending} pending) — skipping {match_key[:40]}")
                    return None
                async with self._qwen_sem:
                    # Reuse a single aiohttp session across calls on the SAME loop.
                    # If the loop changed (task spawned from a different runner),
                    # recreate — a cached session bound to a dead loop raises
                    # "Future attached to a different loop".
                    sess_loop = getattr(self._http_session, "_loop", None) if self._http_session else None
                    if (self._http_session is None
                            or self._http_session.closed
                            or sess_loop is not cur_loop):
                        try:
                            if self._http_session and not self._http_session.closed:
                                await self._http_session.close()
                        except Exception:
                            pass
                        self._http_session = aiohttp.ClientSession()
                    t_wire = time.time()
                    async with self._http_session.post(
                        f"{self._ollama_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=60 if _use_think else 20),
                    ) as resp:
                        data = await resp.json()
                import re as _re
                _raw_content = data.get("message", {}).get("content", "") or data.get("response", "")
                # Strip any <think>...</think> block that leaked into content
                if "</think>" in _raw_content:
                    _raw_content = _raw_content.split("</think>", 1)[1]
                response_text = _re.sub(r'<[^>]+>', '', _raw_content).strip()
                input_tokens = data.get("prompt_eval_count", 0)
                output_tokens = data.get("eval_count", 0)
                cost = 0.0
                latency_ms = (time.time() - t0) * 1000
                wire_ms = (time.time() - t_wire) * 1000
                wait_ms = latency_ms - wire_ms
                tok_per_sec = (output_tokens * 1000.0 / wire_ms) if wire_ms > 0 else 0
                logger.info(
                    f"[LOCAL-LLM] {self._ollama_model} {latency_ms:.0f}ms "
                    f"(wait={wait_ms:.0f}ms infer={wire_ms:.0f}ms | "
                    f"{output_tokens}tok @ {tok_per_sec:.0f}tok/s)"
                )
            else:
                # ─── CLAUDE API (fallback) ────────────────────────
                for attempt in range(2):
                    try:
                        response = await self._client.messages.create(
                            model=self._model,
                            max_tokens=250,
                            temperature=0,
                            system=SYSTEM_PROMPT,
                            messages=[{"role": "user", "content": prompt}],
                        )
                        break
                    except (anthropic.APIConnectionError, anthropic.APITimeoutError) as e:
                        if attempt == 0:
                            logger.warning(f"[CLAUDE-EDGE] Retry after: {e}")
                            continue
                        raise
                response_text = response.content[0].text
                input_tokens = response.usage.input_tokens
                output_tokens = response.usage.output_tokens
                cost = (input_tokens * 0.25 + output_tokens * 1.25) / 1_000_000
            self.total_cost += cost

            # Parse response — extract first valid JSON object
            text = response_text.strip()

            # If local model returned empty or no JSON, fall back to Claude API
            if self._use_local and (not text or "{" not in text):
                try:
                    response = await self._client.messages.create(
                        model=self._model_claude, max_tokens=150, temperature=0,
                        system=SYSTEM_PROMPT, messages=[{"role": "user", "content": prompt}],
                    )
                    text = response.content[0].text.strip()
                    cost = (response.usage.input_tokens * 0.25 + response.usage.output_tokens * 1.25) / 1_000_000
                    self.total_cost += cost
                except Exception:
                    return None
            # Strip markdown code blocks
            if "```" in text:
                parts = text.split("```")
                for part in parts:
                    part = part.strip()
                    if part.startswith("json"):
                        part = part[4:].strip()
                    if part.startswith("{"):
                        text = part
                        break
            # Find first complete JSON object
            if "{" in text:
                start = text.index("{")
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{": depth += 1
                    elif text[i] == "}": depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break
            text = text.strip()

            # Try parsing, if truncated try adding closing brace
            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                # Truncated JSON — try closing it
                if text.startswith("{") and "}" not in text:
                    text = text.rsplit(",", 1)[0] + "}"  # drop last incomplete field
                try:
                    data = json.loads(text)
                except json.JSONDecodeError:
                    logger.warning(f"[CLAUDE-EDGE] JSON parse error: {text[:60]}")
                    return None
            # Handle field name variants
            reason = data.get("reason", "") or data.get("reasoning", "") or data.get("rationale", "")
            action = (data.get("action", "") or data.get("buy_decision", "") or "skip").lower()

            def _clamp(v, lo, hi, default):
                try:
                    x = float(v)
                    if x != x:  # NaN
                        return default
                    return max(lo, min(hi, x))
                except (TypeError, ValueError):
                    return default

            decision = EdgeDecision(
                action=action,
                confidence=_clamp(data.get("confidence", 0.5), 0.0, 1.0, 0.5),
                reason=reason,
                # Qwen owns these — clamp to sane ranges to prevent a runaway output
                # (bet_size>$200 or tp_pct>50%) from vaporizing the balance.
                bet_size=_clamp(data.get("bet_size", 0), 5.0, 200.0, 0.0) if action == "buy" else 0.0,
                tp_pct=_clamp(data.get("tp_pct", 0), 0.01, 0.30, 0.0) if action == "buy" else 0.0,
                sl_pct=_clamp(data.get("sl_pct", 0), 0.05, 0.40, 0.0) if action == "buy" else 0.0,
            )
            # AI-driven mode: VETO guard removed. Qwen sees momentum directly and decides.

            # Cache
            self._cache[cache_key] = (time.time(), decision)

            # Log
            team = game_state.get("buy_team", "?")
            logger.info(
                f"[CLAUDE-EDGE] {decision.action.upper()} {team} | "
                f"conf={decision.confidence:.2f} | ${cost:.4f} | "
                f"{decision.reason[:60]}"
            )

            # Store for dashboard
            entry = {
                "action": decision.action,
                "confidence": decision.confidence,
                "reason": decision.reason[:200],
                "team": team,
                "game": game_state.get("game", "?"),
                "time": time.time(),
                "cost": cost,
                "events": len(events_window),
                "model": self._ollama_model if self._use_local else self._model_claude,
            }
            self.last_decisions.append(entry)
            if len(self.last_decisions) > 50:
                self.last_decisions = self.last_decisions[-50:]
            # Keep ALL buy decisions so they never get pushed off dashboard
            if decision.action == "buy":
                self.buy_decisions.append(entry)

            # Persist EVERY decision to DB for historical analysis
            if self._db:
                try:
                    gs = game_state or {}
                    ms = market_state or {}
                    self._db.save_claude_decision({
                        "timestamp": time.time(),
                        "action": decision.action,
                        "confidence": decision.confidence,
                        "reason": decision.reason,
                        "team": team,
                        "game": gs.get("game", "?"),
                        "match_id": gs.get("match_id", ""),
                        "market_type": ms.get("market_type", ""),
                        "market_price": ms.get("price", 0),
                        "buy_price": ms.get("ask", 0),
                        "spread": ms.get("spread", 0),
                        "gold_lead": gs.get("gold_lead", 0),
                        "kills_a": gs.get("kills_a", 0),
                        "kills_b": gs.get("kills_b", 0),
                        "game_minutes": gs.get("game_minutes", 0),
                        "events_analyzed": len(events_window),
                        "event_window": events_window[-6:],
                        "game_state": gs,
                        "market_state": ms,
                        "cost": cost,
                        "trade_opened": decision.action == "buy",
                    })
                except Exception as e:
                    logger.error(f"Failed to persist decision: {e}")

            return decision

        except json.JSONDecodeError as e:
            logger.warning(f"[CLAUDE-EDGE] JSON parse error: {e}")
            return None
        except Exception as e:
            logger.warning(f"[CLAUDE-EDGE] Error: {type(e).__name__}: {e}", exc_info=True)
            return None

    def _build_prompt(self, events: list[dict], game: dict, market: dict) -> str:
        """Build a minimal prompt for fast Claude analysis."""
        lines = []
        ta = game.get('team_a', 'A')
        tb = game.get('team_b', 'B')
        buy = game.get('buy_team', '?')

        # ─── MATCH STATE (CS2 only — Dota2 disabled) ──────────────────
        lines.append(f"GAME: CS2 | {ta} vs {tb}  (evaluating BUY on {buy})")
        sa, sb = game.get("score_a", 0), game.get("score_b", 0)
        ra, rb = game.get("round_a", 0), game.get("round_b", 0)
        if ra + rb > 0:
            lines.append(f"Series: {sa}-{sb}  Map: {ra}-{rb}")
        elif sa + sb > 0:
            lines.append(f"Series: {sa}-{sb}")

        # Map + side advantage (CT-leaning maps matter)
        map_name = game.get("map_name", "")
        side = game.get("side", "")
        if map_name:
            CT_BIAS = {"de_nuke": "CT 57%", "de_train": "CT 55%", "de_overpass": "CT 53%", "de_vertigo": "CT 53%", "de_ancient": "CT 52%", "de_dust2": "T 52%", "de_mirage": "balanced", "de_inferno": "balanced", "de_anubis": "balanced"}
            clean = map_name.lower().replace(" ", "_")
            bias = CT_BIAS.get(clean, "balanced")
            side_txt = f", {ta} on {side.upper()}" if side else ""
            lines.append(f"Map: {map_name} ({bias}{side_txt})")

        # Economy (huge for CS2)
        eco_a = game.get("economy_a", 0)
        eco_b = game.get("economy_b", 0)
        if eco_a or eco_b:
            eco_diff = eco_a - eco_b
            who = ta if eco_diff > 0 else tb if eco_diff < 0 else "tied"
            lines.append(f"Economy: {ta}=${eco_a:,} vs {tb}=${eco_b:,}  ({who} +${abs(eco_diff):,})")

        # Round-in-progress kills
        rk_a = game.get("round_kills_a", 0)
        rk_b = game.get("round_kills_b", 0)
        if rk_a or rk_b:
            lines.append(f"Round kills so far: {ta} {rk_a} - {rk_b} {tb}")

        # ─── LIVE ROUND STATE (bo3.gg 1s poll — deeper data than score alone) ──
        alive_a = game.get("alive_a", 5)
        alive_b = game.get("alive_b", 5)
        avg_hp_a = game.get("avg_hp_a", 100)
        avg_hp_b = game.get("avg_hp_b", 100)
        dmg_a = game.get("damage_a", 0)
        dmg_b = game.get("damage_b", 0)
        # Alive count — if not 5v5, the round is mid-fight
        if alive_a != 5 or alive_b != 5:
            who_ahead = ta if alive_a > alive_b else tb if alive_b > alive_a else "even"
            lines.append(f"Alive: {alive_a}v{alive_b}  (avg HP {ta}={avg_hp_a} {tb}={avg_hp_b}) — {who_ahead} ahead")
        # Damage dealt this round
        if dmg_a or dmg_b:
            lines.append(f"Damage this round: {ta}→{dmg_b}hp dealt vs {tb}→{dmg_a}hp dealt")
        # Buy-type classification (pistol/eco/force/half/full)
        bta = game.get("buy_type_a", "")
        btb = game.get("buy_type_b", "")
        if bta or btb:
            lines.append(f"Round buys: {ta}={bta or '?'} | {tb}={btb or '?'}")
        # AWP holding + bomb state
        awp_a = game.get("has_awp_a", False)
        awp_b = game.get("has_awp_b", False)
        if awp_a or awp_b:
            awp_line = []
            if awp_a: awp_line.append(f"{ta} has AWP")
            if awp_b: awp_line.append(f"{tb} has AWP")
            lines.append("AWP: " + ", ".join(awp_line))
        bomb = game.get("bomb_carrier", "")
        if bomb == "a":
            lines.append(f"Bomb: {ta} carries (T-side setup)")
        elif bomb == "b":
            lines.append(f"Bomb: {tb} carries (T-side setup)")
        if game.get("has_defuse_kit_a"):
            lines.append(f"{ta} has defuse kit (CT)")

        # ─── BANKROLL CONTEXT ──────────────────
        bal = market.get("balance", 0)
        openp = market.get("open_positions", 0)
        if bal > 0:
            lines.append(f"BANKROLL: ${bal:,.0f}  open_positions={openp}  (5% stake=${bal*0.05:.0f}, 10% cap=${bal*0.10:.0f})")

        # ─── POLYMARKET ORDERBOOK ──────────────────
        bid = market.get("bid", 0)
        ask = market.get("ask", 0)
        spread = market.get("spread", 0)
        price = market.get("price", 0)
        vol = market.get("volume", 0)
        liq = market.get("liquidity", 0)
        mtype = market.get("market_type", "series")
        lines.append(f"PM {mtype}: bid={bid*100:.0f}¢ ask={ask*100:.0f}¢ mid={price*100:.0f}¢ spr={spread*100:.0f}% vol=${vol:,.0f} liq=${liq:,.0f}")

        # Multi-window momentum (10s / 30s / 60s)
        m10 = market.get("momentum_10", {}).get("bid_change", 0) if isinstance(market.get("momentum_10"), dict) else 0
        m30 = market.get("momentum_30", {}).get("bid_change", 0) if isinstance(market.get("momentum_30"), dict) else 0
        m60 = market.get("momentum_60", {}).get("bid_change", 0) if isinstance(market.get("momentum_60"), dict) else 0
        if m10 or m30 or m60:
            lines.append(f"Bid momentum: 10s={m10*100:+.1f}¢  30s={m30*100:+.1f}¢  60s={m60*100:+.1f}¢")
        staleness = market.get("staleness_seconds", 999)
        if staleness > 5:
            lines.append(f"⚠️ Orderbook stale: {staleness:.0f}s old")

        # ─── KAMBI CROSS-MARKET REFERENCE ──────────
        # Kambi removed from prompt — backtest showed qwen mentions it in 52 trades
        # but misreads it 43× (WR 46%, avg -$3.23). Only "aligned" correctly 9×.
        # Kambi is now used as a silent pre-filter in latency.py instead.

        # Risk warnings from filters
        warnings = game.get("warnings", [])
        if warnings:
            lines.append("")
            lines.append("⚠️ RISK WARNINGS:")
            for w in warnings:
                lines.append(f"  - {w}")

        # (CS2 side + round kills already emitted above in the main block)

        # Events — 5 max for speed
        if events:
            lines.append(f"Events({len(events)}):")
            for e in events[-5:]:
                lines.append(f"- {e.get('description', '')[:50]}")

        # Force JSON output — this MUST be the last thing the model sees
        buy_team = game.get("buy_team", game.get("team_a", "?"))
        lines.append("")
        lines.append(f"BUY {buy_team}? Reply with ONLY this JSON, nothing else:")
        lines.append('{"action":"buy or skip","confidence":0.0-1.0,"reason":"one sentence"}')

        return "\n".join(lines)

    async def should_exit(self, position_info: dict, game_state: dict,
                          market_state: dict, recent_events: list) -> Optional[dict]:
        """Ask qwen whether to cut a losing position early. Drop-don't-queue.

        Returns dict {action: 'cut'|'hold', confidence, reason} or None if skipped.
        Used when a position crosses half-way to SL — we ask qwen before the hard
        SL fires in case the thesis is still intact.
        """
        if not self._use_local:
            return None
        import asyncio as _aio
        import aiohttp
        cur_loop = _aio.get_event_loop()
        if self._qwen_sem is None or self._qwen_sem_loop is not cur_loop:
            self._qwen_sem = _aio.Semaphore(1)
            self._qwen_sem_loop = cur_loop
        # DROP if busy — never queue exit checks.
        if self._qwen_sem.locked():
            return None

        # Build lean prompt
        lines = [
            f"OPEN POSITION: {position_info.get('team')} @ {position_info.get('fill_price'):.3f}",
            f"CURRENT BID: {market_state.get('bid',0):.3f} | ASK: {market_state.get('ask',0):.3f} | SPREAD: {market_state.get('spread',0):.3f}",
            f"P&L: {position_info.get('pnl_pct',0):+.1f}% | AGE: {position_info.get('age_s',0):.0f}s | SL_TARGET: {position_info.get('sl_target_pct',0)*100:.1f}%",
            f"GAME: {game_state.get('game')} {game_state.get('team_a')} vs {game_state.get('team_b')} | series {game_state.get('score_a',0)}-{game_state.get('score_b',0)} | round {game_state.get('round_a',0)}-{game_state.get('round_b',0)} | side={game_state.get('side','?')}",
            f"ECO: {game_state.get('team_a')} ${game_state.get('economy_a',0):,} vs {game_state.get('team_b')} ${game_state.get('economy_b',0):,}",
            f"KILLS this round: {game_state.get('round_kills_a',0)}-{game_state.get('round_kills_b',0)}",
            "",
            "RECENT EVENTS (last 3):",
        ]
        for e in (recent_events or [])[-3:]:
            lines.append(f"- {e.get('type','')}: {(e.get('description') or e.get('desc') or '')[:70]}")
        lines.append("")
        lines.append("Decide: should we CUT this position now, or HOLD for recovery?")
        lines.append('Reply ONLY JSON: {"action":"cut"|"hold","confidence":0.0-1.0,"reason":"one sentence"}')
        user_prompt = "\n".join(lines)

        sys_prompt = (
            "You are deciding whether to cut a losing esports trade early. "
            "CUT if: thesis is broken (opponent pulled ahead, momentum reversed, map nearly lost). "
            "HOLD if: thesis still valid, temporary bid dip, or near enough to TP. "
            "Reply JSON only."
        )

        payload = {
            "model": self._ollama_model,
            "stream": False,
            "keep_alive": -1,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0, "num_predict": 200},
            "format": "json",
            "think": False,
        }

        async with self._qwen_sem:
            try:
                if self._http_session is None or self._http_session.closed:
                    self._http_session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=5))
                async with self._http_session.post(
                    f"{self._ollama_url}/api/chat", json=payload) as resp:
                    if resp.status != 200:
                        return None
                    data = await resp.json()
            except Exception as e:
                logger.debug(f"[EXIT-LLM] error: {e}")
                return None

        try:
            text = (data.get("message") or {}).get("content", "") or ""
            if "{" in text:
                start = text.index("{")
                depth = 0
                for i in range(start, len(text)):
                    if text[i] == "{": depth += 1
                    elif text[i] == "}": depth -= 1
                    if depth == 0:
                        text = text[start:i+1]
                        break
            parsed = json.loads(text.strip())
            action = (parsed.get("action") or "hold").lower()
            if action not in ("cut", "hold"):
                action = "hold"
            return {
                "action": action,
                "confidence": float(parsed.get("confidence", 0.5) or 0.5),
                "reason": (parsed.get("reason") or "")[:200],
            }
        except Exception:
            return None

    def _make_cache_key(self, game: dict, events: list) -> str:
        """Create cache key from game state + event count."""
        key_data = f"{game.get('team_a', '')}_{game.get('score_a', 0)}_{game.get('round_a', 0)}_{len(events)}"
        return hashlib.md5(key_data.encode()).hexdigest()

    def get_stats(self) -> dict:
        # Merge: all BUY decisions + last 30 recent, deduplicated, sorted by time
        all_entries = {id(d): d for d in self.buy_decisions}
        for d in self.last_decisions[-30:]:
            all_entries[id(d)] = d
        merged = sorted(all_entries.values(), key=lambda x: x.get("time", 0))
        return {
            "edge_claude_calls": self.call_count,
            "edge_claude_cost": round(self.total_cost, 4),
            "edge_claude_cache_hits": self.cache_hits,
            "edge_claude_decisions": merged[-50:],
        }
