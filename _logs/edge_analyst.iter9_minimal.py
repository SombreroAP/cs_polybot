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

SYSTEM_PROMPT = """CS2 Polymarket trader. Output ONE line of JSON, nothing else:
{"action":"buy"|"skip","confidence":0-1,"reason":"short"}

Default is SKIP. BUY only when the setup is simple and strong.

HARD GATES (outside these → SKIP):
- ask between 25¢ and 70¢
- spread ≤ 10¢

BUY requires BOTH:
1. Buy team is LEADING or TIED in maps (never TRAILING in maps).
2. At least ONE clear current advantage: just won a round OR economy lead OR more players alive.

TAIL-RISK: if ask > 70¢, SKIP unless a map just ended in buy team's favor."""


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
                    # NOTE: first request to a fresh model triggers an Ollama
                    # cold-load that can take 30-90s for 24-70B models. Use a
                    # generous total timeout but short connect timeout. Once
                    # warm, subsequent calls return in 1-5s regardless.
                    async with self._http_session.post(
                        f"{self._ollama_url}/api/chat",
                        json=payload,
                        timeout=aiohttp.ClientTimeout(
                            total=180 if _use_think else 120,
                            connect=10,
                        ),
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
        """Minimal prompt (iter9). <200 tokens. Only the decision-critical fields.

        Philosophy: LLMs with small active-param counts (qwen3 MoE = 3B) can't
        attend to 2000-token prompts well — signal gets diluted. We strip to
        the 4 things that actually predict a winning trade:
          (1) map W-L context  (2) current round advantage
          (3) ask + spread      (4) the single latest trigger event

        All the iter8 extras (HP arrays, AWP, bomb, momentum windows, 10 recent
        events, map CT-bias) are gone — shootout proved models that tried to
        use every signal lost money (gemma4 -$937, qwen2.5:14b -$102).
        """
        ta = game.get('team_a', 'A')
        tb = game.get('team_b', 'B')
        buy = game.get('buy_team', '?')
        other = tb if buy == ta else ta

        lines = [f"Evaluating BUY on {buy} (vs {other})."]

        # (1) MAP CONTEXT — the #1 signal
        sa, sb = game.get("score_a", 0), game.get("score_b", 0)
        ra, rb = game.get("round_a", 0), game.get("round_b", 0)
        if buy == ta:
            buy_maps, opp_maps = sa, sb
            buy_rounds, opp_rounds = ra, rb
        else:
            buy_maps, opp_maps = sb, sa
            buy_rounds, opp_rounds = rb, ra
        if buy_maps > opp_maps:
            status = f"LEADING in maps {buy_maps}-{opp_maps}"
        elif buy_maps < opp_maps:
            status = f"TRAILING in maps {buy_maps}-{opp_maps}"
        else:
            status = f"TIED in maps {buy_maps}-{opp_maps}"
        lines.append(f"{buy} is {status}.  Current map round score: {buy_rounds}-{opp_rounds}.")

        # (2) ONE clearest current advantage (not a full dump)
        eco_a = game.get("economy_a", 0)
        eco_b = game.get("economy_b", 0)
        eco_buy = eco_a if buy == ta else eco_b
        eco_opp = eco_b if buy == ta else eco_a
        alive_a = game.get("alive_a", 5)
        alive_b = game.get("alive_b", 5)
        alive_buy = alive_a if buy == ta else alive_b
        alive_opp = alive_b if buy == ta else alive_a

        advantages = []
        if eco_buy > eco_opp * 1.3:
            advantages.append(f"economy lead (${eco_buy:,} vs ${eco_opp:,})")
        elif eco_opp > eco_buy * 1.3:
            advantages.append(f"economy DEFICIT (${eco_buy:,} vs ${eco_opp:,})")
        if alive_buy != 5 or alive_opp != 5:
            if alive_buy > alive_opp:
                advantages.append(f"alive-count lead {alive_buy}v{alive_opp}")
            elif alive_opp > alive_buy:
                advantages.append(f"alive-count DEFICIT {alive_buy}v{alive_opp}")
        if advantages:
            lines.append("Advantage: " + "; ".join(advantages) + ".")
        else:
            lines.append("No clear mid-round advantage.")

        # (3) PRICE — compact one-liner
        bid = market.get("bid", 0)
        ask = market.get("ask", 0)
        spread = market.get("spread", 0)
        lines.append(f"Ask {ask*100:.0f}¢, bid {bid*100:.0f}¢, spread {spread*100:.0f}¢.")

        # (4) ONE trigger event (the latest, not 5-10)
        if events:
            e = events[-1]
            desc = (e.get('description') or '')[:70]
            team = e.get('team', '')
            if team:
                lines.append(f"Trigger: {desc} (team: {team}).")
            else:
                lines.append(f"Trigger: {desc}.")

        # Decision cue at the END (LLMs weight recency)
        lines.append("")
        lines.append(f"Should we BUY {buy} now? JSON only.")

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
