# CS2 Latency Arbitrage Bot — Master Plan

**Goal:** A fully-automated bot that bets on Counter-Strike 2 map-winner (and occasionally series) markets on Polymarket, exploiting the 2-5 second latency between our real-time game-event feed and Polymarket's orderbook reprice.

**Constraints locked by user (2026-04-17):**

| Item | Value |
|---|---|
| Starting bankroll | $1,000 |
| Target return | +10-20% per month |
| Kill-switch | -50% drawdown ($500) |
| Markets | CS2 map winner (primary), series (secondary) |
| Liquidity floor | $1,000+ orderbook volume |
| Tournament scope | All CS2, tier-1 prioritized for volume |
| Execution mode | Fully automated (no human in the loop) |
| Infrastructure | Must run 24/7 (Mac can shut down → needs VPS) |

---

## 1. The edge (why this works)

The thesis: **our game feed sees events before Polymarket repositions its orderbook.**

- bo3.gg WebSocket delivers round-end + kill events in <1s
- Polymarket's market makers and retail see the same info on a stream delay (Twitch/HLTV dashboards lag 3-8s). Our edge window is ~2-5s.
- During that window the orderbook is "stale" — best ask hasn't moved yet. We buy the winning side before the repricing.
- Exit ~30-90s later when the market has caught up and the ask has moved 7-15% in our favor.

This only works if:
1. We detect the event fast enough
2. We decide and execute fast enough (target: <2s from event to order)
3. The market is thin enough that our fill doesn't move it out of our favor
4. The event is *causal* (will move the price) not noise (will mean-revert)

Point 4 is where the LLM earns its keep — deciding whether a given "round_end for team A" is a real edge signal given map score, economy, side, map bias, etc.

---

## 2. Strategy

### Market selection
- **Map winner** — primary. Shorter duration (~15-30 min/map), bigger moves per round, better liquidity during live play.
- **Series winner** — secondary. Only when a map just ended in a one-sided fashion (e.g. 16-4 → clear momentum into map 2).
- Skip: any market with <$1,000 volume on the live book.

### Decision pipeline
```
Game event  →  Trigger detector  →  Cheap gates  →  LLM decision  →  Execute  →  Monitor exit
   (~1s)         (python, <1ms)    (python, <5ms)   (<1s)           (<500ms)    (every 1s)
```

**Triggers** (fire an evaluation):
| Trigger | Description | Strength |
|---|---|---|
| `round_end` | Score changed in current map | ★★ |
| `map_win` | `game_ended=true` | ★★★ |
| `clutch_win` | 1vN where 1 won | ★★ |
| `kill_streak` | 3+ kills in <20s by same team | ★ |
| `open_kill` | First kill of round | ★ |

**Gate filters** (reject before LLM):
- Market liquidity ≥ $1,000 volume
- Spread ≤ 10¢
- Ask between 25¢ and 70¢ (outside this, edge doesn't compensate for fees + risk)
- Buy team NOT trailing in maps (0-1, 0-2, 1-2) unless map just ended in their favor
- No existing open position on same team+market
- Balance > $200 (never trade when nearly broke)
- Daily loss < 10% of bankroll
- Session has had fewer than 3 losses in a row (else cooldown)

Only ~5% of trigger events pass the gates — they're the ones that get an LLM call. Cuts LLM cost by 95%.

### Position management (code-owned, not LLM-owned)

LLM decides *whether* to enter and with what confidence. Sizing and exits are pure Python rules — deterministic, auditable, not at the mercy of a prompt change:

```python
PER_TRADE_BASE_PCT    = 0.04       # 4% of balance = $40 at $1000
PER_TRADE_STRONG_PCT  = 0.06       # 6% for high-confidence trades
PER_TRADE_MAX_PCT     = 0.08       # hard cap
PER_TRADE_MIN_USD     = 5.0        # Polymarket min + fee tolerance
CONFIDENCE_FLOOR      = 0.70       # skip if LLM conf < 0.70
TAIL_RISK_ASK_MAX     = 0.70       # cap bet at $25 if ask > 70¢
MAX_CONCURRENT        = 3          # no more than 3 open positions
COOLDOWN_N_LOSSES     = 3          # pause 30 min after 3 consecutive losses
DAILY_LOSS_STOP       = 0.10       # stop trading for day at -10% bankroll
HARD_STOP             = 0.50       # disable bot at -50% bankroll (permanent)

TAKE_PROFIT_PCT       = 0.10       # +10% on fill price
STOP_LOSS_PCT         = 0.15       # -15% on fill price
POSITION_TIMEOUT_S    = 600        # force-exit at 10 min regardless
```

**Exit logic (in priority order):**
1. TP hit → exit immediately at best bid
2. SL hit → exit immediately at best bid
3. Timeout (10 min) → exit at best bid
4. *Thesis broken* (opponent wins the map we bet on) → exit immediately
5. *Circuit breaker* (spread widens > 20¢) → exit immediately

### Bankroll rules (hard-coded)
- **Daily reset**: at 00:00 UTC, reset the daily-loss counter
- **Hard stop**: if total balance falls below $500 (−50% from start), the bot writes a `HARD_STOP.lock` file and refuses to start until removed manually
- **Soft stop**: if balance falls below $800 (−20%), cut all per-trade percentages in half automatically until balance recovers

---

## 3. Model choice — Claude API is production, pick the BEST tier

**User decision (2026-04-17):** optimize for decision quality, not API cost. If Opus makes more money than Haiku even after the ~20× higher API cost, we use Opus and increase stake to offset the overhead.

### Why cloud API beats local

After a 13-model local shootout on the same 32-file corpus:

| Pattern | Models | Result |
|---|---|---|
| Reasoning, selective | deepseek-r1:14b | **+$27.41 on 8 trades, 100% win rate** |
| Moderate + selective | mistral-small:24b | +$13.02 on 37 trades, 65% WR |
| Simple, floor-bets | llama3.1:8b | +$1.06 on 14 trades, 57% WR |
| Too cautious (skip all) | qwen2.5:32b, qwen2.5-coder:32b, esports-qwen3, qwen2.5:32b-instruct | $0, 0 trades |
| Overtrade | qwen2.5:14b, phi4:14b | -$100 to -$200 |
| Catastrophic | gemma4:8b | -$937 (blew up) |

The best *local* model made +$27.41 on 32 files. That's 0.85% per match. For a production bot we want the best possible decision quality — the API cost is a rounding error compared to the bet-size delta a smarter decision enables.

### Candidates to benchmark (in order)

| Model | Input $/MTok | Output $/MTok | Cost per decision (500 in + 50 out) | Notes |
|---|---|---|---|---|
| `claude-haiku-4-5-20251001` | $0.80 | $4.00 | ~$0.0006 | Baseline — fastest, cheapest |
| `claude-sonnet-4-5-20251001` | $3.00 | $15.00 | ~$0.0023 | Expected 2-5× better reasoning |
| `claude-opus-4-7-20260115` | $15.00 | $75.00 | ~$0.0113 | Best reasoning — 1M ctx (we don't need it, but it's the smartest) |

**At 500 decisions/day** (post-gate filters):
- Haiku: $0.30/day = **$9/month**
- Sonnet: $1.15/day = **$35/month**
- Opus: $5.65/day = **$170/month**

**Breakeven analysis:** if Opus beats Haiku by more than $160/month in PnL, it's a net win. Given our iter8 best-case was +$27 on 32 matches (~1 day of live data), a 15% edge-quality improvement would clear Opus's overhead easily.

### Architecture implication: gates do more, LLM does less

To keep API cost sane at the higher tiers, tighten the Python gate filters so only the top ~1% of events hit the LLM:

```
Event stream  →  Cheap gates  →  LLM (Opus)  →  Execute
  (thousands)     (rejects 99%)    (dozens)      (handful)
```

Old model: LLM called 1174× per match → even at Haiku pricing that's 30k+ calls/day if we scale trading volume.
New model: LLM called 20-50× per match → Opus is affordable, AND every call is on a pre-vetted high-signal event.

Python gates get stricter:
- Liquidity ≥ $2,000 (was $1,000)
- Spread ≤ 8¢ (was 10¢)
- Ask in 30-65¢ (was 25-70¢) — narrower edge band
- Never during the first 2 rounds of a map (rounds are too random)
- Never with <30s elapsed since last evaluation on same match (dedupe bursts)
- Require buy team to have won at least 1 map OR current-map round lead ≥ 2

These gates are code-owned, easy to unit-test, and cheap to evaluate. The LLM only sees setups that already look profitable.

### Dev cycle
- **Local** (5090 + qwen3 / deepseek-r1 / qwen3.6): free prompt iteration, backtest regression testing, shootouts. Ollama stays as the dev environment.
- **Production** (VPS + Claude Sonnet or Opus): live trading.
- **A/B** (nightly): run the day's recordings through BOTH local winner AND the production Claude model; alert if they disagree on >20% of decisions (prompt drift indicator).

### Production tier SELECTED (2026-04-17): `claude-haiku-4-5-20251001` + iter8 prompt

Ran Haiku, Sonnet, and Opus on the same 32-file corpus × both iter8 and iter9 prompts.

**Results:**

| Tier | iter9 (minimal) | iter8 (detailed rules) |
|---|---|---|
| **Haiku 4.5** | −$52.87 / 265 trades / 33% WR | **+$20.39 / 3 trades / 100% WR** ✅ |
| Sonnet 4.5 | $0 / 0 trades | $0 / 0 trades (too cautious) |
| Opus 4.7 | $0 / 0 trades | $0 / 0 trades (too cautious) |

**Key findings:**
1. Haiku + iter8 beat every local model on $/trade: +$6.80 (vs +$3.43 for deepseek-r1:14b) at a 100% win rate.
2. Sonnet and Opus are *too literal* about iter8's detailed requirements — they always find a reason to skip. Haiku interprets the spirit of the rules, which in our case is the right behavior.
3. iter9-minimal is actively harmful with Haiku (-$52 vs +$20). Claude needs the detailed rules.
4. **Higher-tier Claude loses money** (vs zero) compared to Haiku — the cost argument is moot.

**Operational:**
- API cost: ~$9/month at expected live volume (~300 LLM calls/day, Haiku pricing).
- Fallback: if Haiku misbehaves in live data, the fallback chain is deepseek-r1:14b (local, +$27.41 in shootout) → mistral-small:24b (+$13.02).
- Set in `.env`: `EDGE_CLAUDE_MODEL=claude-haiku-4-5-20251001`.

**Dev/backtest tier still:** local qwen3 / deepseek-r1:14b on the gaming PC. Zero-cost iteration on prompt changes; move to Haiku only when a candidate beats the current prompt on recorded data.

---

## 4. Prompt — iter9-minimal direction

Key insight from shootout: **models with more prompt data trade more and lose more.** Winners are the ones that ignore most signals.

### iter9-minimal system prompt (~100 tokens)
```
CS2 Polymarket trader. Output ONE line of JSON, nothing else:
{"action":"buy"|"skip","confidence":0-1,"reason":"short"}

Default is SKIP. BUY only when the setup is simple and strong.

HARD GATES (outside these → SKIP):
- ask between 25¢ and 70¢
- spread ≤ 10¢

BUY requires BOTH:
1. Buy team is LEADING or TIED in maps (never TRAILING in maps).
2. At least ONE clear current advantage: just won a round OR economy lead OR more players alive.

TAIL-RISK: if ask > 70¢, SKIP unless a map just ended in buy team's favor.
```

### iter9-minimal user prompt (~150 tokens, 4 facts only)
1. Map W-L context ("X leads 1-0 in maps")
2. Single clearest current advantage ("economy lead $18k vs $4k")
3. Ask + spread ("ask 55¢, spread 4¢")
4. The latest trigger event ("Trigger: round win for X")

**Everything else from iter8 is removed:** HP arrays, AWP holdings, bomb state, multi-window momentum, 10 recent events, CT/T bias tables. The shootout proved those distract small-active-param models.

### Output: LLM decides action + confidence only
Bet sizing, TP, SL are owned by Python code. LLM returns `{action, confidence, reason}` — three fields, no numeric hallucination surface.

---

## 5. Infrastructure — VPS architecture

**Move off Mac. Target: VPS + remote Anthropic API + remote Polymarket.**

### Hardware
- **VPS** — Hetzner CCX33 or CCX23 (~$15-25/month, 4-8 vCPU, 16 GB RAM, 24/7 uptime, Germany/Finland)
- **Gaming PC** — stays on, used for dev/backtest, Ollama server for local model iteration
- **Mac** — dev machine, can be off

### Services (all on VPS)
```
┌───────────────────────────────────────────────────────────────┐
│                         VPS (Ubuntu 24.04)                    │
├───────────────────────────────────────────────────────────────┤
│  bot.service  (systemd)  →  edge_bot.py + feeds + executor    │
│  postgres     (systemd)  →  trades, recordings, match state   │
│  redis        (systemd)  →  WS pub/sub, event queue           │
│  grafana      (docker)   →  dashboards on :3000               │
│  prometheus   (docker)   →  metrics                           │
│  nginx        (systemd)  →  reverse proxy + TLS               │
│  watchdog     (systemd)  →  restarts bot.service on crash     │
└───────────────────────────────────────────────────────────────┘
     ▲                              ▲
     │                              │
     │ Anthropic API (Haiku)        │ Polymarket CLOB API
     │                              │
     └───── bo3.gg WS / HLTV / Steam GSI ─── live data feeds ────
```

### Secrets management
- `.env` in the VPS home dir, chmod 600
- Polymarket private key stored only on VPS, never committed
- `ANTHROPIC_API_KEY` scoped to the bot's workspace
- Nightly encrypted backup of `trades.db` + `.env` to user's Mac

### Deploy cadence
- `git push` to VPS triggers `systemctl restart bot.service`
- Blue-green: run new version in dry-run for 1 hour against live feeds before cutting over
- Can always `systemctl stop bot.service` remotely to kill the bot

---

## 6. Observability

**Every trade is a row** (postgres):
```
trade_id | match_id | team | entry_ts | entry_price | bet_usd
         | exit_ts  | exit_price  | exit_reason | pnl_usd | pnl_pct
         | llm_confidence | llm_reason | trigger_event | market_state_snapshot
```

**Every LLM call is a row** (postgres):
```
call_id | ts | model | input_tokens | output_tokens | latency_ms
        | prompt | response | parsed_action | parsed_confidence
```

**Grafana dashboards** (public-readable on VPS via Tailscale):
- Live P&L + balance curve
- Trade heatmap by hour / tournament / map
- Win rate rolling 7d
- LLM latency p50/p95/p99
- Bot uptime

**Alerts** (Telegram via `telegram-send` CLI, zero-cost):
- 🟢 New trade opened (bet size + reason)
- 🟢 Position closed in profit (pnl + held-duration)
- 🔴 Position closed at SL (immediate)
- 🔴 Daily loss limit hit (disabled for day)
- 🔴 Hard stop triggered (bot killed)
- ⚠️ Bot crashed, watchdog restarting
- ⚠️ LLM error rate > 5% over last 100 calls
- ⚠️ Polymarket API returning errors

---

## 7. Roadmap — what ships when

### Phase 0 — finish what's in-flight (this session)
- [x] Complete the 13-model shootout (iter8 prompt, 32 curated files)
- [x] Complete the iter9 A/B tests on qwen3 / deepseek-r1 / qwen3.6
- [x] Write `PLAN.md` (this doc) and `README.md`
- [ ] Decide: iter9 winner becomes the production prompt

### Phase 1 — fix data pipeline (1 day)
1. **Patch cs2_bo3.py (polling feed) to emit raw_snapshot** too, so all recordings are backtest-ready. Today only 2% of snapshots (WS bursts) are rich.
2. **Add token_map to recorder schema** (already done for WS, also needs polling)
3. **Validate** 1 day of new recordings is fully backtestable end-to-end

### Phase 2 — risk engine (1-2 days)
1. Move bet sizing / TP / SL / kill-switches from prompt to `risk.py` module
2. Add daily-loss stop, hard-stop, cooldown after N losses
3. Add `HARD_STOP.lock` file mechanism so bot refuses to start after −50%
4. Unit tests for every risk rule

### Phase 3 — VPS migration (2 days)
1. Provision Hetzner VPS, install Python 3.11, Postgres, Redis, Docker
2. Port edge_bot.py to run as systemd service with auto-restart
3. Migrate SQLite → Postgres (adds concurrent-safe writes)
4. Set up Grafana + Prometheus + Telegram alerts
5. Deploy in dry-run, let it run 24h against live feeds, verify trades match backtest

### Phase 4 — Haiku integration (1 day)
1. Wrap iter9 prompt in `edge_analyst_haiku.py` using Anthropic SDK
2. Add prompt-caching to cut input cost by 90% on repeated system prompt
3. A/B vs local deepseek-r1:14b on 1 week of new recordings
4. Cut over to Haiku if it wins, else stay local

### Phase 5 — shadow-trade + training-wheels go-live (2 weeks)

**Week 1 — shadow trading (no real money at risk):**
1. Bot runs end-to-end on VPS against real live feeds and real Polymarket orderbook
2. Every LLM "buy" decision is logged to `shadow_trades` table instead of sent to the CLOB
3. Position monitor simulates TP/SL/timeout against real bid/ask ticks (reads orderbook, doesn't order)
4. Grafana shows simulated PnL curve
5. Exit criteria: 3 consecutive days of shadow PnL that's NOT catastrophic (no "blew up" scenarios), and reproduces backtest PnL within ±30%

**Week 2 — parallel shadow + tiny live (training-wheels):**
1. Continue shadow trading
2. ALSO start placing real orders but cap `PER_TRADE_MAX_PCT = 0.01` (1% = $10 max)
3. Run both in parallel for 3 days → compare shadow vs real PnL. Differences reveal execution slippage / fill issues.
4. Step up sizing ONLY when shadow ≈ real within ±20% AND live PnL is not red:
    - Day 4-7: `PER_TRADE_MAX_PCT = 0.02` ($20 max)
    - Day 8+: `PER_TRADE_MAX_PCT = 0.04` ($40 max, production target)

**Daily during Phase 5:**
- Dump trades + shadow trades to Grafana
- Review every losing trade manually before the next day
- Weekly: re-run backtest on the week's recordings to confirm prompt is still stable

### Phase 6 — improve loop (ongoing, monthly)
- Re-run shootout on latest recordings to catch model drift
- Try next-gen models when they land on Ollama (DeepSeek-R2, Qwen3.6 variants)
- Adjust risk params based on actual trade distribution
- Ad-hoc: any time a big loss happens, review prompt + data + prompt until root cause is found

### Phase 7 — Telegram admin interface (2-3 days, can parallel with Phase 3)

A python-telegram-bot service running on the VPS, authenticating on the user's Telegram user ID only.

**One-way alerts (bot → user):**
- Trade opened: `🟢 BUY FURIA @ 0.54 (bet $40, conf 0.82, reason: "leads 1-0 with eco advantage")`
- Trade closed (TP): `🟢 EXIT FURIA @ 0.594 (+10% hit, PnL +$4.00, held 2m14s)`
- Trade closed (SL): `🔴 EXIT FURIA @ 0.459 (−15% hit, PnL −$6.00, held 4m02s)`
- Daily summary at 23:59 UTC: `📊 Today: 6 trades, 4W/2L, PnL +$18.20, bankroll $1018.20 (+1.82%)`
- Errors: `⚠️ LLM error rate 8% last hour — check logs`
- Kill-switch: `🛑 DAILY_LOSS_STOP hit at -10.2%, trading halted until 00:00 UTC`

**Two-way commands (user → bot):**
- `/status` — running/paused, open positions, today's PnL
- `/positions` — list open trades with current PnL
- `/pnl [today|week|month|all]` — aggregated stats
- `/pause` — stop opening new trades (existing positions continue to be monitored)
- `/resume` — re-enable new trades
- `/hardstop` — immediate kill + lock file (manual unlock required)
- `/config` — current risk params
- `/backtest last` — run yesterday's recordings through the backtest, reply with summary

**Free-text chat (user → Claude Haiku → user):**
- Context injected: current positions, today's trades, last N errors, current config
- User: "how's today going?" → Claude summarizes with real data
- User: "why did you enter that FURIA trade?" → Claude reads the stored `llm_reason` + `market_state_snapshot` and explains
- Requires message-history cache so conversation has memory

---

## 8. Risk checklist before going live

**Hard blockers — do NOT flip `--live` without these:**

- [ ] Dry-run for 7 consecutive days on VPS, reproduces backtest PnL ±20%
- [ ] All risk rules covered by unit tests (daily-stop, hard-stop, cooldown, position cap)
- [ ] `HARD_STOP.lock` mechanism proven to block startup
- [ ] Watchdog auto-restart tested (kill -9 the bot; verify systemd revives it)
- [ ] Telegram alerts tested end-to-end (trade open / close / error)
- [ ] Polymarket key has ONLY the $1,000 bankroll — not your main wallet
- [ ] Clear kill-switch: you can `ssh vps && systemctl stop bot.service` from phone in <60s
- [ ] Trade log writes are atomic — a crash mid-trade doesn't corrupt state
- [ ] Polymarket rate limits respected; bot never hits 429

**Soft blockers — nice to have:**

- [ ] Live dry-run against paper-trading (simulated fills using real orderbook) matches real PnL over 3 days
- [ ] Ensemble: run Haiku + deepseek-r1:14b in parallel, only trade when both agree (requires Phase 4 done)
- [ ] Volatility-adjusted bet size (higher volatility = smaller bet)

---

## 9. Decisions locked in (2026-04-17)

1. **Polymarket wallet** — use the existing one at `0x4319abd30cc8e1dcf633b25eebf7c5c6d33cb25a` (per `.env`). Fund it with $1,000 for the bot's bankroll.
    - Safety note: since this is a live wallet, we will enforce `PER_TRADE_MAX_PCT` + `HARD_STOP` in code; the bot must never be able to place an order >8% of its last-known balance.

2. **VPS vendor — QuantVPS.** User has an existing account/instance.
    - QuantVPS is a trading-focused VPS host (low-latency, typically CME/NY4/LD4 colo).
    - Need from user: IP address, SSH access method (key/password), OS version, resource specs (CPU/RAM/disk).

3. **Paper-fill validation included** — Phase 5 now includes a **shadow-trading window** where the bot runs end-to-end against the real Polymarket orderbook and makes decisions, but:
   - Does NOT send orders to the CLOB
   - Records "what it would have bought at what ask" in the trade log
   - Position-monitors against the real orderbook (simulates TP/SL/timeout using actual bid/ask ticks)
   - Reports simulated PnL side-by-side with a tiny live tranche (`PER_TRADE_MAX_PCT = 0.01` = $10 max) for a 3-day overlap
   - Only cut to full live sizing after shadow PnL ≈ live PnL within ±20%

4. **Telegram** — confirmed. User wants BOTH alerts AND interactive chat.
    - **Alerts** (one-way, bot → user): trade opens/closes, daily PnL, errors, kill-switch events.
    - **Interactive chat** (two-way): user sends commands / natural-language questions; bot replies.
    - Commands planned: `/status`, `/positions`, `/pnl`, `/pnl_today`, `/stop`, `/resume`, `/hardstop`, `/config`, `/backtest last`.
    - Free-text chat: relay to Claude Haiku with the bot state as context, so user can ask "how's today going?" or "why did you enter that FURIA trade?" and get a real answer.
    - Open item: user needs to create a Telegram bot via `@BotFather` and share the token + their Telegram user ID (for auth — only the user can command the bot).

## 10. Follow-up questions before Phase 3 starts

These are things the user needs to confirm/provide, but they don't block writing code:

- QuantVPS IP + SSH key + OS version + specs
- Telegram bot token (from @BotFather) + Telegram user ID (so the bot only listens to the user)
- Preferred TZ for "daily" window (UTC? user's local? Polymarket's settle clock?)
- Polymarket wallet funding status — is $1k already there or still to be deposited?

---

*Last updated: 2026-04-17. Author: shared session with Claude.*
