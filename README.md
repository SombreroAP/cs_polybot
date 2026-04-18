# Esports Latency Arbitrage Bot

**Bets on CS2 map-winner markets on Polymarket in real time.** Exploits the 2-5s latency window between our game-event feed and Polymarket's orderbook reprice.

**Stack:** Python 3.11 · Polymarket CLOB · bo3.gg WebSocket · Anthropic Claude Haiku (production) · Local Ollama (dev/backtest) · Postgres · systemd/VPS · Grafana/Telegram observability.

**Status:** pre-live. Working backtest + live data collection. Design doc locked in [`PLAN.md`](./PLAN.md).

---

## Quick links

| | |
|---|---|
| Master plan, architecture, roadmap | [`PLAN.md`](./PLAN.md) |
| Current best prompt (iter8) | `edge_analyst.py` |
| Minimal prompt being A/B tested (iter9) | `_logs/edge_analyst.iter9_minimal.py` |
| Frozen iter8 reference | `_logs/edge_analyst.iter8_frozen.py` |
| Live backtest dashboard (while shootout runs) | http://localhost:8084/ |
| Shootout index dashboard (always-on) | http://localhost:8085/ |
| Shootout runner | `_model_shootout.sh` |
| iter9 runner | `_iter9_shootout.sh` |
| Data recordings | `data/recordings/` (1,362 files, 2.4 GB) |

---

## What we've built so far

### The decision pipeline
```
Live feeds  ─► Trigger detector ─► Gate filters ─► LLM decision ─► Executor ─► Position monitor
(bo3.gg WS    (round_end,         (liquidity,      (qwen/Haiku)   (Polymarket)  (TP/SL/timeout)
 HLTV,         map_win,            spread,
 Steam GSI)    clutch, kill)       price band)
```

### Core files

| File | Role |
|---|---|
| `bot.py` | Main orchestrator (`EsportsBot` class). Wires feeds → analyzer → executor → recorder. |
| `edge_bot.py` | Production entrypoint — spawns `EsportsBot` with edge-bot config |
| `feeds/cs2_bo3_ws.py` | bo3.gg WebSocket feed (live CS2 round/kill data) |
| `feeds/cs2_bo3.py` | bo3.gg REST polling (fallback when WS silent) |
| `feeds/cs2_hltv.py` | HLTV scorebot (secondary source) |
| `feeds/cs2_gsi.py` | Steam GSI (local spectator data, disabled by default) |
| `latency.py` | Matches game events to Polymarket markets; filters for latency edge |
| `edge_analyst.py` | Builds the LLM prompt + parses decisions (**iter8** = current production) |
| `match_analyzer.py` | Tracks match state, builds rolling feature vectors |
| `match_recorder.py` | Records every event + snapshot to JSONL for backtesting |
| `executor.py` | Sends orders to Polymarket, manages open positions, TP/SL/timeout |
| `polymarket_ws.py` | Polymarket orderbook WebSocket |
| `persistence.py` | SQLite (will migrate to Postgres) — trades, balance, state |
| `dashboard.py` | Flask dashboard on :8082 (live bot) and :8083 (edge bot) |
| `replay_backtest.py` | Replays recorded JSONL through the full pipeline — headline backtest tool |
| `_model_shootout.sh` | Runs the iter8 prompt against N models back-to-back |
| `_iter9_shootout.sh` | Runs the iter9-minimal prompt against qwen3/deepseek/qwen3.6 |
| `_shootout_index_server.py` | Always-on HTTP server on :8085 showing shootout results |

### What's known to work
- **Recorder** — actively capturing live matches (1,362 files / 2.4 GB). Bot runs on Mac in RECORD_ONLY mode.
- **Backtester** — replays JSONL files end-to-end, produces PnL + trade log. See `replay_backtest.py`.
- **Latency filter** — matches CS2 games to Polymarket markets using team-name fuzzy match + market-type filter.
- **Executor** — places market orders, tracks positions, enforces TP/SL in the code.
- **Prompt iteration** — iter1 → iter8 documented under `_logs/edge_analyst.iter*.py` + `_logs/iter*.log`.
- **Model shootout infrastructure** — 13 models tested head-to-head on the same 32-file corpus.

### What's broken / incomplete
1. **Recording format — polling fallback doesn't include rich bo3.gg payload.** Only ~2% of SNAPSHOT_MATCH_UPDATE lines have full player_states / round_phase / HP data. Fix: patch `feeds/cs2_bo3.py` to emit the same `_emit_raw_snapshot` as the WS feed.
2. **token_map in old recordings** — 735 older files have a META block missing `token_id_a` / `token_id_b`. They parse to 0-trade backtests. Going forward the recorder writes these correctly (Phase 1 in PLAN).
3. **Mac-only deployment** — bot currently runs on Mac. Needs VPS migration for 24/7 uptime (Phase 3 in PLAN).
4. **Risk rules live in prompt, not code** — bet sizing and TP/SL are currently qwen's output. Should be Python rules with the LLM deciding action+confidence only (Phase 2 in PLAN).

---

## Shootout results (for the record)

**Setup:** iter8 prompt, 32 curated CS2 replay files, $1,000 starting balance, `calls_per_match=40`.

| Rank | Model | PnL | Trades | W/L | Behavior |
|---|---|---|---|---|---|
| 🥇 | **deepseek-r1:14b** | **+$27.41** | 8 | **8W/0L** | Reasoning model — ultra-selective |
| 🥈 | mistral-small:24b | +$13.02 | 37 | 24W/13L | Moderate volume, 65% WR |
| 🥉 | llama3.1:8b | +$1.06 | 14 | 8W/6L | Hits bet floor on everything |
| — | esports-qwen3 (custom) | $0.00 | 0 | — | Never fires buy (num_predict=80 too tight) |
| — | qwen2.5:32b-instruct | $0.00 | 0 | — | Won't commit to buys |
| — | qwen2.5-coder:32b | $0.00 | 0 | — | Won't commit to buys |
| — | qwen3:30b-a3b (baseline) | −$1.46 | 18 | 12W/6L | Our current dev default |
| ❌ | qwen2.5:14b-instruct | −$101.80 | 209 | 121W/83L | Overtrades, avg loss > avg win |
| ❌ | phi4:14b | −$205.22 | 117 | 81W/32L | Overtrades, took 1 huge loss |
| ❌ | deepseek-r1:32b | −$45.22 | 39 | 21W/18L | Bigger r1 did worse than 14b |
| 💀 | gemma4:8b | −$936.77 | 326 | 134/175 | Blew up — fires buy on noise |

**Key finding:** selectivity (ignoring signals) beats intelligence (analyzing them) for our small active-param models. Winners: moderate trade count + high win rate. Losers: either trade count too high, or too low.

### Still queued
- `qwen3:30b-a3b-think`
- `llama3.3:70b-instruct-q4_K_M`
- `qwen3.6:35b-a3b` (iter8 followup, new model released 2026-04-16)
- iter9-minimal on: qwen3:30b-a3b, deepseek-r1:14b, deepseek-r1:8b-0528, qwen3.6:35b-a3b

---

## Dev workflow

### Run a backtest locally
```bash
python3 replay_backtest.py \
    --files-list _audit/good_files.txt \
    --duration 0 \
    --calls-per-match 40 \
    --port 8084 \
    --starting-balance 1000 \
    --tag my_experiment \
    --model qwen3:30b-a3b
```
Watch http://localhost:8084/ for live match progress.

### Run a model shootout
```bash
bash _model_shootout.sh
```
Dashboard at http://localhost:8085/ shows all models side-by-side.

### Start the recorder (live CS2 match data collection)
```bash
RECORD_ONLY=true python3 edge_bot.py --port 8082
```
Data lands in `data/recordings/{match_id}_{teams}_{date}.jsonl`.

### Freeze iter8 before experimenting
```bash
cp edge_analyst.py _logs/edge_analyst.iter8_frozen.py
# ... experiment ...
cp _logs/edge_analyst.iter8_frozen.py edge_analyst.py   # restore
```

---

## Prompt guide for future sessions (read this first)

This project has **a lot** of history. When starting a new session, use the following guide so we don't reinvent work.

### Before asking "what should we do?"
1. **Read `PLAN.md` first.** The master strategy + architecture + roadmap is there.
2. **Check recent git log / `_logs/*.log`** — there's usually a shootout or backtest in flight.
3. **Check process list:** `ps aux | grep -E "replay_backtest|_model_shootout|_iter|edge_bot"` — tells you what's live right now.

### Default to these design rules (they're hard-won)

1. **Risk rules go in Python code, not the LLM prompt.** We tried having qwen decide bet_size/TP/SL and it hallucinated numbers. Python owns everything except action + confidence.

2. **Minimal prompt beats rich prompt.** iter7→iter8→iter9 research proved models with more context trade more and lose more. Target <500 prompt tokens.

3. **Selectivity beats intelligence.** Our best model (deepseek-r1:14b) trades 8x in 32 matches. Any model trading >50x is losing money.

4. **Cloud Haiku for production, local for dev.** $5/month total API cost vs running a gaming PC 24/7 — the math is obvious. Keep local for free iteration.

5. **Never break the recorder.** The live bot has been collecting data since day 1. Touching `match_recorder.py` or `bot.py`'s event path needs extreme care.

6. **Always freeze before experimenting.** `cp edge_analyst.py _logs/edge_analyst.iter8_frozen.py` before any prompt change. We've lost work to this.

7. **Backtest before you believe it.** Prompt that wins on 32 files but hasn't been tested on new recordings is not validated. Run it through the full pipeline on fresh data before rolling out.

8. **One process, one port.** Flask dashboard bound to :8084 for live backtest, :8085 for shootout index, :8082 for live bot, :8083 for legacy edge bot. Never overlap.

### Common questions and their answers

**"What model are we using?"**
- Dev/backtest: `qwen3:30b-a3b` on Ollama at the gaming PC (env var `OLLAMA_URL`).
- Production target: `claude-haiku-4-5-20251001` via Anthropic API (not yet deployed).

**"Where are the recordings?"**
- `data/recordings/*.jsonl` (1,362 files, 2.4 GB as of 2026-04-17).
- Format: JSONL with `_META`, `SNAPSHOT_MATCH_UPDATE`, `_PM_best_bid_ask`, `GAME_EVENT_*` lines.

**"What's the current prompt?"**
- Iter8 (production): `edge_analyst.py` lines 21-97 (SYSTEM) and 453-581 (user builder).
- Iter9-minimal (candidate): `_logs/edge_analyst.iter9_minimal.py`.

**"How do I test a new model?"**
1. Pull on the Ollama server: `curl -X POST http://192.168.76.196:11434/api/pull -d '{"name":"<tag>","stream":false}'`
2. Add to `MODELS=(...)` in `_model_shootout.sh` or `_iter9_shootout.sh`
3. Run the shootout; results land in `_logs/shootout/*.log`

**"Ollama port is wrong / server won't respond"**
- Port is **11434** (default). Earlier we had 11435 — that was a misconfig that's been fixed.
- `OLLAMA_URL` in `.env` should be `http://192.168.76.196:11434`.

**"The bot is doing badly live"**
- Live vs backtest drift is a known issue. Research agent flagged this previously (64% backtest win rate → 26% live). Likely cause: feature-parity bug between live prompts and backtest prompts. Needs audit before going live for real.

**"I want to pull a new LLM model"**
- Use the gaming PC's Ollama at `192.168.76.196:11434`.
- Pulling from Mac can fail with "requires macOS" errors for some quantization variants. Prefer remote pull via API: `curl -X POST http://192.168.76.196:11434/api/pull -d '{"name":"<tag>"}'`

**"I want to run iter9 / new prompt tests"**
- Edit `_iter9_shootout.sh` MODELS list, kill+restart driver (`pkill -f _iter9_shootout && nohup bash _iter9_shootout.sh > _logs/iter9_driver.out 2>&1 &`).
- iter9 driver waits for main shootout + followup to finish before starting, so restarts are safe.

### Files you probably don't need to touch
- `feeds/lol_*.py`, `feeds/dota2_*.py`, `feeds/valorant_*.py` — other games, not active
- `finetune_gemma4.py`, `generate_training_data.py`, `prepare_training_data.py` — fine-tuning work, deferred
- `claude_analyst.py` — old 11-expert panel, superseded by `edge_analyst.py`
- `news_scraper.py`, `team_intel.py` — priors, not currently used
- `_audit/*` — one-off diagnostic scripts

### Files that are load-bearing (tread carefully)
- `bot.py` (~1600 lines) — the main orchestrator. Every feed, recorder, and executor wires through here.
- `edge_analyst.py` — the prompt + LLM call logic. Every production decision flows through `should_buy()`.
- `latency.py` — matches game events to Polymarket markets. Team-name matching lives here.
- `executor.py` — places real orders. Do not touch without dry-run validation.

---

## Glossary

| Term | Meaning |
|---|---|
| **iter8** | Current production prompt — ~600 tokens, detailed rules, tail-risk guard |
| **iter9** | Minimal prompt (~200 tokens) — A/B test in progress |
| **trigger** | A game event (round_end, kill_streak, etc.) that causes the bot to *consider* a trade |
| **gate** | A Python-level filter (liquidity, spread, map score) that blocks most triggers before they reach the LLM |
| **tail risk** | A "big favorite" trade at ask > 70¢ that can lose >60¢ if the team collapses. We cap bet size for these. |
| **RECORD_ONLY** | Env var that makes the bot ingest feeds + record, but skip all trading logic |
| **shootout** | Head-to-head comparison of N models on the same backtest corpus |
| **followup** | Secondary shootout run for models pulled after main shootout started |
| **a3b** | Suffix for qwen3's 3B-active MoE models (3B active params out of 30-35B total) |

---

## Contributing / picking up where we left off

If you open this project cold:

1. Read `PLAN.md` (the spec).
2. Read this README's "Prompt guide for future sessions".
3. Check `ps aux | grep -E "_iter9|_model_shootout|_shootout_follow|replay_backtest|edge_bot"` to see what's running.
4. Check `_logs/shootout/*.log` for the most recent model comparisons.
5. Check `data/recordings/` for the freshest match data.

Then ask the user one question: *what's the next blocker?*

---

*Project initialized: late 2025. Latest major update: 2026-04-17 — shootout + iter9 A/B + PLAN.md lockdown.*
