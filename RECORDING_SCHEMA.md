# Recording Schema — `data/recordings/*.jsonl`

Every recording file is newline-delimited JSON (`.jsonl`). Each line is an envelope with this shape:

```json
{
  "ts": 1776069279.632,
  "ts_iso": "2026-04-13T08:01:19.632165+00:00",
  "match_id": "117108",
  "message_type": "SNAPSHOT_MATCH_UPDATE",
  "raw": { ...payload-specific... }
}
```

- `ts` — unix float seconds, ground truth ordering
- `ts_iso` — ISO 8601 mirror of `ts` for readability
- `match_id` — stable bo3.gg or `mkt_{id}` identifier; consistent within a file
- `message_type` — one of the enumerated types below (see §2)
- `raw` — payload. Schema depends on `message_type`.

---

## 1. File naming

`{match_id}_{team_a}_vs_{team_b}_{YYYY-MM-DD}.jsonl`

Examples:
- `117108_Vitality_vs_Natus_Vincere_2026-04-17.jsonl`
- `mkt_1929853_FURIA_vs_GamerLegion_2026-04-16.jsonl`

---

## 2. Message types

### `_META` — once per file (first line, usually)

```json
{
  "match_id": "117108",
  "team1": "Vitality",
  "team2": "Natus Vincere",
  "game": "cs2",
  "bo_type": 3,
  "pm_slug": "1929841",
  "token_id_a": "10668929701080965043259138626356003756479207498582559634639293870778148395055",
  "token_id_b": "48650764458787858368929139291491163473797439324069833645848083086857727018282",
  "start_date": "2026-04-13T08:20:34.027034+00:00"
}
```

**Required for backtesting:** `token_id_a` + `token_id_b`. Without these we cannot map bot decisions to Polymarket sides.

### `SNAPSHOT_MATCH_UPDATE` — repeating, high volume

The full bo3.gg provider payload. Carries everything we need for feature extraction:

```json
{
  "message_type": "SNAPSHOT_MATCH_UPDATE",
  "match_id": "117108",
  "map_name": "de_mirage",
  "game_number": 1,
  "round_number": 13,
  "game_ended": false,
  "round_phase": "LIVE",          // BUY_TIME | LIVE | BOMB_PLANTED | POST_ROUND
  "is_bomb_planted": false,
  "round_time": 28,
  "round_time_remaining": 87,
  "match_status": "live",
  "team_one": {
    "name": "Vitality",
    "side": "CT",                 // CT | TERRORIST
    "score": 7,                   // rounds this map
    "match_score": 1,             // maps won in series
    "equipment_value": 21500,
    "players_alive": 4,
    "player_states": [            // optional but strongly preferred
      { "name": "apEX", "hp": 100, "armor": 100, "money": 3400,
        "weapons": ["ak47","deagle"], "kills": 2, "deaths": 1,
        "has_bomb": false, "has_defuse_kit": true }
    ]
  },
  "team_two": { ...same shape... }
}
```

**Fields the backtest needs (bare minimum):**
- `round_number`, `round_phase`, `is_bomb_planted`
- `team_one.score`, `team_two.score`, `match_score` for both sides
- `team_one.equipment_value`, `team_two.equipment_value`
- `team_one.players_alive`, `team_two.players_alive`
- `team_one.side` (CT / T bias)

**Fields training ALSO needs:**
- `player_states` (HP, weapons, money — lets model learn economy effects)
- `map_name` (CT-bias correlations)
- `round_time_remaining` (clutch situations have time pressure)

### `GAME_EVENT_*` — trigger events

Emitted at discrete moments (round end, kill streak, etc.). These are what the trigger engine fires on.

```json
{
  "event_type": "round_end",
  "team": "a",              // a | b | neutral
  "description": "Round to Vitality (8-5)",
  "team_one": { "score": 8, "match_score": 1 },
  "team_two": { "score": 5, "match_score": 0 },
  "kills_a": 3, "kills_b": 1,
  "map_name": "de_mirage",
  "side": "ct"              // buy team's current side
}
```

Types seen:
- `GAME_EVENT_MATCH_END_ROUND` — round_end, map_win, match_end
- `GAME_EVENT_PLAYER_KILL` — kill_streak, open_kill, clutch_win, teamfight_won
- `SNAPSHOT_MATCH_UPDATE` — score_update fallback

### `_PM_best_bid_ask` — repeating, high volume

Polymarket orderbook best bid/ask for ONE token_id. We get one message per side per update.

```json
{
  "tokenId": "10668929701080965043259138626356003756479207498582559634639293870778148395055",
  "matchId": "117108",
  "bestBid": 0.59,
  "bestAsk": 0.68,
  "asOf": "2026-04-13T08:01:19.632162+00:00"
}
```

### `_PM_book` — full orderbook snapshot (rarer)

```json
{
  "tokenId": "...",
  "matchId": "117108",
  "bids": [ { "price": 0.59, "size": 200 }, { "price": 0.58, "size": 500 } ],
  "asks": [ { "price": 0.68, "size": 150 } ],
  "asOf": "..."
}
```

### `_PM_last_trade_price` — fills reported by Polymarket

```json
{
  "tokenId": "...",
  "matchId": "117108",
  "price": 0.65,
  "asOf": "..."
}
```

### `_PM_trade` — our own fills (live only, skipped in record-only mode)

```json
{
  "tokenId": "...",
  "side": "buy",
  "price": 0.54,
  "size": 40.0,
  "order_id": "0xabc...",
  "asOf": "..."
}
```

---

## 3. Invariants

A **conformant** recording file satisfies:
1. First line is `_META` with non-empty `token_id_a` AND `token_id_b`
2. All subsequent lines have `ts` monotonically non-decreasing (within a 1-second tolerance)
3. `match_id` is constant across the file
4. At least one `SNAPSHOT_MATCH_UPDATE` per game (per map)
5. `game_ended=true` appears at most once per map in `SNAPSHOT_MATCH_UPDATE`

`data/processor.py validate` checks all of these.

---

## 4. Training-data derivations

`data/processor.py` converts recordings into two training artifacts.

### SFT format — `data/training/sft.jsonl`

Supervised fine-tuning examples. One line per decision point. Used to teach a model the iter8 prompt + ideal reasoning.

```json
{
  "messages": [
    { "role": "system", "content": "CS2 Polymarket trader..." },
    { "role": "user", "content": "Evaluating BUY on Vitality..." },
    { "role": "assistant", "content": "{\"action\":\"buy\",\"confidence\":0.85,\"reason\":\"leads 1-0, eco lead 3x\"}" }
  ],
  "label_source": "hindsight",       // hindsight | shootout | human
  "realized_pnl_usd": 4.20,          // if this trade had been taken
  "match_id": "117108",
  "decision_ts": 1776069279.632,
  "map_context": "map 2, round 9, ct side"
}
```

**Label source hierarchy:**
- `hindsight` — we computed "correct" action with knowledge of what happened next (map/series outcome + price trajectory)
- `shootout` — decision was taken in a backtest and ended in profit/loss
- `human` — you reviewed a trade and graded it (`/mark_trade 42 good`)

### RL format — `data/training/rl.jsonl`

State-action-reward tuples for reinforcement learning or offline policy evaluation.

```json
{
  "state": {
    "map_score_a": 1, "map_score_b": 0,
    "round_score_a": 8, "round_score_b": 5,
    "buy_team": "a", "side_a": "ct",
    "eco_a": 21500, "eco_b": 4000,
    "alive_a": 5, "alive_b": 4,
    "ask_a": 0.55, "bid_a": 0.53, "spread_a": 0.02,
    "liquidity_a": 3200,
    "latest_trigger": { "type": "round_end", "team": "a" },
    "... 30+ more features"
  },
  "action": "buy",
  "size_pct_of_bankroll": 0.04,
  "reward_usd": 4.20,                  // PnL realized on this decision
  "terminal": false,
  "next_state": { ... }
}
```

---

## 5. Retrofit story for old recordings

Pre-2026-04-18 recordings have thin `SNAPSHOT_MATCH_UPDATE` lines (only 8 fields). They cannot be fully backtested or training-labeled. Options:

- **Use only `_META` + `GAME_EVENT_*` + `_PM_best_bid_ask`** from old files → limited training signal
- **Skip them entirely** for training, use only for market-latency feature extraction

The processor marks old-format files with `rich_snapshots: false` in its validation output. The shootout corpus in `_audit/good_files.txt` is all "new format" (from `Test Data/` which was produced by a richer recorder).

---

## 6. Versioning

This schema is **v1**. Any future breaking change must:
1. Bump `_META.schema_version` (new optional field, default 1)
2. Update the processor to handle both v1 and vN
3. Document the change in this file
