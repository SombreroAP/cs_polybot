# Go-Live Runbook — Polymarket MM

**Status today:** code is wired and tested in shadow mode. To go live with real orders, you flip env vars on the VPS. **I cannot do this for you — the wallet keys are yours and must never leave your control.**

---

## Prerequisites (one-time)

### 1. A Polymarket account + wallet

- Sign up at https://polymarket.com if you haven't
- Polymarket creates an **email proxy wallet** for you (the address that holds your USDC)
- Note your proxy wallet address: `0x...` (you'll need it as `POLYMARKET_PROXY_WALLET`)

### 2. Deposit USDC + a little MATIC

- USDC on **Polygon** network. Start with **$20-50** — small enough that the worst case is acceptable.
- A few cents of **MATIC** for gas. Polymarket usually handles this for you on email wallets.

### 3. Get your trading private key

This is the wallet that **signs orders**. With an email-proxy wallet:
- The proxy wallet *holds* the funds
- A separate **trading key** signs orders on behalf of the proxy

Get your trading private key from Polymarket:
- Polymarket UI → **Profile → Export Private Key** (or the equivalent in current UI)
- Treat this key like a password. Never commit it. Never share it. Never paste it in chat.

### 4. CLOB approval (one-time, browser)

- In Polymarket UI, place ANY tiny test trade ($1) to trigger the on-chain CLOB approval
- This signs a one-time message that lets the CLOB move USDC + tokens on your behalf
- Without this, the bot's orders will be rejected

---

## VPS setup (you do this — one-time)

SSH to the VPS:

```bash
ssh bot@85.137.174.57
```

Edit the systemd service env file (or create one):

```bash
sudo systemctl edit cs2bot.service
```

Add the following block (replace placeholders):

```ini
[Service]
Environment="MM_LIVE_TRADING=true"
Environment="POLYMARKET_PRIVATE_KEY=0xYOUR_TRADING_PRIVATE_KEY"
Environment="POLYMARKET_PROXY_WALLET=0xYOUR_PROXY_WALLET_ADDRESS"

# Hard safety limits — adjust later when you have track record
Environment="MM_MAX_DAILY_LOSS_USD=10"
Environment="MM_MAX_INVENTORY_USD_PER_TOKEN=20"
Environment="MM_MAX_QUOTE_SIZE_USD=5"
```

**File permissions matter.** The override file lives at:
```
/etc/systemd/system/cs2bot.service.d/override.conf
```
Should be `root:root`, mode `0600` (Polymarket key in there is a secret).

Reload + restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart cs2bot.service
```

Verify the runner picked it up:

```bash
journalctl -u cs2bot.service -n 30 --no-pager | grep -E "MM-TRADER|LIVE"
```

Should see:
```
[MM-TRADER] connected; API creds derived
[MM-TRADER] address: 0xYOUR_PROXY_WALLET_ADDRESS
```

You'll also get a Telegram message:
```
🟢 MM live-trader activated. Limits: ...
```

---

## What changes when you flip the switch

| Behavior | Shadow (now) | Live (after flip) |
|---|---|---|
| Strategy decisions | logged to `mm.db` | logged to `mm.db` AND submitted as real orders |
| Real USDC at risk | $0 | up to `MM_MAX_INVENTORY × n_active_tokens` |
| Order placement | "WOULD BID/ASK" in logs | `BID placed: ... id=...` |
| Fills | simulated from book transitions | real Polymarket trade confirmations |
| Daily kill switch | none | auto-cancel all orders if daily PnL ≤ `-MM_MAX_DAILY_LOSS` |
| Telegram alerts | optional | every fill, every error, every kill-switch |

---

## Safety design

Three independent caps that can't be bypassed:

1. **`MM_MAX_QUOTE_SIZE_USD`** — every single order is capped at this $ size, regardless of what the strategy requests. Bot will resize down if strategy asks for more.
2. **`MM_MAX_INVENTORY_USD_PER_TOKEN`** — total $ exposure across both sides on any single token. New orders that would breach this are skipped.
3. **`MM_MAX_DAILY_LOSS_USD`** — running daily PnL. When this is breached (negative), the bot:
   - Cancels every open order via `cancel_all()`
   - Sets a hard `_killed = True` flag — no new orders until UTC midnight reset
   - Telegrams the kill event

**The defaults above ($5 / $20 / $10) are tuned for proving-it-works mode.** At scale these scale up. Don't increase them until you have a week of live data matching backtest.

---

## What to watch

Dashboard at `http://alpaca-vps:8082` (via Tailscale):
- Top-bar: "MM Mode" goes from **SHADOW** (blue) → **LIVE** (red)
- Top-bar: "Sim PnL 24h" now reflects REAL PnL once fills come in
- New panel showing live-trader status: `active_quotes`, `tokens_with_inv`, `daily_realized_pnl`, `killed` flag

Telegram alerts:
- 🟢 BID placed / 🔵 ASK placed
- ✅ FILL events with running inventory + daily PnL
- 🔴 Errors / 🔴 Kill switch

---

## Recommended ramp

| Week | Quote size | Max inv | Max daily loss | Notes |
|---|---|---|---|---|
| 1 | $5 | $20 | $10 | prove the pipeline works |
| 2 | $10 | $50 | $25 | scale if week-1 PnL ≈ backtest |
| 3 | $25 | $150 | $75 | only if winners > losers cleanly |
| 4+ | $50-100 | $500+ | $250+ | full deployment after 3 weeks of data |

**Don't skip stages.** Backtest ≠ live. Polygon latency, market makers reacting to us, regime drift — all are unknowns that can only show up live.

---

## Kill switches

In order of escalation:

1. **Auto** — daily loss cap auto-cancels all orders and stops trading
2. **Soft stop** — `sudo systemctl edit cs2bot.service` → change `MM_LIVE_TRADING=false` → restart. Bot continues recording + shadow; no new orders.
3. **Hard stop** — `sudo systemctl stop cs2bot.service`. Bot dies entirely. Orders on Polymarket stay live until you cancel them via the UI or the bot restarts.
4. **Cancel everything via Polymarket UI** — log in, go to Open Orders, cancel all manually.

---

## Files / commits relevant

- `mm_live_trader.py` — the live order placer (commit `df01758`+)
- `mm_live.py` — strategy + queue + worker; calls trader on each decision
- `mm_strategy.py` — CatBoost v2 toxic-flow predictor; same as shadow mode
- `models/toxic_flow_v2_cb.pkl` — the model file
- `dashboard.py + templates/edge_dashboard.html` — UI

---

## When you're ready

Step 1: get the wallet credentials in hand (private key + proxy address).
Step 2: SSH in, add the env block, restart.
Step 3: watch the dashboard for 30 minutes; verify a Telegram "🟢 placed" message.
Step 4: if everything looks right, leave it for 24 hours.
Step 5: come back, check `daily_realized_pnl`. If positive or near zero, you have a real strategy. If solidly negative, restart with shadow mode and re-investigate.

I can help with the ramp / debugging at any step. **What I can't do is type the private key for you — that's the one boundary I won't cross.**
