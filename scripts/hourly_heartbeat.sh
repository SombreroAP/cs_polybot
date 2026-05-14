#!/usr/bin/env bash
# hourly_heartbeat.sh — runs on the VPS via cron each hour, sends a one-line
# status to Telegram. Designed to be quiet on a healthy bot, loud on anything
# unusual. Failure modes:
#   - service not active                -> red dot
#   - last decision > 60 min ago        -> warn
#   - errors in stderr in last hour     -> count
#
# Output: one Telegram message per hour. No-op if TELEGRAM_*  not set.

set -euo pipefail
cd "$(dirname "$0")/.."
set -a; source .env; set +a

[ -z "${TELEGRAM_BOT_TOKEN:-}" ] && exit 0
[ -z "${TELEGRAM_USER_ID:-}" ] && exit 0

# 1. Service health
if systemctl is-active cs2bot.service >/dev/null 2>&1; then
  svc="🟢 active"
else
  svc="🔴 DOWN"
fi

# 2. Recent activity from sqlite (handles WAL transparently when reading live)
DB="data/trades.db"
HOUR_AGO=$(( $(date +%s) - 3600 ))
DAY_AGO=$(( $(date +%s) - 86400 ))

decisions_1h=$(sqlite3 "$DB" "SELECT COUNT(*) FROM claude_decisions WHERE timestamp > $HOUR_AGO;" 2>/dev/null || echo 0)
decisions_24h=$(sqlite3 "$DB" "SELECT COUNT(*) FROM claude_decisions WHERE timestamp > $DAY_AGO;" 2>/dev/null || echo 0)
buys_24h=$(sqlite3 "$DB" "SELECT COUNT(*) FROM claude_decisions WHERE timestamp > $DAY_AGO AND action='buy';" 2>/dev/null || echo 0)
last_dec=$(sqlite3 "$DB" "SELECT MAX(timestamp) FROM claude_decisions;" 2>/dev/null || echo "")
last_dec_ago_min=""
if [ -n "$last_dec" ] && [ "$last_dec" != "" ]; then
  now=$(date +%s)
  last_dec_int=${last_dec%.*}
  last_dec_ago_min=$(( (now - last_dec_int) / 60 ))
fi

shadow_open=$(sqlite3 "$DB" "SELECT COUNT(*) FROM shadow_trades WHERE exit_price IS NULL;" 2>/dev/null || echo 0)
shadow_total=$(sqlite3 "$DB" "SELECT COUNT(*) FROM shadow_trades;" 2>/dev/null || echo 0)

# 2b. Market-maker stats from dedicated mm.db
MMDB="data/mm.db"
mm_dec_1h=$(sqlite3 "$MMDB" "SELECT COUNT(*) FROM mm_decisions WHERE ts > $HOUR_AGO;" 2>/dev/null || echo 0)
mm_dec_24h=$(sqlite3 "$MMDB" "SELECT COUNT(*) FROM mm_decisions WHERE ts > $DAY_AGO;" 2>/dev/null || echo 0)
mm_fills_1h=$(sqlite3 "$MMDB" "SELECT COUNT(*) FROM mm_fills WHERE ts > $HOUR_AGO;" 2>/dev/null || echo 0)
mm_fills_24h=$(sqlite3 "$MMDB" "SELECT COUNT(*) FROM mm_fills WHERE ts > $DAY_AGO;" 2>/dev/null || echo 0)
# Sim PnL: sum of latest sim_cash_after across tokens that had a fill in last 24h
mm_pnl_24h=$(sqlite3 "$MMDB" "
  SELECT COALESCE(ROUND(SUM(c), 2), 0)
  FROM (
    SELECT (SELECT sim_cash_after FROM mm_fills f2
            WHERE f2.token_id = f1.token_id
            ORDER BY ts DESC LIMIT 1) AS c
    FROM mm_fills f1
    WHERE f1.ts > $DAY_AGO
    GROUP BY token_id
  );" 2>/dev/null || echo 0)

# 3. Recent errors
errors_1h=$(awk -v cut="$(date -u -d '1 hour ago' '+%Y-%m-%d %H:%M:%S' 2>/dev/null || date -u -v-1H '+%Y-%m-%d %H:%M:%S')" \
  '$0 ~ /ERROR/ && $0 >= cut' _logs/cs2bot.stderr.log 2>/dev/null | wc -l | tr -d ' ' || echo 0)

# 4. Compose
mode="🔵 SHADOW"
[ "${MM_LIVE_TRADING:-false}" = "true" ] && mode="🟢 LIVE"

stale_warn=""
if [ -n "$last_dec_ago_min" ] && [ "$last_dec_ago_min" -gt 60 ]; then
  stale_warn=" ⚠️ stale ${last_dec_ago_min}m"
fi

msg="⏱ Hourly heartbeat — $(date -u '+%H:%M UTC')

svc: $svc | mode: $mode
MM: ${mm_dec_1h} dec/h, ${mm_fills_1h} fills/h (${mm_dec_24h}/${mm_fills_24h} in 24h)
sim PnL 24h: \$${mm_pnl_24h}
legacy decisions: ${decisions_24h}/24h${stale_warn}
errors last 1h: ${errors_1h}"

curl -s -X POST "https://api.telegram.org/bot${TELEGRAM_BOT_TOKEN}/sendMessage" \
  -d chat_id="${TELEGRAM_USER_ID}" \
  --data-urlencode text="$msg" >/dev/null
