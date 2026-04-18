#!/usr/bin/env python3
"""
Pure Edge Bot — Speed-Only Latency Arbitrage

No Claude API calls. Zero AI cost. Trades purely on latency edge.
When our feed detects an event before the market reacts, buy immediately.
Auto-exits via take-profit (+10%), stop-loss (-5%), or timeout (90s).

Usage:
    python edge_bot.py              # Dry-run with dashboard on port 8083
    python edge_bot.py --live       # Live trading
    python edge_bot.py --port 8084  # Custom port

Can run simultaneously with the main Claude-powered bot (main.py on port 8082).
"""
import os
import sys
import signal
import argparse
import logging
import threading

# Override config BEFORE anything imports it
os.environ.setdefault("DASHBOARD_PORT", "8082")
os.environ.setdefault("MAX_OPEN_POSITIONS", "50")
os.environ.setdefault("EDGE_BOT_MIN_EDGE", "0.01")
os.environ.setdefault("EDGE_BOT_BET_SIZE", "25.0")
os.environ.setdefault("TAKE_PROFIT_PCT", "0.075")
os.environ.setdefault("STOP_LOSS_PCT", "0.125")
os.environ.setdefault("EXIT_TIMEOUT_SECONDS", "600")
os.environ.setdefault("POSITION_MONITOR_INTERVAL", "1")
os.environ.setdefault("EDGE_BOT_NO_SYNTHETIC", "true")  # live matches only
os.environ.setdefault("ENABLED_GAMES", "cs2")

# Use main DB — this is the only bot now
os.environ.setdefault("EDGE_BOT_DB", "data/trades.db")

import config  # noqa: E402 — must import after env overrides

# Force override TP/SL — .env has old values that override setdefault
config.TAKE_PROFIT_PCT = 0.075   # +7.5%
config.STOP_LOSS_PCT = 0.125     # -12.5%
config.EDGE_BOT_BET_SIZE = 25.0
config.EDGE_BOT_MIN_EDGE = 0.01
config.MAX_OPEN_POSITIONS = 50
config.EDGE_CLAUDE_MAX_CALLS = 500   # AI-driven mode: don't throttle qwen — let it evaluate every event

# Focused on CS2 + Dota2 only (LoL/Valorant disabled — awaiting GRID data access)
config.ENABLED_GAMES = ["cs2"]

# Override DB path for edge bot
_EDGE_DB = os.environ.get("EDGE_BOT_DB", "data/edge_trades.db")

from bot import EsportsBot  # noqa: E402
from dashboard import create_app, push_state  # noqa: E402

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Pure Edge Bot — Latency Arbitrage (No AI)")
    parser.add_argument("--live", action="store_true", help="Enable live trading (real money)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    parser.add_argument("--port", type=int, default=int(os.environ.get("DASHBOARD_PORT", "8082")))
    args = parser.parse_args()

    dry_run = not args.live
    config.DASHBOARD_PORT = args.port

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.LOG_LEVEL, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    print("\n" + "=" * 60)
    print("  ⚡ PURE EDGE BOT — Speed-Only Latency Arbitrage ⚡")
    print("=" * 60)
    print(f"  Mode:        {'LIVE 🔴' if not dry_run else 'DRY-RUN 🟢'}")
    _claude_status = f"ENABLED ({config.EDGE_CLAUDE_MODEL}) 🧠" if config.ANTHROPIC_API_KEY else "DISABLED ❌"
    print(f"  Claude AI:   {_claude_status}")
    print(f"  Min Edge:    {config.EDGE_BOT_MIN_EDGE * 100:.0f}%")
    print(f"  Bet Size:    ${config.EDGE_BOT_BET_SIZE:.0f}")
    print(f"  Take Profit: +{config.TAKE_PROFIT_PCT * 100:.1f}%")
    print(f"  Stop Loss:   -{config.STOP_LOSS_PCT * 100:.1f}%")
    print(f"  Timeout:     {config.EXIT_TIMEOUT_SECONDS:.0f}s")
    print(f"  Dashboard:   http://localhost:{args.port}")
    print(f"  Database:    {_EDGE_DB}")
    print("=" * 60 + "\n")

    # Create bot with separate DB
    bot = EsportsBot(dry_run=dry_run)
    # Override executor's DB — persists trades across restarts
    from persistence import TradeDB
    bot.executor.db = TradeDB(db_path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), _EDGE_DB
    ))
    # Reload state from DB (balance, positions, trade count)
    bot.executor.positions = []
    bot.executor._trade_count = 0
    bot.executor._load_from_db()
    # If DB had no saved state (fresh), use starting balance
    if not bot.executor.db.load_state():
        bot.executor.balance = config.STARTING_BALANCE
        bot.executor.total_fees_paid = 0.0

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Edge bot shutting down...")
        bot._running = False
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start dashboard (edge-specific template)
    if not args.no_dashboard:
        app = create_app(bot)

        # Override the default "/" route with edge-specific dashboard
        app.view_functions['index'] = lambda: app.jinja_env.get_template("edge_dashboard.html").render()

        from waitress import serve
        dash_thread = threading.Thread(
            target=serve,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": args.port, "threads": 2, "_quiet": True},
            daemon=True,
        )
        dash_thread.start()
        logger.info(f"Edge bot dashboard: http://localhost:{args.port}")

    # Run bot (blocking)
    bot.run_blocking()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        logging.getLogger(__name__).critical(f"FATAL: {e}", exc_info=True)
        sys.exit(1)  # Non-zero exit triggers auto-restart wrapper
