"""
Esports Latency Arbitrage Bot for Polymarket.

Detects latency gaps between real-time esports game feeds (HLTV, Riot API, etc.)
and Polymarket order book updates, then places bets during the mispricing window.

Usage:
    python main.py                  # Dry-run with dashboard
    python main.py --live           # Live trading (requires credentials)
    python main.py --no-dashboard   # No web UI
    python main.py --port 8083      # Custom dashboard port

Architecture:
    Game Feeds (CS2/LoL/Dota2/Val) → Latency Analyzer → Trade Signals → Executor
                                                                           ↓
                                                               Polymarket CLOB
"""
import argparse
import logging
import os
import signal
import sys
import threading

# Prevent Flask from calling os.getcwd() via dotenv in sandboxed environments
os.environ.setdefault("FLASK_SKIP_DOTENV", "1")

import config
from bot import EsportsBot
from dashboard import create_app

# ─── Logging ─────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Esports Latency Arbitrage Bot")
    parser.add_argument("--live", action="store_true", help="Enable live trading (default: dry-run)")
    parser.add_argument("--no-dashboard", action="store_true", help="Disable web dashboard")
    parser.add_argument("--port", type=int, default=config.DASHBOARD_PORT, help="Dashboard port")
    args = parser.parse_args()

    dry_run = not args.live

    logger.info("=" * 60)
    logger.info("  ESPORTS LATENCY ARBITRAGE BOT")
    logger.info(f"  Powered by Claude AI Expert Panel (11 experts)")
    logger.info(f"  Mode: {'DRY-RUN (simulated)' if dry_run else 'LIVE TRADING'}")
    logger.info(f"  Games: {', '.join(config.ENABLED_GAMES)}")
    logger.info(f"  Panel confidence threshold: {config.MIN_CONFIDENCE}")
    logger.info(f"  Bet range: ${config.MIN_BET} - ${config.MAX_BET}")
    logger.info("=" * 60)

    if not dry_run:
        logger.warning("LIVE TRADING MODE — real money at risk!")
        if not config.POLYMARKET_PRIVATE_KEY or config.POLYMARKET_PRIVATE_KEY.startswith("0xYOUR"):
            logger.error("Set POLYMARKET_PRIVATE_KEY in .env for live trading")
            sys.exit(1)

    # Create bot
    bot = EsportsBot(dry_run=dry_run)

    # Graceful shutdown
    def shutdown(signum, frame):
        logger.info("Shutting down...")
        bot.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    # Start dashboard in a background thread (Flask/waitress)
    if not args.no_dashboard:
        app = create_app(bot)
        logger.info(f"Dashboard: http://localhost:{args.port}")
        from waitress import serve
        import threading
        dash_thread = threading.Thread(
            target=serve,
            args=(app,),
            kwargs={"host": "0.0.0.0", "port": args.port, "threads": 4},
            daemon=True,
        )
        dash_thread.start()

    # Run bot in main thread (asyncio event loop lives here)
    bot.run_blocking()


if __name__ == "__main__":
    main()
