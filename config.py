"""
Configuration for the Esports Latency Arbitrage Bot.
All settings loaded from environment variables with sensible defaults.
"""
import os
from dotenv import load_dotenv

# override=False so shell-set env vars beat .env defaults. Needed for the
# Claude-tier shootout which sets OLLAMA_URL="" + EDGE_CLAUDE_MODEL="..."
# on the command line to route the analyst through the Anthropic API instead
# of the local Ollama path. Without this flag, the .env would reinstate the
# local-LLM settings and every Claude shootout run would silently fall back
# to Ollama.
load_dotenv(override=False)

# ─── Polymarket ──────────────────────────────────────────────────────────────
CLOB_API_URL = os.getenv("CLOB_API_URL", "https://clob.polymarket.com")
GAMMA_API_URL = os.getenv("GAMMA_API_URL", "https://gamma-api.polymarket.com")
CHAIN_ID = int(os.getenv("CHAIN_ID", "137"))  # Polygon

POLYMARKET_PRIVATE_KEY = os.getenv("POLYMARKET_PRIVATE_KEY", "")
POLYMARKET_FUNDER_ADDRESS = os.getenv("POLYMARKET_FUNDER_ADDRESS", "")
POLYMARKET_SIGNATURE_TYPE = int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "1"))

# ─── Trading ─────────────────────────────────────────────────────────────────
MIN_BET = float(os.getenv("MIN_BET", "10.0"))
MAX_BET = float(os.getenv("MAX_BET", "100.0"))
BASE_BET = float(os.getenv("BASE_BET", "15.0"))
MIN_CONFIDENCE = float(os.getenv("MIN_CONFIDENCE", "0.45"))  # Claude AI panel threshold (0-1)
STARTING_BALANCE = float(os.getenv("STARTING_BALANCE", "1000.0"))

# Max concurrent positions across all games
MAX_OPEN_POSITIONS = int(os.getenv("MAX_OPEN_POSITIONS", "25"))

# ─── Auto Exit (Latency Arb Mode) ──────────────────────────────────────────
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.10"))      # +10% → sell
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.05"))          # -5% → cut
EXIT_TIMEOUT_SECONDS = float(os.getenv("EXIT_TIMEOUT_SECONDS", "90"))  # 90s max hold
POSITION_MONITOR_INTERVAL = float(os.getenv("POSITION_MONITOR_INTERVAL", "3"))  # check every 3s

# ─── Edge Bot (no-Claude fast path) ────────────────────────────────────────
EDGE_BOT_MIN_EDGE = float(os.getenv("EDGE_BOT_MIN_EDGE", "0.03"))     # 3% min edge to trade
EDGE_BOT_BET_SIZE = float(os.getenv("EDGE_BOT_BET_SIZE", "25.0"))     # flat $25 per trade

# ─── Edge Analysis (Local LLM or Claude API) ─────────────────────────────
EDGE_CLAUDE_MODEL = os.getenv("EDGE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
EDGE_CLAUDE_MAX_CALLS = int(os.getenv("EDGE_CLAUDE_MAX_CALLS", "30"))  # per match
OLLAMA_URL = os.getenv("OLLAMA_URL", "")  # e.g. http://192.168.76.196:11434
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:7b")

# ─── Limit Order Settings ──────────────────────────────────────────────────
LIMIT_BID_MARGIN = float(os.getenv("LIMIT_BID_MARGIN", "0.05"))       # bid 5% below model price
LIMIT_ORDER_TIMEOUT = float(os.getenv("LIMIT_ORDER_TIMEOUT", "30"))    # cancel after 30s
LIMIT_FILL_THRESHOLD = float(os.getenv("LIMIT_FILL_THRESHOLD", "0.02"))  # fill if market within 2%

# ─── Latency Thresholds ─────────────────────────────────────────────────────
# Minimum edge window (seconds) — don't trade if latency gap < this
MIN_LATENCY_EDGE_SECONDS = float(os.getenv("MIN_LATENCY_EDGE_SECONDS", "2.0"))
# Maximum staleness (seconds) — if Polymarket odds haven't moved in this long, suspect stale
MAX_ODDS_STALENESS_SECONDS = float(os.getenv("MAX_ODDS_STALENESS_SECONDS", "30.0"))

# ─── CS2 Specific ────────────────────────────────────────────────────────────
CS2_ONLY_MODE = os.getenv("CS2_ONLY_MODE", "false").lower() == "true"
CS2_CT_WIN_RATE = float(os.getenv("CS2_CT_WIN_RATE", "0.53"))
CS2_PISTOL_WIN_RATE = float(os.getenv("CS2_PISTOL_WIN_RATE", "0.50"))
CS2_ECO_THRESHOLD = int(os.getenv("CS2_ECO_THRESHOLD", "5000"))     # <$5k = true eco (pistol only)
CS2_FORCE_THRESHOLD = int(os.getenv("CS2_FORCE_THRESHOLD", "20000"))  # <$20k = force
CS2_DISCOVERY_INTERVAL = int(os.getenv("CS2_DISCOVERY_INTERVAL", "60"))

# ─── Game Feeds ──────────────────────────────────────────────────────────────
if CS2_ONLY_MODE:
    ENABLED_GAMES = ["cs2"]
else:
    ENABLED_GAMES = os.getenv("ENABLED_GAMES", "cs2,dota2,lol,valorant").split(",")

# HLTV (CS2)
HLTV_SCOREBOT_URL = os.getenv("HLTV_SCOREBOT_URL", "wss://scorebot-secure.hltv.org/socket.io/?EIO=3&transport=websocket")
HLTV_MATCHES_URL = os.getenv("HLTV_MATCHES_URL", "https://www.hltv.org/matches")
HLTV_REQUEST_DELAY = float(os.getenv("HLTV_REQUEST_DELAY", "2.0"))

# PandaScore (commercial multi-game feed — not used, free feeds only)
PANDASCORE_TOKEN = os.getenv("PANDASCORE_TOKEN", "")
PANDASCORE_BASE_URL = os.getenv("PANDASCORE_BASE_URL", "https://api.pandascore.co")

# ─── Claude AI Analyst ───────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
USE_CLAUDE_ANALYST = os.getenv("USE_CLAUDE_ANALYST", "true").lower() == "true"
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-6")
CLAUDE_MAX_CALLS_PER_MATCH = int(os.getenv("CLAUDE_MAX_CALLS_PER_MATCH", "30"))
CLAUDE_MIN_EDGE_FOR_CALL = float(os.getenv("CLAUDE_MIN_EDGE_FOR_CALL", "0.03"))

# ─── Steam (Dota2 Game Coordinator) ──────────────────────────────────────────
STEAM_API_KEY = os.getenv("STEAM_API_KEY", "")

# ─── Reference Odds ──────────────────────────────────────────────────────────
ODDS_API_KEY = os.getenv("ODDS_API_KEY", "")

# ─── Dashboard ───────────────────────────────────────────────────────────────
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "7777"))

# ─── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
