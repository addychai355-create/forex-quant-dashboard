"""
Forex Quant Trading — Configuration
"""
import os
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
LOGS_DIR = ROOT / "logs"

# Ensure directories exist
for d in [RAW_DIR, PROCESSED_DIR, LOGS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# === Trading Parameters ===
TIMEFRAMES = {
    "1m": "M1",
    "5m": "M5",
    "15m": "M15",
    "30m": "M30",
    "1h": "H1",
    "4h": "H4",
    "1d": "D1",
}

# Major forex pairs to watch
MAJOR_PAIRS = [
    "EUR_USD", "GBP_USD", "USD_JPY", "USD_CHF",
    "AUD_USD", "USD_CAD", "NZD_USD",
]

# Cross pairs
CROSS_PAIRS = [
    "EUR_GBP", "EUR_JPY", "GBP_JPY", "EUR_AUD",
    "AUD_JPY", "CHF_JPY", "EUR_CHF",
]

DEFAULT_PAIR = "EUR_USD"
DEFAULT_TF = "H1"

# === Risk Management ===
RISK_PER_TRADE = 0.01       # 1% risk per trade
MAX_SPREAD_PIPS = 2.0       # Max acceptable spread
STOP_LOSS_ATR_MULT = 1.5    # SL = ATR * this
TAKE_PROFIT_ATR_MULT = 3.0  # TP = ATR * this

# === Data Sources ===
# OANDA practice account — sign up free: https://www.oanda.com/demo-account/
OANDA_KEY = os.getenv("OANDA_API_KEY", "")
OANDA_ACCOUNT = os.getenv("OANDA_ACCOUNT_ID", "")
OANDA_ENV = os.getenv("OANDA_ENVIRONMENT", "practice")

# === Mode ===
DRY_RUN = True  # paper trade until you're confident
