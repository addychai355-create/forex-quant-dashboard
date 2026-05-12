"""
Forex Data Pipeline
====================
Sources (in order of preference):
  1. yfinance        — free, no API key, works out of the box
  2. Dukascopy       — free historical tick data (BI5 format)
  3. OANDA API       — live/practice account, needs API key
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR, OANDA_KEY, OANDA_ACCOUNT, OANDA_ENV

# ──────────────────────────────────────────────
# YAHOO FINANCE — Simplest, most reliable free source
# ──────────────────────────────────────────────

# Yahoo ticker format for forex: EURUSD=X
YAHOO_PAIRS = {
    # Forex pairs
    "EUR_USD": "EURUSD=X",
    "GBP_USD": "GBPUSD=X",
    "USD_JPY": "USDJPY=X",
    "USD_CHF": "USDCHF=X",
    "AUD_USD": "AUDUSD=X",
    "USD_CAD": "USDCAD=X",
    "NZD_USD": "NZDUSD=X",
    "EUR_GBP": "EURGBP=X",
    "EUR_JPY": "EURJPY=X",
    "GBP_JPY": "GBPJPY=X",
    "EUR_AUD": "EURAUD=X",
    "AUD_JPY": "AUDJPY=X",
    "CHF_JPY": "CHFJPY=X",
    "EUR_CHF": "EURCHF=X",
    # Commodities
    "XAU_USD": "GC=F",       # Gold Futures (~= spot XAU/USD)
    "XAG_USD": "SI=F",       # Silver Futures
    "BTC_USD": "BTC-USD",    # Bitcoin
}

TIMEFRAMES_YAHOO = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "60m", "1d": "1d",
}


def get_yahoo_data(
    pair: str = "EUR_USD",
    tf: str = "1h",
    years_back: int = 2,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Download forex data from Yahoo Finance.
    Uses yfinance library — no API key needed.
    
    Timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d
    Max history varies by timeframe (1m = 7 days, 1h = 730 days, 1d = max)
    """
    import yfinance as yf

    ticker = YAHOO_PAIRS.get(pair, pair.replace("_", "") + "=X")
    yahoo_tf = TIMEFRAMES_YAHOO.get(tf, tf)
    cache_file = RAW_DIR / f"yahoo_{ticker}_{tf}.parquet"

    # Try cache first (handle missing pyarrow gracefully)
    if cache:
        try:
            if cache_file.exists():
                df = pd.read_parquet(cache_file)
                last_dt = df["time"].max()
                if last_dt and pd.Timestamp.now(timezone.utc) - last_dt < timedelta(hours=2):
                    return df
        except Exception:
            pass  # cache read failed, re-download

    try:
        # Determine period and interval based on timeframe
        if tf in ("1m", "5m"):
            period = "7d" if tf == "1m" else "1mo"
        elif tf in ("15m", "30m"):
            period = "3mo"
        elif tf in ("1h",):
            period = "2y"  # Max for 60m
        elif tf in ("4h",):
            period = "max"  # Max for 60m too
            tf = "1h"
            yahoo_tf = "60m"
        else:
            period = f"{years_back}y"

        data = yf.download(
            tickers=ticker,
            period=period,
            interval=yahoo_tf,
            progress=False,
            auto_adjust=True,
        )

        if data.empty:
            print(f"  ⚠️  No data from Yahoo for {ticker}")
            return pd.DataFrame()

        # Flatten multi-level columns if needed
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)

        df = data.reset_index()
        df.columns = [c.lower().strip() for c in df.columns]

        # Rename columns
        col_map = {
            "datetime": "time", "datetime": "time",
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        }
        df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
        df["pair"] = pair

        # Ensure time column is datetime
        if "time" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "time"})
        if "time" not in df.columns and len(df.columns) > 0:
            # Try first column
            first_col = df.columns[0]
            df = df.rename(columns={first_col: "time"})

        df["time"] = pd.to_datetime(df["time"])
        df = df.sort_values("time").reset_index(drop=True)

        if cache:
            try:
                cache_file.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(cache_file, index=False)
            except Exception:
                pass

        return df

    except Exception as e:
        print(f"  ⚠️  Yahoo Finance error for {ticker}: {e}")
        return pd.DataFrame()


# ──────────────────────────────────────────────
# OANDA API — Live/Recent Data (needs API key)
# ──────────────────────────────────────────────

OANDA_BASE = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}


def _oanda_headers() -> dict:
    return {
        "Authorization": f"Bearer {OANDA_KEY}",
        "Content-Type": "application/json",
    }


def get_oanda_candles(
    pair: str = "EUR_USD",
    tf: str = "H1",
    count: int = 500,
) -> pd.DataFrame:
    """Fetch recent candles from OANDA REST API."""
    import requests

    if not OANDA_KEY:
        raise ValueError("OANDA_API_KEY not set. Add it to .env or use yfinance/Dukascopy.")

    base = OANDA_BASE.get(OANDA_ENV, OANDA_BASE["practice"])
    url = f"{base}/v3/instruments/{pair}/candles"
    params = {
        "granularity": tf,
        "count": min(count, 5000),
        "price": "M",
    }

    resp = requests.get(url, headers=_oanda_headers(), params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for c in data.get("candles", []):
        rows.append({
            "time": c["time"].replace("Z", "+00:00"),
            "open": float(c["mid"]["o"]),
            "high": float(c["mid"]["h"]),
            "low": float(c["mid"]["l"]),
            "close": float(c["mid"]["c"]),
            "volume": int(c["volume"]),
            "pair": pair,
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df["time"] = pd.to_datetime(df["time"])
    return df


def get_oanda_price(pair: str = "EUR_USD") -> dict:
    """Get current live price from OANDA."""
    import requests

    base = OANDA_BASE.get(OANDA_ENV, OANDA_BASE["practice"])
    url = f"{base}/v3/accounts/{OANDA_ACCOUNT}/pricing"
    params = {"instruments": pair}

    resp = requests.get(url, headers=_oanda_headers(), params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ──────────────────────────────────────────────
# DUKASCOPY — Free Historical Data (no API key)
# ──────────────────────────────────────────────

DUKASCOPY_SYMBOLS = {k.replace("_", ""): v for k, v in YAHOO_PAIRS.items()}
DUKASCOPY_SYMBOLS = {v.replace("=X", ""): v.replace("=X", "")
                     for v in YAHOO_PAIRS.values()}
# Build proper mapping
DUKASCOPY_SYMBOLS = {}
for our, yahoo in YAHOO_PAIRS.items():
    sym = yahoo.replace("=X", "")
    DUKASCOPY_SYMBOLS[our] = sym

TF_MAP = {
    "M1": 60, "M5": 300, "M15": 900, "M30": 1800,
    "H1": 3600, "H4": 14400, "D1": 86400,
}


def get_dukascopy_month(
    pair: str, year: int, month: int, cache: bool = True
) -> pd.DataFrame:
    """
    Download one month of 1-minute OHLC from Dukascopy.
    Returns DataFrame with time, open, high, low, close, volume.
    
    NOTE: Dukascopy may reject some requests — yfinance is more reliable.
    """
    import certifi
    import ssl
    import urllib.request, urllib.error

    sym = DUKASCOPY_SYMBOLS.get(pair, pair.replace("_", ""))
    cache_file = RAW_DIR / f"duka_{sym}_{year}_{month:02d}.parquet"
    if cache and cache_file.exists():
        return pd.read_parquet(cache_file)

    url = (
        f"https://data.dukascopy.com/datafeed/{sym}/"
        f"{year:04d}/{month-1:02d}/"
        f"{year}{month:02d}.zip"
    )

    ctx = ssl.create_default_context(cafile=certifi.where())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=30, context=ctx) as resp:
            data = resp.read()
    except (urllib.error.HTTPError, urllib.error.URLError) as e:
        print(f"  ⚠️  No Dukascopy data for {sym} {year}-{month:02d}: {e}")
        return pd.DataFrame()

    # Parse BI5 binary format
    import gzip, struct
    try:
        decompressed = gzip.decompress(data)
    except Exception:
        decompressed = data

    n_records = len(decompressed) // 20
    if n_records == 0:
        return pd.DataFrame()

    fmt = ">" + "i" * (n_records * 5)
    vals = struct.unpack(fmt, decompressed[: n_records * 20])
    arr = np.array(vals, dtype=np.int64).reshape(-1, 5)
    df = pd.DataFrame(arr, columns=["time_offset", "open", "close", "high", "low", "volume"])

    month_start = datetime(year, month, 1, tzinfo=timezone.utc)
    df["time"] = month_start + pd.to_timedelta(df["time_offset"], unit="s")

    is_jpy = sym.endswith("JPY")
    scale = 100.0 if is_jpy else 10000.0
    for col in ["open", "high", "low", "close"]:
        df[col] = df[col].astype(float) / scale

    df["pair"] = pair
    df = df[["time", "pair", "open", "high", "low", "close", "volume"]].sort_values("time")

    if cache:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_file, index=False)

    return df


# ──────────────────────────────────────────────
# Unified interface
# ──────────────────────────────────────────────

def get_forex_data(
    pair: str = "EUR_USD",
    tf: str = "1h",
    years_back: int = 2,
    source: str = "yahoo",
    cache: bool = True,
) -> pd.DataFrame:
    """
    Get forex OHLC data.
    
    Parameters
    ----------
    pair : str
        e.g. "EUR_USD", "GBP_USD", "USD_JPY"
    tf : str
        Timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d
        For Dukascopy: M1, M5, ..., D1
    years_back : int
        How many years of history
    source : str
        "yahoo" (default, free, no key) | "oanda" | "dukascopy"
    cache : bool
        Cache to parquet files
    """
    if source == "oanda":
        try:
            return get_oanda_candles(pair, tf.replace("1h", "H1").upper(), 5000)
        except Exception as e:
            print(f"OANDA error: {e}. Falling back to yahoo.")
            source = "yahoo"

    if source == "dukascopy":
        now = datetime.now(timezone.utc)
        start = now - timedelta(days=365 * years_back)
        # Map timeframe
        tf_map_rev = {v: k for k, v in TF_MAP.items()}
        duka_tf = tf_map_rev.get(int(tf.replace("h", "")) * 3600 if "h" in tf else
                                 int(tf.replace("m", "")) * 60 if "m" in tf else 86400, "H1")
        # Download monthly, resample later if needed
        all_dfs = []
        y, m = start.year, start.month
        while (y, m) <= (now.year, now.month):
            df = get_dukascopy_month(pair, y, m, cache=cache)
            if not df.empty:
                all_dfs.append(df)
            m += 1
            if m > 12:
                m = 1; y += 1
        if not all_dfs:
            return pd.DataFrame()
        df = pd.concat(all_dfs, ignore_index=True)
        df = df.drop_duplicates(subset=["time"]).sort_values("time")
        df = df[(df["time"] >= pd.Timestamp(start)) & (df["time"] <= pd.Timestamp(now))]
        # Resample if not M1
        if duka_tf != "M1" and duka_tf in TF_MAP:
            rule = f"{TF_MAP[duka_tf]}s"
            df = df.set_index("time").resample(rule).agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum", "pair": "last",
            }).dropna().reset_index()
        return df

    # Default: Yahoo
    return get_yahoo_data(pair, tf, years_back, cache=cache)


# ──────────────────────────────────────────────
# Available pairs & check
# ──────────────────────────────────────────────

AVAILABLE_PAIRS = list(YAHOO_PAIRS.keys())

def list_pairs():
    """Print all available forex pairs."""
    print("Available forex pairs:")
    for p in AVAILABLE_PAIRS:
        print(f"  • {p}")
    return AVAILABLE_PAIRS


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing Yahoo Finance forex downloader...")
    df = get_forex_data("EUR_USD", "1d", years_back=1)
    if not df.empty:
        print(f"✅ EUR/USD Daily: {len(df):,} candles")
        print(f"   Range: {df['time'].min():%Y-%m-%d} → {df['time'].max():%Y-%m-%d}")
        print(f"   Latest: {df[['time', 'close', 'volume']].tail(3).to_string(index=False)}")
    else:
        print("⚠️  No data. Trying Dukascopy as fallback...")
        df = get_forex_data("EUR_USD", "D1", years_back=1, source="dukascopy")
        if not df.empty:
            print(f"✅ Dukascopy EUR/USD: {len(df):,} candles")
        else:
            print("❌ Both sources failed. Check internet or use OANDA.")
