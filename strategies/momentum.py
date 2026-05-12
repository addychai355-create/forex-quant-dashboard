"""
Momentum + Volatility Filter Strategy

Core logic:
- Buy when short-term MA crosses above medium-term MA
- MACD momentum confirmation
- Volatility filter via ATR
- RSI overbought/oversold exit

Works with OR without TA-Lib (uses pandas rolling if TA-Lib unavailable).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

# Try TA-Lib, fall back to pandas implementation
try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


def _sma(values, period):
    """Simple Moving Average."""
    if HAS_TALIB:
        return talib.SMA(values, timeperiod=period)
    return pd.Series(values).rolling(period).mean().values


def _ema(values, period):
    """Exponential Moving Average."""
    if HAS_TALIB:
        return talib.EMA(values, timeperiod=period)
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def _rsi(values, period=14):
    """Relative Strength Index."""
    if HAS_TALIB:
        return talib.RSI(values, timeperiod=period)
    series = pd.Series(values)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi.values


def _macd(values, fast=12, slow=26, signal=9):
    """MACD."""
    if HAS_TALIB:
        macd, macd_signal, macd_hist = talib.MACD(values, fast, slow, signal)
        return macd, macd_signal, macd_hist
    ema_fast = _ema(values, fast)
    ema_slow = _ema(values, slow)
    macd = ema_fast - ema_slow
    signal_line = _ema(macd, signal)
    hist = macd - signal_line
    return macd, signal_line, hist


def _atr(high, low, close, period=14):
    """Average True Range."""
    if HAS_TALIB:
        return talib.ATR(high, low, close, timeperiod=period)
    high, low, close = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean().values


def _bbands(values, period=20, nbdev=2):
    """Bollinger Bands."""
    if HAS_TALIB:
        return talib.BBANDS(values, timeperiod=period, nbdevup=nbdev, nbdevdn=nbdev)
    series = pd.Series(values)
    sma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = sma + nbdev * std
    lower = sma - nbdev * std
    return upper.values, sma.values, lower.values


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLC DataFrame."""
    df = df.copy()
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values.astype(np.float64)

    # Moving averages
    df["ma_fast"] = _sma(close, 8)
    df["ma_mid"] = _sma(close, 21)
    df["ma_slow"] = _sma(close, 50)

    # EMA
    df["ema_fast"] = _ema(close, 12)
    df["ema_slow"] = _ema(close, 26)

    # MACD
    macd, macd_signal, macd_hist = _macd(close, 12, 26, 9)
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd_hist

    # RSI
    df["rsi"] = _rsi(close, 14)

    # ATR
    df["atr"] = _atr(high, low, close, 14)
    df["atr_pct"] = df["atr"] / close * 100

    # Bollinger Bands
    upper, mid, lower = _bbands(close, 20, 2)
    df["bb_upper"] = upper
    df["bb_mid"] = mid
    df["bb_lower"] = lower
    df["bb_width"] = (upper - lower) / mid * 100

    return df


def generate_signals(
    df: pd.DataFrame,
    atr_min_pct: float = 0.05,
    atr_max_pct: float = 1.0,
    rsi_oversold: float = 30.0,
    rsi_overbought: float = 70.0,
    use_macd_filter: bool = True,
) -> pd.DataFrame:
    """
    Generate trading signals from indicators.
    Returns df with added 'signal' column: 1 = long, -1 = short, 0 = flat.
    """
    df = df.copy()
    df["signal"] = 0

    if len(df) < 60:
        return df

    # Core momentum entry conditions
    bull_trend = df["ma_fast"] > df["ma_mid"]
    macd_bull = (df["macd_hist"] > 0) & (df["macd_hist"].shift(1) <= 0)
    price_strong = (
        (df["close"] > df["ma_fast"]) &
        (df["close"] > df["ma_mid"]) &
        (df["close"] > df["ma_slow"])
    )

    # Core momentum exit
    bear_trend = df["ma_fast"] < df["ma_mid"]
    macd_bear = (df["macd_hist"] < 0) & (df["macd_hist"].shift(1) >= 0)

    # Volatility filter
    valid_vol = (df["atr_pct"] >= atr_min_pct) & (df["atr_pct"] <= atr_max_pct)

    # RSI filter
    rsi_not_overbought = df["rsi"] < rsi_overbought

    # Long entry
    long_entry = bull_trend & valid_vol & price_strong & rsi_not_overbought
    if use_macd_filter:
        long_entry = long_entry & macd_bull

    # Long exit
    long_exit = bear_trend | macd_bear | (df["rsi"] > rsi_overbought + 10)

    # Apply signals
    df.loc[long_entry, "signal"] = 1
    df.loc[long_exit & (df["signal"].shift(1) == 1), "signal"] = 0

    # Forward-fill (hold between entries/exits)
    df["position"] = df["signal"].replace(0, np.nan).ffill().fillna(0)

    return df


def calculate_performance(df: pd.DataFrame, spread_cost: float = 0.0001) -> dict:
    """
    Calculate basic strategy metrics.
    spread_cost = estimated spread + commission in price units (~1 pip for EUR/USD)
    """
    df = df.copy()
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]

    # Subtract transaction costs on signal changes
    df["trades"] = df["position"].diff().abs().clip(0)
    df["strategy_returns"] -= df["trades"] * spread_cost / df["close"]

    total_return = (1 + df["strategy_returns"]).prod() - 1
    buy_hold_return = (1 + df["returns"]).prod() - 1

    sharpe = np.nan
    if df["strategy_returns"].std() > 0:
        sharpe = (
            df["strategy_returns"].mean()
            / df["strategy_returns"].std()
            * np.sqrt(252)
        )

    max_drawdown = _max_drawdown((1 + df["strategy_returns"]).cumprod())

    # Count actual trades (entry = position change from 0 to 1)
    pos = df["position"].values
    entries = np.where((pos[1:] == 1) & (pos[:-1] == 0))[0]
    num_trades = len(entries)

    win_rate = np.nan
    if num_trades > 0:
        trade_returns = []
        for entry_idx in entries:
            # Find exit after this entry
            exit_idx = np.where((pos[entry_idx + 1:] == 0))[0]
            if len(exit_idx) > 0:
                exit_idx = entry_idx + 1 + exit_idx[0]
                trade_return = df["close"].iloc[exit_idx] / df["close"].iloc[entry_idx] - 1
                trade_returns.append(trade_return)
            else:
                trade_returns.append(df["close"].iloc[-1] / df["close"].iloc[entry_idx] - 1)
        if trade_returns:
            win_rate = sum(1 for r in trade_returns if r > 0) / len(trade_returns)

    return {
        "total_return_pct": total_return * 100,
        "buy_hold_return_pct": buy_hold_return * 100,
        "sharpe_ratio": round(sharpe, 2),
        "max_drawdown_pct": max_drawdown * 100,
        "win_rate_pct": win_rate * 100 if not np.isnan(win_rate) else 0,
        "num_trades": num_trades,
        "exposure_pct": (df["position"] != 0).mean() * 100,
    }


def _max_drawdown(equity_curve: pd.Series) -> float:
    peak = equity_curve.expanding().max()
    dd = (equity_curve - peak) / peak
    return min(dd.min(), 0)


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data.fx_data import get_forex_data

    print(f"TA-Lib: {'✅ enabled' if HAS_TALIB else '❌ not available (using pandas fallback)'}")
    print("Loading EUR/USD daily data...")
    df = get_forex_data("EUR_USD", "1d", years_back=2)
    if df.empty:
        print("No data loaded.")
        exit(1)

    df = add_indicators(df)
    df = generate_signals(df)

    perf = calculate_performance(df)
    print("\n📊 Strategy Performance (EUR/USD Daily, 2yr)")
    for k, v in perf.items():
        print(f"   {k}: {v}")
