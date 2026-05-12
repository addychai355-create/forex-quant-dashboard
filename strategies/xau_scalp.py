"""
Gold Scalping Strategy - XAU/USD

Optimized for 1m-5m charts with 5-15 minute hold times.
Focuses on micro-momentum and mean reversion in gold's volatile moves.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


def _ema(values, period):
    if HAS_TALIB: return talib.EMA(values.astype(float), timeperiod=period)
    return pd.Series(values).ewm(span=period, adjust=False).mean().values


def _sma(values, period):
    if HAS_TALIB: return talib.SMA(values.astype(float), timeperiod=period)
    return pd.Series(values).rolling(period).mean().values


def _rsi(values, period=7):
    if HAS_TALIB: return talib.RSI(values.astype(float), timeperiod=period)
    series = pd.Series(values)
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).values


def _macd(values, fast=6, slow=13, signal=5):
    if HAS_TALIB: return talib.MACD(values.astype(float), fast, slow, signal)
    ema_f, ema_s = _ema(values, fast), _ema(values, slow)
    macd = ema_f - ema_s
    sig = _ema(macd, signal)
    return macd, sig, macd - sig


def _atr(high, low, close, period=10):
    if HAS_TALIB: return talib.ATR(high.astype(float), low.astype(float), close.astype(float), timeperiod=period)
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean().values


def _stoch(high, low, close, k=5, d=3):
    if HAS_TALIB: return talib.STOCH(high.astype(float), low.astype(float), close.astype(float),
                                      fastk_period=k, slowk_period=d, slowd_period=d)
    low_k = pd.Series(low).rolling(k).min()
    high_k = pd.Series(high).rolling(k).max()
    k_vals = 100 * (pd.Series(close) - low_k) / (high_k - low_k).replace(0, np.nan)
    return k_vals.values, k_vals.rolling(d).mean().values


def add_indicators_xau(df: pd.DataFrame) -> pd.DataFrame:
    """Add scalping indicators for XAU/USD."""
    df = df.copy()
    close = df["close"].values.astype(float)
    high = df["high"].values.astype(float)
    low = df["low"].values.astype(float)
    volume = df["volume"].values.astype(float)

    # Fast EMAs
    df["ema_5"] = _ema(close, 5)
    df["ema_8"] = _ema(close, 8)
    df["ema_13"] = _ema(close, 13)
    df["ema_21"] = _ema(close, 21)

    # MACD (faster)
    macd, macd_sig, macd_hist = _macd(close, 6, 13, 5)
    df["macd"] = macd
    df["macd_signal"] = macd_sig
    df["macd_hist"] = macd_hist

    # RSI (faster)
    df["rsi"] = _rsi(close, 7)

    # Stochastic
    df["stoch_k"], df["stoch_d"] = _stoch(high, low, close, 5, 3)

    # ATR
    df["atr"] = _atr(high, low, close, 10)
    df["atr_pct"] = df["atr"] / close * 100

    # Price delta rankings
    df["price_change"] = df["close"].pct_change()
    df["price_rank_5"] = df["price_change"].rolling(5).apply(
        lambda x: (x.iloc[-1] > 0 and x.iloc[-1] >= x.quantile(0.8)) or
                  (x.iloc[-1] < 0 and x.iloc[-1] <= x.quantile(0.2)),
        raw=False
    )

    # Volume confirmation
    df["volume_ma"] = _sma(volume, 20)
    df["volume_ratio"] = volume / df["volume_ma"].replace(0, np.nan)

    # Momentum score (composite)
    df["mom_score"] = 0.0
    df["mom_score"] += (df["ema_5"] > df["ema_8"]).astype(float) * 0.2
    df["mom_score"] += (df["ema_8"] > df["ema_13"]).astype(float) * 0.15
    df["mom_score"] += (df["ema_13"] > df["ema_21"]).astype(float) * 0.15
    df["mom_score"] += ((df["macd_hist"] > 0) & (df["macd_hist"] > df["macd_hist"].shift(1))).astype(float) * 0.2
    df["mom_score"] += (df["rsi"] > 50).astype(float) * 0.15
    df["mom_score"] += (df["close"] > df["ema_8"]).astype(float) * 0.15

    df["mom_score_rev"] = 0.0
    df["mom_score_rev"] += (df["ema_5"] < df["ema_8"]).astype(float) * 0.2
    df["mom_score_rev"] += (df["ema_8"] < df["ema_13"]).astype(float) * 0.15
    df["mom_score_rev"] += (df["ema_13"] < df["ema_21"]).astype(float) * 0.15
    df["mom_score_rev"] += ((df["macd_hist"] < 0) & (df["macd_hist"] < df["macd_hist"].shift(1))).astype(float) * 0.2
    df["mom_score_rev"] += (df["rsi"] < 50).astype(float) * 0.15
    df["mom_score_rev"] += (df["close"] < df["ema_8"]).astype(float) * 0.15

    return df


def generate_signals_xau(
    df: pd.DataFrame,
    mom_threshold: float = 0.55,       # Min momentum score to enter
    atr_min_pct: float = 0.02,         # Min volatility
    atr_max_pct: float = 0.40,         # Max volatility (avoid crazy moves)
    rsi_low: float = 35,
    rsi_high: float = 65,
    min_vol_ratio: float = 1.0,
    atr_sl_mult: float = 0.8,          # Stop loss as ATR multiple
    atr_tp_mult: float = 1.2,          # Take profit as ATR multiple
    max_hold_bars: int = 15,           # Max hold in bars
    trail_start: int = 3,              # Start trailing after N bars
) -> pd.DataFrame:
    """
    Generate scalping signals with proper SL/TP simulation.
    """
    df = df.copy()
    df["signal"] = 0
    df["position"] = 0
    df["entry_price"] = np.nan
    df["sl_price"] = np.nan
    df["tp_price"] = np.nan
    df["exit_reason"] = ""

    if len(df) < 60:
        return df

    atr = df["atr"].values
    close = df["close"].values
    rsi = df["rsi"].values

    # Valid volatility zone
    valid_vol = (df["atr_pct"] >= atr_min_pct) & (df["atr_pct"] <= atr_max_pct)

    # Potential entries (raw signals without position management)
    raw_long = (
        (df["mom_score"] >= mom_threshold) &
        valid_vol &
        (rsi < rsi_high) &
        (df["volume_ratio"] >= min_vol_ratio)
    )

    raw_short = (
        (df["mom_score_rev"] >= mom_threshold) &
        valid_vol &
        (rsi > (100 - rsi_high)) &
        (df["volume_ratio"] >= min_vol_ratio)
    )

    # Simulate trading with proper SL/TP
    pos = 0
    entry_bar = 0
    entry_px = 0.0
    sl_px = 0.0
    tp_px = 0.0
    direction = 0  # 1=long, -1=short

    for i in range(len(df)):
        if pos == 0:
            # ─── LOOK FOR ENTRY ───
            if raw_long.iloc[i]:
                pos = 1
                direction = 1
                entry_bar = i
                entry_px = close[i]
                sl_px = entry_px - atr[i] * atr_sl_mult
                tp_px = entry_px + atr[i] * atr_tp_mult
                df.loc[df.index[i], "signal"] = 1
                df.loc[df.index[i], "entry_price"] = entry_px
                df.loc[df.index[i], "sl_price"] = sl_px
                df.loc[df.index[i], "tp_price"] = tp_px
            elif raw_short.iloc[i]:
                pos = -1
                direction = -1
                entry_bar = i
                entry_px = close[i]
                sl_px = entry_px + atr[i] * atr_sl_mult
                tp_px = entry_px - atr[i] * atr_tp_mult
                df.loc[df.index[i], "signal"] = -1
                df.loc[df.index[i], "entry_price"] = entry_px
                df.loc[df.index[i], "sl_price"] = sl_px
                df.loc[df.index[i], "tp_price"] = tp_px

        else:
            # ─── MANAGE POSITION ───
            bars_held = i - entry_bar

            # Trail stop
            if bars_held >= trail_start:
                if direction == 1:
                    trail_px = close[i] - atr[i] * atr_sl_mult * 0.5
                    if trail_px > sl_px:
                        sl_px = trail_px
                else:
                    trail_px = close[i] + atr[i] * atr_sl_mult * 0.5
                    if trail_px < sl_px:
                        sl_px = trail_px

            # Check exits
            exit_now = False
            reason = ""

            if direction == 1:
                if close[i] <= sl_px:
                    exit_now, reason = True, "stop_loss"
                elif close[i] >= tp_px:
                    exit_now, reason = True, "take_profit"
            else:
                if close[i] >= sl_px:
                    exit_now, reason = True, "stop_loss"
                elif close[i] <= tp_px:
                    exit_now, reason = True, "take_profit"

            if not exit_now and bars_held >= max_hold_bars:
                exit_now, reason = True, "timeout"

            # Reversal
            if not exit_now:
                if direction == 1 and raw_short.iloc[i]:
                    exit_now, reason = True, "reversal"
                elif direction == -1 and raw_long.iloc[i]:
                    exit_now, reason = True, "reversal"

            if exit_now:
                df.loc[df.index[i], "position"] = 0
                df.loc[df.index[i], "exit_reason"] = reason
                pos = 0
                direction = 0
            else:
                df.loc[df.index[i], "position"] = direction
                df.loc[df.index[i], "sl_price"] = sl_px
                df.loc[df.index[i], "tp_price"] = tp_px

    return df


def calculate_performance_xau(df: pd.DataFrame) -> dict:
    """Calculate scalping strategy metrics."""
    df = df.copy()

    pos_series = df["position"]
    close = df["close"].values

    # Simple return calculation per bar
    df["bar_return"] = df["close"].pct_change()

    # Entry returns
    entries = df[df["signal"] != 0].index
    exits = df[df["exit_reason"] != ""].index

    trade_returns = {}
    for e_idx, entry_idx in enumerate(entries):
        # Find the matching exit
        valid_exits = [x for x in exits if x > entry_idx]
        if valid_exits:
            exit_idx = valid_exits[0]
            ret = close[df.index.get_loc(exit_idx)] / close[df.index.get_loc(entry_idx)] - 1
            trade_returns[entry_idx] = {"exit": exit_idx, "return": ret, "hold": df.index.get_loc(exit_idx) - df.index.get_loc(entry_idx)}

    trade_returns_list = [v["return"] for v in trade_returns.values()]
    hold_times = [v["hold"] for v in trade_returns.values()]
    num_trades = len(trade_returns_list)

    # Overall returns
    df["strategy_returns"] = pos_series.shift(1) * df["bar_return"]
    total_return = (1 + df["strategy_returns"]).prod() - 1
    buy_hold_return = (1 + df["bar_return"]).prod() - 1

    # Sharpe
    sharpe = np.nan
    if df["strategy_returns"].std() > 0:
        bars_per_year = 252 * 24 * 60
        sharpe = round(df["strategy_returns"].mean() / df["strategy_returns"].std() * np.sqrt(bars_per_year), 2)

    # Max drawdown
    equity = (1 + df["strategy_returns"]).cumprod()
    peak = equity.expanding().max()
    dd = (equity - peak) / peak
    max_dd = dd.min()

    win_rate = sum(1 for r in trade_returns_list if r > 0) / num_trades * 100 if num_trades > 0 else 0
    avg_hold_bars = np.mean(hold_times) if hold_times else 0
    avg_trade_return = np.mean(trade_returns_list) * 100 if trade_returns_list else 0
    best_trade = max(trade_returns_list) * 100 if trade_returns_list else 0
    worst_trade = min(trade_returns_list) * 100 if trade_returns_list else 0

    exit_counts = df["exit_reason"].value_counts().to_dict()

    return {
        "total_return_pct": round(total_return * 100, 2),
        "buy_hold_return_pct": round(buy_hold_return * 100, 2),
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": round(max_dd * 100, 2),
        "win_rate_pct": round(win_rate, 1),
        "num_trades": num_trades,
        "avg_hold_bars": round(avg_hold_bars, 1),
        "avg_trade_pct": round(avg_trade_return, 3),
        "best_trade_pct": round(best_trade, 3),
        "worst_trade_pct": round(worst_trade, 3),
        "exposure_pct": round((pos_series != 0).mean() * 100, 1),
        "exit_reasons": {k: v for k, v in exit_counts.items() if k},
    }


# ──────────────────────────────────────────────
# Quick test
# ──────────────────────────────────────────────

if __name__ == "__main__":
    from data.fx_data import get_forex_data

    print("Loading XAU/USD 1m data...")
    df = get_forex_data("XAU_USD", "1m", years_back=0.02, cache=True)
    if df.empty or len(df) < 100:
        print("Trying 5m...")
        df = get_forex_data("XAU_USD", "5m", years_back=0.1, cache=True)

    if df.empty:
        print("No data.")
        exit(1)

    print(f"Loaded {len(df):,} candles ({df['time'].min():%m/%d %H:%M} → {df['time'].max():%m/%d %H:%M})")

    df = add_indicators_xau(df)
    df = generate_signals_xau(df)

    perf = calculate_performance_xau(df)
    print("\n📊 XAU/USD Scalping Performance:")
    for k, v in perf.items():
        if isinstance(v, dict):
            print(f"   {k}:", {kk: vv for kk, vv in v.items()})
        else:
            print(f"   {k}: {v}")

    # Show last signals
    signals = df[df["signal"] != 0].tail(10)
    if not signals.empty:
        print(f"\n🔔 Last {len(signals)} signals:")
        cols = ["time", "close", "rsi", "atr_pct", "signal", "sl_price", "tp_price", "exit_reason"]
        print(signals[[c for c in cols if c in signals.columns]].to_string(index=False))
