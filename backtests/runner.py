"""
Backtest Runner — VectorBT based backtesting

Usage:
    python backtests/runner.py --pair EUR_USD --tf 1h --years 2
    python backtests/runner.py --pair GBP_USD --tf 1h --years 3 --plot
    python backtests/runner.py --multi EUR_USD GBP_USD USD_JPY
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from data.fx_data import get_forex_data, AVAILABLE_PAIRS
from strategies.momentum import add_indicators, generate_signals

# Silence TensorFlow warnings if any
import warnings
warnings.filterwarnings("ignore")


def run_backtest(
    pair: str = "EUR_USD",
    tf: str = "1h",
    years_back: int = 2,
    plot: bool = False,
    spread_pips: float = 1.0,
    cash: float = 10_000,
) -> dict:
    """Run a backtest on the momentum strategy."""

    print(f"\n📥 Loading {pair} @ {tf} ({years_back}yr)...")
    df = get_forex_data(pair, tf, years_back=years_back, cache=True)
    if df.empty or len(df) < 100:
        print(f"❌ Not enough data for {pair} (got {len(df)} rows)")
        return {}

    df = df.sort_values("time").reset_index(drop=True)
    print(f"   Candles: {len(df):,} ({df['time'].min():%Y-%m-%d} → {df['time'].max():%Y-%m-%d})")

    # Generate signals
    df = add_indicators(df)
    df = generate_signals(df, use_macd_filter=True)

    # Vectorized backtest
    close = df["close"].values
    position = df["position"].values

    returns = np.diff(close) / close[:-1]
    strategy_returns = position[:-1] * returns

    # Transaction costs
    trades = np.abs(np.diff(position))
    spread_decimal = spread_pips * 0.0001 if "JPY" not in pair else spread_pips * 0.01
    strategy_returns -= trades * spread_decimal / close[:-1]

    # Equity curve
    equity = cash * np.cumprod(1 + np.concatenate([[0], strategy_returns]))

    # Metrics
    total_return = (equity[-1] / cash) - 1
    buy_hold_return = (close[-1] / close[0]) - 1

    sharpe = 0.0
    if strategy_returns.std() > 0:
        # Annualize based on data frequency
        ann_factor = 252 if tf == "1d" else 252 * 24 if "h" in tf else 252 * 24 * 60
        sharpe = round((strategy_returns.mean() / strategy_returns.std()) * np.sqrt(ann_factor), 2)

    # Max drawdown
    peak = np.maximum.accumulate(equity)
    dd = (equity - peak) / peak
    max_dd = dd.min()

    # Win rate
    trade_runs = strategy_returns[trades > 0]
    win_rate = (trade_runs > 0).mean() if len(trade_runs) > 2 else 0.0

    # Count trades: pair of entry (0→1) + exit (1→0)
    num_trades = max(int(np.sum(trades) / 2), 1 if int(np.sum(trades)) >= 1 else 0)
    exposure = np.mean(position != 0) * 100

    print(f"\n📊 Results for {pair} @ {tf}")
    print(f"   Total Return:    {total_return:+.2%}")
    print(f"   Buy & Hold:      {buy_hold_return:+.2%}")
    print(f"   Sharpe:          {sharpe}")
    print(f"   Max DD:          {max_dd:.2%}")
    print(f"   Win Rate:        {win_rate:.1%}")
    print(f"   Trades:          {num_trades}")
    print(f"   Exposure:        {exposure:.1f}%")

    if plot:
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)

            axes[0].plot(df["time"], close, label=f"{pair} Close", alpha=0.7)
            axes[0].set_title(f"{pair} @ {tf} — Momentum Strategy")
            axes[0].legend()
            axes[0].grid(alpha=0.3)

            axes[1].plot(df["time"][1:], strategy_returns, label="Strategy Returns", alpha=0.5)
            axes[1].axhline(0, color="black", linewidth=0.5)
            axes[1].legend()
            axes[1].grid(alpha=0.3)

            axes[2].plot(equity, label="Equity", color="green")
            axes[2].fill_between(range(len(equity)), equity, cash, alpha=0.1, color="green")
            axes[2].axhline(cash, color="gray", linestyle="--", alpha=0.5)
            axes[2].legend()
            axes[2].grid(alpha=0.3)

            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"   Plot error: {e}")

    return {
        "pair": pair,
        "tf": tf,
        "total_return": round(total_return * 100, 2),
        "buy_hold_return": round(buy_hold_return * 100, 2),
        "sharpe": sharpe,
        "max_dd": round(max_dd * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "trades": num_trades,
        "exposure_pct": round(exposure, 1),
    }


def run_multi_backtest(pairs: list[str], tf: str = "1h", years: int = 2):
    """Run backtest across multiple pairs and compare."""
    all_results = []
    for pair in pairs:
        print(f"\n{'='*55}")
        r = run_backtest(pair, tf, years, plot=False)
        if r:
            all_results.append(r)
        print(f"{'='*55}")

    if all_results:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values("sharpe", ascending=False)
        print(f"\n🏆 Multi-Pair Summary ({tf}, {years}yr)")
        print(results_df.to_string(index=False))
        out_path = Path("logs") / f"multi_{tf}_{years}yr.csv"
        out_path.parent.mkdir(exist_ok=True)
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved to {out_path}")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Forex Quant Backtest")
    parser.add_argument("--pair", default="EUR_USD", help="Forex pair")
    parser.add_argument("--tf", default="1h", help="Timeframe: 1m, 5m, 15m, 30m, 1h, 4h, 1d")
    parser.add_argument("--years", type=int, default=2, help="Years of history")
    parser.add_argument("--plot", action="store_true", help="Show plot")
    parser.add_argument("--multi", nargs="*", help="Pairs for multi-run (e.g. EUR_USD GBP_USD)")
    parser.add_argument("--spread", type=float, default=1.0, help="Spread in pips")
    parser.add_argument("--cash", type=float, default=10_000, help="Starting capital")
    parser.add_argument("--list-pairs", action="store_true", help="List available pairs")

    args = parser.parse_args()

    if args.list_pairs:
        print("Available pairs:")
        for p in AVAILABLE_PAIRS:
            print(f"  • {p}")
        sys.exit(0)

    if args.multi:
        run_multi_backtest(args.multi, args.tf, args.years)
    else:
        run_backtest(args.pair, args.tf, args.years, args.plot, args.spread, args.cash)
