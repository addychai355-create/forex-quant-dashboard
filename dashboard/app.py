"""
Forex Quant Dashboard — Streamlit App
Monitor signals, performance, and live prices from anywhere.

Deploy to Streamlit Community Cloud for free:
  1. Push this folder to GitHub
  2. Go to https://streamlit.io/cloud
  3. Connect repo → Deploy
"""
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta, timezone

from data.fx_data import get_forex_data, AVAILABLE_PAIRS
from strategies.momentum import add_indicators, generate_signals, calculate_performance

st.set_page_config(
    page_title="Forex Quant Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Color scheme ───
COLORS = {
    "bg": "#0E1117",
    "card": "#1A1D23",
    "green": "#00C853",
    "red": "#FF1744",
    "blue": "#448AFF",
    "yellow": "#FFD600",
    "text": "#E0E0E0",
}

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .css-1r6slb0 { background-color: #1A1D23; }
    .metric-card {
        background: #1A1D23;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #2D3039;
    }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #9E9E9E; }
    .positive { color: #00C853; }
    .negative { color: #FF1744; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
st.sidebar.title("📊 Forex Monitor")
st.sidebar.markdown("---")

# Pair selector
pair = st.sidebar.selectbox("Pair", AVAILABLE_PAIRS, index=0)

# Timeframe
tf_options = {"1m": "1 Min", "5m": "5 Min", "15m": "15 Min", "30m": "30 Min",
              "1h": "1 Hour", "4h": "4 Hour", "1d": "1 Day"}
tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()),
                           format_func=lambda x: tf_options[x], index=4)

# Date range
years_back = st.sidebar.slider("History", 1, 5, 2)

st.sidebar.markdown("---")
st.sidebar.subheader("Strategy Params")
atr_min = st.sidebar.slider("Min ATR %", 0.01, 0.50, 0.05, 0.01)
use_macd = st.sidebar.checkbox("MACD Filter", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("Data: Yahoo Finance (free)")
st.sidebar.caption(f"Updated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
auto_refresh = st.sidebar.checkbox("Auto-refresh every 60s", value=False)

if auto_refresh:
    st.sidebar.info("🔄 Refreshing...")
    st.rerun(60)

# ─── Load Data ───
@st.cache_data(ttl=300)  # 5 min cache
def load_data(pr, tf_str, yrs):
    """Load forex data with caching."""
    df = get_forex_data(pr, tf_str, years_back=yrs, cache=True, source="yahoo")
    if df.empty or len(df) < 50:
        return None
    df = add_indicators(df)
    df = generate_signals(df, atr_min_pct=atr_min, use_macd_filter=use_macd)
    return df

@st.cache_data(ttl=300)
def load_all_pairs_data(tf_str, yrs):
    """Load latest data for all pairs (for overview)."""
    results = {}
    for p in AVAILABLE_PAIRS:
        try:
            df = get_forex_data(p, tf_str, years_back=yrs, cache=True, source="yahoo")
            if not df.empty and len(df) > 20:
                results[p] = df
        except Exception:
            continue
    return results

# ─── Main Dashboard ───

# Row 1: Live Prices Overview
st.subheader("💰 Live Prices Overview")

with st.spinner("Loading market data..."):
    all_data = load_all_pairs_data("1h", 1)

if all_data:
    cols = st.columns(4)
    for i, (p, df) in enumerate(sorted(all_data.items())):
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        change = latest["close"] - prev["close"]
        change_pct = change / prev["close"] * 100

        with cols[i % 4]:
            color = COLORS["green"] if change >= 0 else COLORS["red"]
            arrow = "▲" if change >= 0 else "▼"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{p.replace('_', '/')}</div>
                <div class="metric-value">{latest['close']:.5f}</div>
                <div style="color:{color}">
                    {arrow} {change:.5f} ({change_pct:+.3f}%)
                </div>
            </div>
            """, unsafe_allow_html=True)
else:
    st.warning("Could not load price data. Check internet connection.")

st.markdown("---")

# Row 2: Main Strategy Chart
st.subheader(f"📈 {pair.replace('_', '/')} — Strategy Analysis")

data = load_data(pair, tf, years_back)

if data is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Price + signals chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.55, 0.25, 0.20],
            subplot_titles=(f"{pair.replace('_', '/')} Price & Signals", "MACD", "RSI"),
        )

        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data["time"],
            open=data["open"],
            high=data["high"],
            low=data["low"],
            close=data["close"],
            name="Price",
            showlegend=False,
        ), row=1, col=1)

        # Buy/Sell markers
        buy_signals = data[data["signal"] == 1]
        fig.add_trace(go.Scatter(
            x=buy_signals["time"],
            y=buy_signals["close"],
            mode="markers",
            marker=dict(symbol="triangle-up", size=12, color=COLORS["green"]),
            name="Enter Long",
        ), row=1, col=1)

        # MAs
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["ma_fast"],
            line=dict(color=COLORS["blue"], width=1),
            name="MA-8",
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["ma_mid"],
            line=dict(color=COLORS["yellow"], width=1),
            name="MA-21",
        ), row=1, col=1)

        # MACD
        fig.add_trace(go.Bar(
            x=data["time"], y=data["macd_hist"],
            marker_color=np.where(data["macd_hist"] >= 0, COLORS["green"], COLORS["red"]),
            name="MACD Hist",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["macd"],
            line=dict(color=COLORS["blue"], width=1.5),
            name="MACD",
        ), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["macd_signal"],
            line=dict(color=COLORS["yellow"], width=1.5),
            name="Signal",
        ), row=2, col=1)

        # RSI
        fig.add_trace(go.Scatter(
            x=data["time"], y=data["rsi"],
            line=dict(color=COLORS["blue"], width=1.5),
            name="RSI",
        ), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], row=3, col=1)

        fig.update_layout(
            height=650,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", y=1.02, x=0),
        )
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Strategy metrics
        perf = calculate_performance(data)

        st.markdown("### 📊 Performance")
        metrics = [
            ("Return", f"{perf['total_return_pct']:+.2f}%", "positive" if perf['total_return_pct'] > 0 else "negative"),
            ("Buy & Hold", f"{perf['buy_hold_return_pct']:+.2f}%", "positive" if perf['buy_hold_return_pct'] > 0 else "negative"),
            ("Sharpe", f"{perf['sharpe_ratio']}", "positive" if perf['sharpe_ratio'] > 1 else "neutral" if perf['sharpe_ratio'] > 0 else "negative"),
            ("Max Drawdown", f"{perf['max_drawdown_pct']:.2f}%", "negative"),
            ("Win Rate", f"{perf['win_rate_pct']:.1f}%", "positive" if perf['win_rate_pct'] > 50 else "negative"),
            ("Trades", f"{perf['num_trades']}", "neutral"),
            ("Exposure", f"{perf['exposure_pct']:.1f}%", "neutral"),
        ]
        for label, value, cls in metrics:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #2D3039;">
                <span style="color:#9E9E9E;">{label}</span>
                <span class="{cls}" style="font-weight:600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Current signal
        latest_signal = data["signal"].iloc[-1]
        latest_position = data["position"].iloc[-1]
        latest_rsi = data["rsi"].iloc[-1]
        latest_atr = data["atr_pct"].iloc[-1]

        st.markdown("### 🔔 Current Status")
        signal_icon = "🟢" if latest_position == 1 else "🔴" if latest_position == -1 else "⚪"
        signal_text = "LONG" if latest_position == 1 else "SHORT" if latest_position == -1 else "FLAT"

        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div style="font-size:2rem;">{signal_icon}</div>
            <div style="font-size:1.5rem; font-weight:700;">{signal_text}</div>
            <div style="color:#9E9E9E;">RSI: {latest_rsi:.1f} | ATR%: {latest_atr:.3f}%</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error(f"Could not load data for {pair}. Try a different pair or timeframe.")

st.markdown("---")

# Row 3: Equity Curve + Drawdown
st.subheader("💰 Equity Curve")

if data is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        # Compute equity curve from signals
        df = data.copy()
        df["returns"] = df["close"].pct_change()
        df["strategy_returns"] = df["position"].shift(1) * df["returns"]
        df["trades"] = df["position"].diff().abs().clip(0)
        df["strategy_returns"] -= df["trades"] * 0.0001 / df["close"]
        df["equity"] = 10000 * (1 + df["strategy_returns"]).cumprod()
        df["buy_hold"] = 10000 * (1 + df["returns"]).cumprod()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=[0.7, 0.3],
        )

        fig.add_trace(go.Scatter(
            x=df["time"], y=df["equity"],
            line=dict(color=COLORS["green"], width=2),
            name="Strategy",
        ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=df["time"], y=df["buy_hold"],
            line=dict(color="#9E9E9E", width=1, dash="dash"),
            name="Buy & Hold",
        ), row=1, col=1)

        # Drawdown
        peak = df["equity"].expanding().max()
        dd = (df["equity"] - peak) / peak * 100
        fig.add_trace(go.Scatter(
            x=df["time"], y=dd,
            fill="tozeroy",
            line=dict(color=COLORS["red"], width=1),
            name="Drawdown %",
        ), row=2, col=1)

        fig.update_layout(
            height=400,
            template="plotly_dark",
            hovermode="x unified",
            margin=dict(l=0, r=0, t=10, b=0),
            legend=dict(orientation="h", y=1.02, x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📋 Recent Signals")
        sig_cols = ["time", "close", "rsi", "atr_pct", "position", "signal"]
        recent = data[sig_cols].tail(20).copy()
        recent["position"] = recent["position"].map({1: "LONG", 0: "FLAT", -1: "SHORT"})
        recent["signal"] = recent["signal"].map({1: "🟢 BUY", 0: "⚪", -1: "🔴 SELL"})
        recent = recent.rename(columns={
            "time": "Time", "close": "Price", "rsi": "RSI",
            "atr_pct": "ATR%", "position": "Pos", "signal": "Signal"
        })
        recent["Time"] = recent["Time"].dt.strftime("%m/%d %H:%M")
        st.dataframe(recent, use_container_width=True, hide_index=True)

st.markdown("---")

# Row 4: Multi-Pair Heatmap
st.subheader("🌍 Multi-Pair Comparison")

with st.spinner("Loading all pairs..."):
    comparison_data = {}
    for p in AVAILABLE_PAIRS:
        try:
            d = load_data(p, "1d", 2)
            if d is not None:
                perf = calculate_performance(d)
                comparison_data[p] = perf
        except Exception:
            continue

if comparison_data:
    comp_df = pd.DataFrame(comparison_data).T
    comp_df.index.name = "Pair"

    col1, col2 = st.columns([1, 2])

    with col1:
        metrics_select = st.selectbox("Metric", ["total_return_pct", "sharpe_ratio", "max_drawdown_pct", "win_rate_pct"])
        metric_labels = {
            "total_return_pct": "Total Return %",
            "sharpe_ratio": "Sharpe Ratio",
            "max_drawdown_pct": "Max Drawdown %",
            "win_rate_pct": "Win Rate %",
        }

        fig = px.bar(
            comp_df.sort_values(metrics_select, ascending=False),
            y=metrics_select,
            color=metrics_select,
            color_continuous_scale=["red", "yellow", "green"],
            title=f"{metric_labels[metrics_select]} by Pair",
            text_auto=".1f",
        )
        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📊 Comparison Table")
        display = comp_df[[
            "total_return_pct", "buy_hold_return_pct",
            "sharpe_ratio", "max_drawdown_pct",
            "win_rate_pct", "num_trades", "exposure_pct"
        ]].round(2)
        display.columns = [
            "Return%", "BH Return%", "Sharpe", "Max DD%",
            "Win Rate%", "Trades", "Exposure%"
        ]
        st.dataframe(display, use_container_width=True)

st.markdown("---")

# Footer
st.caption("""
**Forex Quant Monitor** — Data from Yahoo Finance | Strategy: Momentum + Volatility Filter
Built with Streamlit | Deploy free on streamlit.io/cloud
""")
