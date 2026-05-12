"""
Forex Quant Dashboard — Streamlit App
Monitor signals, performance, and live prices from anywhere.
Default focus: XAU/USD Gold Scalping (5m, 5-15 min holds)
"""
import sys
from pathlib import Path
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
from strategies.xau_scalp import add_indicators_xau, generate_signals_xau, calculate_performance_xau

st.set_page_config(
    page_title="XAU Scalp Monitor",
    page_icon="🥇",
    layout="wide",
    initial_sidebar_state="expanded",
)

COLORS = {"bg": "#0E1117", "card": "#1A1D23", "green": "#00C853",
          "red": "#FF1744", "blue": "#448AFF", "yellow": "#FFD600", "text": "#E0E0E0"}

st.markdown("""
<style>
    .stApp { background-color: #0E1117; }
    .css-1r6slb0 { background-color: #1A1D23; }
    .metric-card { background: #1A1D23; padding: 1rem; border-radius: 8px; border: 1px solid #2D3039; }
    .metric-value { font-size: 1.8rem; font-weight: 700; }
    .metric-label { font-size: 0.8rem; color: #9E9E9E; }
    .positive { color: #00C853; }
    .negative { color: #FF1744; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ───
st.sidebar.title("🥇 XAU Scalp Monitor")
st.sidebar.markdown("---")

pair = st.sidebar.selectbox("Instrument", AVAILABLE_PAIRS, index=AVAILABLE_PAIRS.index("XAU_USD"))

tf_options = {"1m": "1 Min", "5m": "5 Min", "15m": "15 Min", "30m": "30 Min",
              "1h": "1 Hour", "4h": "4 Hour", "1d": "1 Day"}
tf = st.sidebar.selectbox("Timeframe", list(tf_options.keys()),
                           format_func=lambda x: tf_options[x], index=1)  # default 5m

# Volume of data
if tf == "1m":
    default_days = 7
elif tf == "5m":
    default_days = 30
elif tf in ("15m", "30m"):
    default_days = 60
else:
    default_days = 90

days_back = st.sidebar.slider("Lookback (days)", 1, 180, default_days)

st.sidebar.markdown("---")
st.sidebar.subheader("Scalping Params")
mom_thresh = st.sidebar.slider("Mom Threshold", 0.30, 0.80, 0.55, 0.05)
sl_mult = st.sidebar.slider("SL (ATR mult)", 0.5, 2.0, 1.2, 0.1)
tp_mult = st.sidebar.slider("TP (ATR mult)", 1.0, 3.0, 2.0, 0.1)
max_hold = st.sidebar.slider("Max Hold (bars)", 2, 30, 4)

# Convert hold to minutes hint
hold_minutes = max_hold * (1 if tf == "1m" else 5 if tf == "5m" else 15 if tf == "15m" else 30)
st.sidebar.caption(f"≈ {hold_minutes} min max hold")

st.sidebar.markdown("---")
st.sidebar.caption(f"Data: Yahoo Finance (free)")
st.sidebar.caption(f"Updated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC")
auto_refresh = st.sidebar.checkbox("Auto-refresh 60s", value=False)

if auto_refresh:
    st.sidebar.info("🔄 Auto-refreshing...")
    st.rerun(60)

# ─── Load Data ───
@st.cache_data(ttl=120)
def load_data(pr, tf_str, days):
    df = get_forex_data(pr, tf_str, years_back=max(0.01, days/365), cache=True)
    if df.empty or len(df) < 60:
        return None
    # Use scalping strategy for gold, momentum for others
    if "XAU" in pr or "XAG" in pr:
        df = add_indicators_xau(df)
        df = generate_signals_xau(df, mom_threshold=mom_thresh, atr_sl_mult=sl_mult,
                                   atr_tp_mult=tp_mult, max_hold_bars=max_hold)
    else:
        from strategies.momentum import add_indicators, generate_signals
        df = add_indicators(df)
        df = generate_signals(df)
    return df

# ─── Main Dashboard ───
st.subheader("💰 Live Prices")

with st.spinner("Loading market data..."):
    key_pairs = ["XAU_USD", "EUR_USD", "GBP_USD", "USD_JPY", "XAG_USD"]
    cols = st.columns(len(key_pairs))
    for i, p in enumerate(key_pairs):
        try:
            d = get_forex_data(p, "5m", 0.02, cache=True)
            if d is not None and len(d) > 2:
                l = d.iloc[-1]; pv = d.iloc[-2]
                chg = (l["close"] - pv["close"]) / pv["close"] * 100
                arrow = "▲" if chg >= 0 else "▼"
                color = COLORS["green"] if chg >= 0 else COLORS["red"]
                label = "XAU/USD" if p == "XAU_USD" else p.replace("_", "/")
                with cols[i]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-weight:700;">🥇 {label}</div>
                        <div class="metric-value">{l['close']:.2f}</div>
                        <div style="color:{color}">{arrow} {chg:+.3f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
        except:
            pass

st.markdown("---")

# ─── Main Chart ───
is_gold = "XAU" in pair
asset_label = "XAU/USD Gold" if is_gold else pair.replace("_", "/")
st.subheader(f"📈 {asset_label} — {'Scalping' if is_gold else 'Momentum'} Strategy")

data = load_data(pair, tf, days_back)

if data is not None:
    col1, col2 = st.columns([2, 1])

    with col1:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            vertical_spacing=0.04, row_heights=[0.50, 0.25, 0.25],
                            subplot_titles=(f"{asset_label} Price & Signals", "MACD (Fast)", "RSI"))

        # Candlestick
        fig.add_trace(go.Candlestick(x=data["time"], open=data["open"], high=data["high"],
                                      low=data["low"], close=data["close"], name="Price",
                                      showlegend=False), row=1, col=1)

        # Buy/Sell signals
        buys = data[data["signal"] == 1]
        sells = data[data["signal"] == -1]
        if not buys.empty:
            fig.add_trace(go.Scatter(x=buys["time"], y=buys["close"],
                mode="markers", marker=dict(symbol="triangle-up", size=10, color=COLORS["green"]),
                name="🟢 Buy"), row=1, col=1)
        if not sells.empty:
            fig.add_trace(go.Scatter(x=sells["time"], y=sells["close"],
                mode="markers", marker=dict(symbol="triangle-down", size=10, color=COLORS["red"]),
                name="🔴 Sell"), row=1, col=1)

        # EMAs for gold, MAs for forex
        if is_gold:
            for col_name, label, color in [("ema_5", "EMA-5", "#00E5FF"), ("ema_8", "EMA-8", COLORS["blue"]),
                                            ("ema_13", "EMA-13", COLORS["yellow"]), ("ema_21", "EMA-21", "#FF9100")]:
                if col_name in data.columns:
                    fig.add_trace(go.Scatter(x=data["time"], y=data[col_name],
                        line=dict(color=color, width=1), name=label), row=1, col=1)
        else:
            for col_name, label, color in [("ma_fast", "MA-8", COLORS["blue"]), ("ma_mid", "MA-21", COLORS["yellow"])]:
                if col_name in data.columns:
                    fig.add_trace(go.Scatter(x=data["time"], y=data[col_name],
                        line=dict(color=color, width=1), name=label), row=1, col=1)

        # SL/TP lines
        if "sl_price" in data.columns:
            sl_data = data.dropna(subset=["sl_price"])
            if not sl_data.empty:
                fig.add_trace(go.Scatter(x=sl_data["time"], y=sl_data["sl_price"],
                    line=dict(color=COLORS["red"], width=0.5, dash="dot"), name="Stop Loss",
                    opacity=0.4), row=1, col=1)
            tp_data = data.dropna(subset=["tp_price"])
            if not tp_data.empty:
                fig.add_trace(go.Scatter(x=tp_data["time"], y=tp_data["tp_price"],
                    line=dict(color=COLORS["green"], width=0.5, dash="dot"), name="Take Profit",
                    opacity=0.4), row=1, col=1)

        # MACD
        if "macd" in data.columns:
            fig.add_trace(go.Bar(x=data["time"], y=data["macd_hist"],
                marker_color=np.where(data["macd_hist"] >= 0, COLORS["green"], COLORS["red"]),
                name="MACD Hist"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data["time"], y=data["macd"],
                line=dict(color=COLORS["blue"], width=1.5), name="MACD"), row=2, col=1)
            fig.add_trace(go.Scatter(x=data["time"], y=data["macd_signal"],
                line=dict(color=COLORS["yellow"], width=1.5), name="Signal"), row=2, col=1)

        # RSI
        if "rsi" in data.columns:
            fig.add_trace(go.Scatter(x=data["time"], y=data["rsi"],
                line=dict(color=COLORS["blue"], width=1.5), name="RSI"), row=3, col=1)
            fig.add_hline(y=70, line_dash="dash", line_color=COLORS["red"], row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color=COLORS["green"], row=3, col=1)

        fig.update_layout(height=650, template="plotly_dark", hovermode="x unified",
                          margin=dict(l=0, r=0, t=30, b=0),
                          legend=dict(orientation="h", y=1.02, x=0))
        fig.update_xaxes(rangeslider_visible=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        perf = calculate_performance_xau(data) if is_gold else (
            __import__('strategies.momentum', fromlist=['calculate_performance']).calculate_performance(data))

        st.markdown("### 📊 Performance")
        metrics = [
            ("Return", f"{perf.get('total_return_pct', 0):+.2f}%", "positive" if perf.get('total_return_pct', 0) > 0 else "negative"),
            ("Buy & Hold", f"{perf.get('buy_hold_return_pct', 0):+.2f}%", "positive" if perf.get('buy_hold_return_pct', 0) > 0 else "negative"),
            ("Sharpe", f"{perf.get('sharpe_ratio', 'N/A')}", "positive" if isinstance(perf.get('sharpe_ratio'), (int,float)) and perf['sharpe_ratio'] > 1 else "negative"),
            ("Max DD", f"{perf.get('max_drawdown_pct', 0):.2f}%", "negative"),
            ("Win Rate", f"{perf.get('win_rate_pct', 0):.1f}%", "positive" if perf.get('win_rate_pct', 50) > 50 else "negative"),
        ]
        if is_gold:
            metrics += [
                ("Trades", f"{perf.get('num_trades', 0)}", "neutral"),
                ("Avg Hold", f"{perf.get('avg_hold_bars', 0)} bars", "neutral"),
                ("Avg Trade", f"{perf.get('avg_trade_pct', 0):+.3f}%", "positive" if perf.get('avg_trade_pct', 0) > 0 else "negative"),
                ("Exposure", f"{perf.get('exposure_pct', 0):.1f}%", "neutral"),
            ]
        else:
            metrics += [("Trades", f"{perf.get('num_trades', 0)}", "neutral"),
                        ("Exposure", f"{perf.get('exposure_pct', 0):.1f}%", "neutral")]

        for label, value, cls in metrics:
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; padding:4px 0; border-bottom:1px solid #2D3039;">
                <span style="color:#9E9E9E;">{label}</span>
                <span class="{cls}" style="font-weight:600;">{value}</span>
            </div>
            """, unsafe_allow_html=True)

        if is_gold and "exit_reasons" in perf and perf["exit_reasons"]:
            st.markdown("---")
            st.markdown("### 🚪 Exit Reasons")
            total_exits = sum(perf["exit_reasons"].values())
            for reason, count in sorted(perf["exit_reasons"].items(), key=lambda x: -x[1]):
                pct = count / total_exits * 100 if total_exits > 0 else 0
                emoji = {"stop_loss": "🔴", "take_profit": "🟢", "timeout": "⏰", "reversal": "🔄"}.get(reason, "⚪")
                st.markdown(f"{emoji} **{reason}**: {count} ({pct:.0f}%)")

        st.markdown("---")
        latest = data.iloc[-1]
        pos = latest.get("position", 0)
        signal_icon = "🟢" if pos == 1 else "🔴" if pos == -1 else "⚪"
        signal_text = "LONG" if pos == 1 else "SHORT" if pos == -1 else "FLAT"
        rsi_val = latest.get("rsi", 50)
        atr_val = latest.get("atr_pct", 0)

        st.markdown("### 🔔 Current Status")
        st.markdown(f"""
        <div class="metric-card" style="text-align:center;">
            <div style="font-size:2rem;">{signal_icon}</div>
            <div style="font-size:1.5rem; font-weight:700;">{signal_text}</div>
            <div style="color:#9E9E9E;">Price: {latest['close']:.2f} | RSI: {rsi_val:.1f} | ATR%: {atr_val:.4f}%</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.error(f"Could not load data for {pair}.")

st.markdown("---")

# ─── Equity Curve ───
st.subheader("💰 Equity Curve")

if data is not None:
    df = data.copy()
    df["returns"] = df["close"].pct_change()
    df["strategy_returns"] = df["position"].shift(1) * df["returns"]
    df["equity"] = 10000 * (1 + df["strategy_returns"]).cumprod()
    df["buy_hold"] = 10000 * (1 + df["returns"]).cumprod()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    fig.add_trace(go.Scatter(x=df["time"], y=df["equity"], line=dict(color=COLORS["green"], width=2), name="Strategy"), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["time"], y=df["buy_hold"], line=dict(color="#9E9E9E", width=1, dash="dash"), name="Buy & Hold"), row=1, col=1)

    peak = df["equity"].expanding().max()
    dd = (df["equity"] - peak) / peak * 100
    fig.add_trace(go.Scatter(x=df["time"], y=dd, fill="tozeroy", line=dict(color=COLORS["red"], width=1), name="Drawdown"), row=2, col=1)

    fig.update_layout(height=350, template="plotly_dark", hovermode="x unified",
                      margin=dict(l=0, r=0, t=10, b=0), legend=dict(orientation="h", y=1.02, x=0))
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ─── Recent Signals ───
st.subheader("📋 Recent Activity")

if data is not None:
    col1, col2 = st.columns(2)

    with col1:
        sig_cols = ["time", "close", "rsi", "atr_pct", "signal"]
        if is_gold:
            sig_cols += ["sl_price", "tp_price", "exit_reason"]
        else:
            sig_cols += ["position"]

        recent = data[sig_cols].tail(30).copy()
        recent["signal"] = recent["signal"].map({1: "🟢 BUY", -1: "🔴 SELL", 0: "⚪"})

        if is_gold and "exit_reason" in recent.columns:
            recent["exit_reason"] = recent["exit_reason"].replace("", "-")
            recent = recent.rename(columns={"time": "Time", "close": "Price", "rsi": "RSI",
                                            "atr_pct": "ATR%", "signal": "Signal",
                                            "sl_price": "SL", "tp_price": "TP", "exit_reason": "Exit"})
            recent["Time"] = recent["Time"].dt.strftime("%H:%M")
            recent["Price"] = recent["Price"].round(2)
            recent["SL"] = recent["SL"].round(2)
            recent["TP"] = recent["TP"].round(2)
            display_cols = ["Time", "Price", "RSI", "Signal", "SL", "TP", "Exit"]
        else:
            recent = recent.rename(columns={"time": "Time", "close": "Price", "rsi": "RSI",
                                            "atr_pct": "ATR%", "signal": "Signal"})
            recent["Time"] = recent["Time"].dt.strftime("%H:%M" if tf in ("1m","5m","15m","30m") else "%m/%d %H:%M")
            recent["Price"] = recent["Price"].round(5) if not is_gold else recent["Price"]
            display_cols = ["Time", "Price", "RSI", "ATR%", "Signal"]

        st.markdown("**Recent candles & signals**")
        st.dataframe(recent[display_cols], use_container_width=True, hide_index=True)

    with col2:
        if is_gold and not data[data["signal"] != 0].empty:
            signals = data[data["signal"] != 0].tail(20).copy()
            st.markdown("**Trade exits breakdown**")
            exit_data = signals[signals["exit_reason"] != ""].copy()
            if not exit_data.empty:
                exit_data["hold_bars"] = 0
                for i in range(len(exit_data)):
                    idx = exit_data.index[i]
                    prev_sig = signals[signals.index < idx]
                    if not prev_sig.empty:
                        entry_idx = prev_sig.index[-1]
                        exit_data.loc[idx, "hold_bars"] = signals.index.get_loc(idx) - signals.index.get_loc(entry_idx)

                exit_data["entry_time"] = ""
                for i in range(len(exit_data)):
                    idx = exit_data.index[i]
                    prev = signals[signals.index < idx]
                    if not prev.empty:
                        exit_data.loc[idx, "entry_time"] = prev.iloc[-1]["time"]

                exit_display = exit_data[["time", "close", "exit_reason"]].tail(10).copy()
                exit_display["time"] = exit_display["time"].dt.strftime("%H:%M")
                exit_display = exit_display.rename(columns={"time": "Time", "close": "Price", "exit_reason": "Exit"})
                st.dataframe(exit_display, use_container_width=True, hide_index=True)
            else:
                st.info("No exits yet in recent data.")
        else:
            st.markdown("**Strategy metrics**")
            if perf:
                cols_left, cols_right = st.columns(2)
                perf_items = [(k, v) for k, v in perf.items() if not isinstance(v, dict)]
                mid = len(perf_items) // 2
                with cols_left:
                    for k, v in perf_items[:mid]:
                        st.metric(k.replace("_", " ").title(), v)
                with cols_right:
                    for k, v in perf_items[mid:]:
                        st.metric(k.replace("_", " ").title(), v)

# Footer
st.markdown("---")
st.caption("""
**XAU Scalp Monitor** — Data: Yahoo Finance | Strategy: Gold Scalping (5-15 min holds)
Deployed on Streamlit Community Cloud · Fully automated · Free forever
""")
