"""
XAU/USD Gold Signals — Clean Monitor
Shows: current signal, entry, SL, TP. Nothing else.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

st.set_page_config(page_title="XAU Signals", page_icon="🥇", layout="centered")

# Hide streamlit branding
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp { background: #0D1117; }
    .block-container { padding: 1rem 1rem !important; max-width: 500px; }
    div[data-testid="stSidebar"] {display: none;}
</style>
""", unsafe_allow_html=True)

# ─── Import strategy ───
from strategies.xau_scalp import add_indicators_xau, generate_signals_xau, calculate_performance_xau

# ─── Fetch XAU/USD data directly ───
def fetch_gold(tf="5m", days=3):
    raw = yf.download("GC=F", period=f"{max(1,days)}d", interval=tf, progress=False)
    if raw is None or raw.empty:
        return None
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    df = raw.reset_index()
    df.columns = [c.lower().strip() for c in df.columns]
    m = {"datetime":"time","date":"time","open":"open","high":"high","low":"low","close":"close","volume":"volume"}
    df = df.rename(columns={k:v for k,v in m.items() if k in df.columns})
    df["time"] = pd.to_datetime(df["time"])
    return df.sort_values("time").reset_index(drop=True)

def format_price(v):
    return f"${v:,.2f}" if v == v else "—"

# ─── Load & compute ───
df = fetch_gold("5m", 3)
if df is None or len(df) < 60:
    st.error("Failed to load XAU/USD data. Try again in a minute.")
    st.stop()

df = add_indicators_xau(df)
df = generate_signals_xau(df, mom_threshold=0.55, atr_sl_mult=1.2, atr_tp_mult=2.0, max_hold_bars=4)

latest = df.iloc[-1]
pos = latest.get("position", 0)
price = latest["close"]
rsi_val = latest.get("rsi", 50)
atr_val = latest.get("atr_pct", 0)

# ─── Find latest signal ───
signals = df[df["signal"] != 0]
last_signal = signals.iloc[-1] if not signals.empty else None

# Also look at last 5 for history
recent_signals = signals.tail(10) if not signals.empty else pd.DataFrame()

# ─── UI ───
cols = st.columns([1, 2, 1])
with cols[1]:
    st.markdown(f"<h2 style='text-align:center;color:#E0E0E0;margin:0;'>🥇 XAU/USD</h2>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align:center;color:#9E9E9E;font-size:0.8rem;margin-top:-5px;'>{datetime.now(timezone.utc).strftime('%H:%M UTC')}</p>", unsafe_allow_html=True)

# ─── Current Price ───
st.markdown(f"<h1 style='text-align:center;font-size:3.5rem;font-weight:800;color:#E0E0E0;margin:0;'>{format_price(price)}</h1>", unsafe_allow_html=True)

# ─── Signal Badge ───
if pos == 1:
    signal_color = "#00C853"
    signal_text = "🟢 BUY"
    signal_bg = "#003D1A"
elif pos == -1:
    signal_color = "#FF1744"
    signal_text = "🔴 SHORT"
    signal_bg = "#3D0010"
else:
    signal_color = "#757575"
    signal_text = "⚪ WAIT"
    signal_bg = "#1A1A1A"

st.markdown(f"""
<div style="text-align:center;padding:0.3rem;border-radius:8px;background:{signal_bg};margin:0.5rem 0;">
    <span style="font-size:2rem;font-weight:700;color:{signal_color};">{signal_text}</span>
</div>
""", unsafe_allow_html=True)

# ─── SL / TP ───
if last_signal is not None:
    entry_px = last_signal["close"]
    sl_px = last_signal.get("sl_price", np.nan)
    tp_px = last_signal.get("tp_price", np.nan)

    pos2 = last_signal.get("position", last_signal.get("signal", 0))
    if pos2 == 0:
        pos2 = last_signal["signal"]

    if pos2 == 1:
        sl_label = "🛑 Stop Loss"
        tp_label = "🎯 Take Profit"
        sl_color = "#FF5252"
        tp_color = "#69F0AE"
    elif pos2 == -1:
        sl_label = "🛑 Stop Loss"
        tp_label = "🎯 Take Profit"
        sl_color = "#FF5252"
        tp_color = "#69F0AE"
    else:
        sl_label = "SL"
        tp_label = "TP"
        sl_color = "#757575"
        tp_color = "#757575"

    col_sl, col_tp = st.columns(2)
    with col_sl:
        sl_val = format_price(sl_px) if sl_px == sl_px else "—"
        st.markdown(f"""
        <div style="text-align:center;padding:0.8rem;border-radius:8px;background:#1A1D23;border:1px solid #2D3039;">
            <div style="color:#9E9E9E;font-size:0.8rem;">{sl_label}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{sl_color};">{sl_val}</div>
        </div>
        """, unsafe_allow_html=True)
    with col_tp:
        tp_val = format_price(tp_px) if tp_px == tp_px else "—"
        if pos2 == 1:
            diff_px = tp_px - entry_px
            diff_pct = diff_px / entry_px * 100
            tp_detail = f"+${diff_px:.2f} (+{diff_pct:.2f}%)" if diff_px == diff_px else ""
        elif pos2 == -1:
            diff_px = entry_px - tp_px
            diff_pct = diff_px / entry_px * 100
            tp_detail = f"+${diff_px:.2f} (+{diff_pct:.2f}%)" if diff_px == diff_px else ""
        else:
            tp_detail = ""
        st.markdown(f"""
        <div style="text-align:center;padding:0.8rem;border-radius:8px;background:#1A1D23;border:1px solid #2D3039;">
            <div style="color:#9E9E9E;font-size:0.8rem;">{tp_label}</div>
            <div style="font-size:1.5rem;font-weight:700;color:{tp_color};">{tp_val}</div>
            <div style="font-size:0.7rem;color:#69F0AE;">{tp_detail}</div>
        </div>
        """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div style="text-align:center;padding:1rem;color:#9E9E9E;">
        No active signal. Waiting for setup conditions...
    </div>
    """, unsafe_allow_html=True)

# ─── Context row ───
st.markdown("---")
col_rsi, col_atr, col_vol = st.columns(3)
with col_rsi:
    rsi_c = "#00C853" if 40 <= rsi_val <= 60 else "#FF5252"
    st.markdown(f"<div style='text-align:center;'><span style='color:#9E9E9E;font-size:0.7rem;'>RSI</span><br><span style='color:{rsi_c};font-size:1.2rem;font-weight:600;'>{rsi_val:.1f}</span></div>", unsafe_allow_html=True)
with col_atr:
    st.markdown(f"<div style='text-align:center;'><span style='color:#9E9E9E;font-size:0.7rem;'>ATR%</span><br><span style='color:#E0E0E0;font-size:1.2rem;font-weight:600;'>{atr_val:.3f}%</span></div>", unsafe_allow_html=True)
with col_vol:
    vol = latest.get("volume", 0)
    st.markdown(f"<div style='text-align:center;'><span style='color:#9E9E9E;font-size:0.7rem;'>Volume</span><br><span style='color:#E0E0E0;font-size:1.2rem;font-weight:600;'>{int(vol):,}</span></div>", unsafe_allow_html=True)

# ─── Mini Price Line ───
st.markdown("---")
st.markdown("<p style='color:#9E9E9E;font-size:0.7rem;margin:0;'>Price (last 50 candles)</p>", unsafe_allow_html=True)

chart_data = df[["time", "close"]].tail(50).copy()
chart_data.columns = ["t", "price"]
st.line_chart(chart_data.set_index("t"), height=150, color="#FFD600")

# ─── Recent Signal History ───
if not recent_signals.empty:
    st.markdown("---")
    st.markdown("<p style='color:#9E9E9E;font-size:0.7rem;margin:0;'>Recent Signals</p>", unsafe_allow_html=True)

    hist = recent_signals[["time", "close", "signal", "exit_reason"]].copy()
    hist["time"] = hist["time"].dt.strftime("%H:%M")
    hist["signal"] = hist["signal"].map({1: "🟢 BUY", -1: "🔴 SELL"})
    hist = hist.rename(columns={"time": "T", "close": "Price", "signal": "Sig", "exit_reason": "Exit"})
    hist["Exit"] = hist["Exit"].replace("", "—")
    st.dataframe(hist, width="stretch", hide_index=True, height=200)
else:
    st.markdown("---")
    st.markdown("<p style='color:#9E9E9E;font-size:0.7rem;text-align:center;'>No signals generated in recent data.</p>", unsafe_allow_html=True)

# ─── Footer ───
st.markdown("---")
st.markdown("<p style='text-align:center;color:#3D4048;font-size:0.6rem;'>Data: Yahoo Finance GC=F · Refresh page to update</p>", unsafe_allow_html=True)
