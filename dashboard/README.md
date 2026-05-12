# Forex Quant Dashboard

Real-time monitoring dashboard for your forex quant trading strategy.
Free to run locally or deploy to Streamlit Community Cloud.

## Quick Start (Local)

```bash
cd ~/forex-quant
source .venv/bin/activate
streamlit run dashboard/app.py
```

Open http://localhost:8501 in your browser.

## Deploy for Free (Remote Access)

### Option 1: Streamlit Community Cloud ⭐ (recommended)

1. Push `dashboard/` to a GitHub repo
2. Go to https://streamlit.io/cloud
3. Sign in with GitHub
4. Click "New app" → select your repo
5. Main file path: `dashboard/app.py`
6. Deploy → your app is live at `https://your-app.streamlit.app`

**No credit card needed. Free forever.**
- 1 app included
- Unlimited team members
- Community support

### Option 2: Local + Cloudflare Tunnel

```bash
# Install cloudflared
brew install cloudflare/cloudflared/cloudflared

# Run dashboard
streamlit run dashboard/app.py --server.port 8501

# In another terminal, expose it
cloudflared tunnel --url http://localhost:8501
```

You get a `*.trycloudflare.com` URL — accessible from anywhere.

## Dashboard Features

- **Live prices** — all major forex pairs
- **Strategy signals** — MACD-confirmed momentum
- **Performance metrics** — Sharpe, drawdown, win rate
- **Equity curve** — strategy vs buy & hold
- **Multi-pair comparison** — find what's working
- **Adjustable parameters** — tweak in real-time

## Data Source

All data from Yahoo Finance — **free, no API key required**.
