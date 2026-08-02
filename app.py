import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.graph_objects as go
import time
import random
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Quant Risk Engine",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

    :root {
        --bg:        #08090d;
        --surface:   #0f1117;
        --surface2:  #161820;
        --border:    #1e2030;
        --border2:   #2a2d42;
        --accent:    #4fffb0;
        --accent2:   #00c9ff;
        --warn:      #ff6b6b;
        --muted:     #4a4f6a;
        --text:      #e8eaf0;
        --text2:     #8b90ab;
        --font-head: 'Space Grotesk', 'Trebuchet MS', sans-serif;
        --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background: var(--bg) !important; }

    * { font-family: var(--font-mono) !important; }

    [data-testid="stHeader"],
    header[data-testid="stHeader"],
    #stDecoration,
    [data-testid="stToolbar"],
    [data-testid="stStatusWidget"],
    [data-testid="stMainMenuPopover"] { display: none !important; visibility: hidden !important; }

    .block-container {
        padding-top: 1.8rem !important;
        padding-left: 2.5rem !important;
        padding-right: 2.5rem !important;
        padding-bottom: 4rem !important;
        max-width: 1600px !important;
    }

    [data-testid="stSidebar"] {
        min-width: 280px !important;
        width: 280px !important;
        transform: none !important;
        visibility: visible !important;
        display: block !important;
        position: relative !important;
    }
    [data-testid="stSidebarCollapseButton"],
    [data-testid="collapsedControl"] {
        display: none !important;
    }

    .disclaimer-dropdown {
        margin-top: 2rem;
        font-size: 0.7rem;
    }
    .disclaimer-dropdown details {
        background: transparent !important;
        border: none !important;
    }
    .disclaimer-dropdown summary {
        cursor: pointer;
        list-style: none;
        display: inline-block;
        color: var(--muted);
        letter-spacing: 0.08em;
        text-transform: uppercase;
        font-size: 0.65rem;
    }
    .disclaimer-dropdown summary::-webkit-details-marker,
    .disclaimer-dropdown summary::marker { display: none; }
    .disclaimer-dropdown summary::after {
        content: " ▼";
        font-size: 10px;
        color: var(--accent);
    }
    .disclaimer-dropdown details[open] summary::after {
        content: " ▲";
    }
    .disclaimer-dropdown ul {
        margin-top: 0.5rem;
        padding-left: 1.2rem;
        color: var(--text2);
    }
    .disclaimer-dropdown li {
        margin: 0.2rem 0;
        line-height: 1.4;
    }

    [data-testid="stSidebar"] {
        background: var(--surface) !important;
        border-right: 1px solid var(--border) !important;
    }
    [data-testid="stSidebar"] * { color: var(--text) !important; }
    [data-testid="stSidebar"] .stTextInput input,
    [data-testid="stSidebar"] .stSelectbox select,
    [data-testid="stSidebar"] .stDateInput input {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text) !important;
        border-radius: 4px !important;
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text2) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }
    [data-testid="stSidebar"] h3 {
        font-family: var(--font-head) !important;
        color: var(--accent) !important;
        font-size: 0.7rem !important;
        letter-spacing: 0.2em !important;
        text-transform: uppercase !important;
        margin-top: 1.4rem !important;
    }
    [data-testid="stSidebar"] hr {
        border-color: var(--border) !important;
        margin: 0.8rem 0 !important;
    }

    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="progressbar"] {
        background: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
        background: var(--accent) !important;
        box-shadow: 0 0 8px var(--accent) !important;
    }

    .qre-header {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid var(--border);
    }
    .qre-title-block { display: flex; flex-direction: column; gap: 0.3rem; }
    .qre-logo {
        font-family: 'Space Grotesk', 'Trebuchet MS', Arial, sans-serif !important;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.02em;
        line-height: 1;
        white-space: nowrap;
    }
    .qre-logo span { color: var(--accent); }
    .qre-subtitle {
        font-family: 'Space Grotesk', 'Trebuchet MS', Arial, sans-serif !important;
        font-size: 1rem;
        font-weight: 600;
        color: var(--text2);
        letter-spacing: 0.04em;
        white-space: nowrap;
    }
    .qre-tagline {
        font-size: 0.65rem;
        color: var(--muted);
        letter-spacing: 0.16em;
        text-transform: uppercase;
        align-self: flex-end;
        margin-bottom: 0.1rem;
    }
    .qre-badge {
        margin-left: auto;
        background: rgba(79,255,176,0.07);
        border: 1px solid rgba(79,255,176,0.25);
        color: var(--accent);
        font-size: 0.65rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        padding: 0.3rem 0.7rem;
        border-radius: 3px;
        white-space: nowrap;
    }

    .sec-label {
        font-family: var(--font-head);
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.75rem;
        margin-top: 2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .sec-label::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }
    .sec-label .dot {
        width: 5px; height: 5px;
        border-radius: 50%;
        background: var(--accent);
        display: inline-block;
    }

    .kpi-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 1px;
        background: var(--border);
        border: 1px solid var(--border);
        border-radius: 6px;
        overflow: hidden;
        margin-bottom: 0.5rem;
    }
    .kpi-cell {
        background: var(--surface);
        padding: 1.1rem 1.3rem;
        position: relative;
    }
    .kpi-cell:hover { background: var(--surface2); }
    .kpi-cell::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, var(--accent), var(--accent2));
        opacity: 0;
        transition: opacity 0.2s;
    }
    .kpi-cell:hover::before { opacity: 1; }
    .kpi-label {
        font-size: 0.62rem;
        letter-spacing: 0.15em;
        text-transform: uppercase;
        color: var(--text2);
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        font-family: var(--font-head);
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--text);
        line-height: 1;
    }
    .kpi-value.pos { color: var(--accent); }
    .kpi-value.neg { color: var(--warn); }
    .kpi-sub {
        font-size: 0.6rem;
        color: var(--muted);
        margin-top: 0.3rem;
    }

    .panel {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 6px;
        padding: 1.2rem 1.4rem;
    }
    .panel-title {
        font-size: 0.62rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: var(--text2);
        margin-bottom: 1rem;
        padding-bottom: 0.6rem;
        border-bottom: 1px solid var(--border);
    }

    .wt-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.45rem 0;
        border-bottom: 1px solid var(--border);
        gap: 0.5rem;
    }
    .wt-row:last-child { border-bottom: none; }
    .wt-ticker {
        font-size: 0.78rem;
        font-weight: 500;
        color: var(--text);
        min-width: 68px;
    }
    .wt-bar-wrap {
        flex: 1;
        height: 4px;
        background: var(--border2);
        border-radius: 2px;
        overflow: hidden;
    }
    .wt-bar { height: 100%; border-radius: 2px;
              background: linear-gradient(90deg, var(--accent), var(--accent2)); }
    .wt-pct {
        font-size: 0.72rem;
        color: var(--accent);
        min-width: 44px;
        text-align: right;
    }

    .geo-badge {
        display: inline-block;
        background: rgba(255,107,107,0.1);
        border: 1px solid rgba(255,107,107,0.3);
        color: #ff6b6b;
        font-size: 0.6rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        padding: 0.2rem 0.5rem;
        border-radius: 3px;
        margin-top: 0.5rem;
    }

    .resolve-note {
        font-size: 0.62rem;
        color: var(--muted);
        margin-top: 0.4rem;
        letter-spacing: 0.03em;
    }
    .resolve-note b { color: var(--accent); }

    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="qre-header">
  <div class="qre-title-block">
    <div class="qre-logo" style="font-family:'Space Grotesk','Trebuchet MS',Arial,sans-serif !important;">
      QUANT <span>RISK</span> ENGINE
    </div>
    <div class="qre-subtitle" style="font-family:'Space Grotesk','Trebuchet MS',Arial,sans-serif !important;">
      Portfolio Optimiser
    </div>
  </div>
  <div class="qre-tagline">
    Black-Litterman &nbsp;&mdash;&nbsp; Ledoit-Wolf &nbsp;&mdash;&nbsp; Geopolitical Overlay
  </div>
  <div class="qre-badge">&#9679;&nbsp; Live Data</div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════
# GLOBAL EXCHANGE SUFFIX MAP
# Used to auto-resolve a bare ticker (or a wrong suffix) to the correct
# Yahoo Finance listing across major world markets.
# ═══════════════════════════════════════════════════════════════════════
EXCHANGE_SUFFIXES = [
    "",       # US markets (NYSE / NASDAQ) — no suffix
    ".NS",    # India — NSE
    ".BO",    # India — BSE
    ".T",     # Japan — Tokyo
    ".KS",    # South Korea — KOSPI
    ".KQ",    # South Korea — KOSDAQ
    ".HK",    # Hong Kong
    ".SS",    # China — Shanghai
    ".SZ",    # China — Shenzhen
    ".TW",    # Taiwan
    ".SI",    # Singapore
    ".AX",    # Australia
    ".NZ",    # New Zealand
    ".L",     # UK — London
    ".DE",    # Germany
    ".PA",    # France
    ".AS",    # Netherlands
    ".SW",    # Switzerland
    ".MI",    # Italy
    ".MC",    # Spain
    ".LS",    # Portugal
    ".ST",    # Sweden
    ".OL",    # Norway
    ".CO",    # Denmark
    ".HE",    # Finland
    ".VI",    # Austria
    ".BR",    # Belgium
    ".IR",    # Ireland
    ".IS",    # Turkey
    ".TA",    # Israel
    ".SR",    # Saudi Arabia — Tadawul
    ".QA",    # Qatar
    ".AE",    # UAE
    ".KW",    # Kuwait
    ".BH",    # Bahrain
    ".ZA",    # South Africa (some feeds)
    ".JO",    # South Africa — Johannesburg
    ".TO",    # Canada — Toronto
    ".V",     # Canada — TSX Venture
    ".SA",    # Brazil
    ".MX",    # Mexico
    ".BA",    # Argentina
    ".SN",    # Chile
]

# Region label shown to the user when a suffix is auto-resolved
SUFFIX_REGION = {
    "": "US", ".NS": "India (NSE)", ".BO": "India (BSE)", ".T": "Japan",
    ".KS": "South Korea (KOSPI)", ".KQ": "South Korea (KOSDAQ)", ".HK": "Hong Kong",
    ".SS": "China (Shanghai)", ".SZ": "China (Shenzhen)", ".TW": "Taiwan",
    ".SI": "Singapore", ".AX": "Australia", ".NZ": "New Zealand", ".L": "UK",
    ".DE": "Germany", ".PA": "France", ".AS": "Netherlands", ".SW": "Switzerland",
    ".MI": "Italy", ".MC": "Spain", ".LS": "Portugal", ".ST": "Sweden",
    ".OL": "Norway", ".CO": "Denmark", ".HE": "Finland", ".VI": "Austria",
    ".BR": "Belgium", ".IR": "Ireland", ".IS": "Turkey", ".TA": "Israel",
    ".SR": "Saudi Arabia", ".QA": "Qatar", ".AE": "UAE", ".KW": "Kuwait",
    ".BH": "Bahrain", ".ZA": "South Africa", ".JO": "South Africa",
    ".TO": "Canada", ".V": "Canada (TSXV)", ".SA": "Brazil", ".MX": "Mexico",
    ".BA": "Argentina", ".SN": "Chile",
}

# --- SIDEBAR (permanently visible) ---
with st.sidebar:
    st.markdown("### Tickers")
    default_tickers = "AAPL, MSFT, JPM, TCS.BO, INFY.NS, ASML"
    assets = st.text_input("Comma-separated", default_tickers, label_visibility="collapsed")
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]
    st.markdown(
        '<div class="resolve-note">Type a bare symbol (e.g. <b>7203</b>, <b>005930</b>, <b>2222</b>) '
        'or add a suffix yourself. Unresolved symbols are auto-matched across '
        'US, India, Japan, South Korea, China, Gulf, Europe & more.</div>',
        unsafe_allow_html=True
    )

    st.markdown("### Date Range")
    default_start = datetime(2020, 1, 2)
    start_date = st.date_input("Start", value=default_start, label_visibility="collapsed")
    default_end = datetime.now()
    end_date   = st.date_input("End", value=default_end, max_value=default_end, label_visibility="collapsed")

    st.divider()
    st.markdown("### Geopolitical Overlay")
    geo_events = st.multiselect(
        "Active events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability",
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes", "India Policy"],
        default=["US-China Tech Tensions"],
        label_visibility="collapsed"
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### Black-Litterman View")
    view_ticker = st.selectbox("Asset", ticker_list if ticker_list else ["AAPL"], label_visibility="collapsed")
    view_return = st.slider("Expected Return (%)", -20, 40, 10) / 100
    view_conf   = st.slider("Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Risk Controls")
    max_cap     = st.slider("Max Weight per Asset (%)", 10, 100, 35) / 100
    div_penalty = st.slider("L2 Diversification Penalty", 0.0, 2.0, 0.5)

    st.divider()

# ═══════════════════════════════════════════════════════════════════════
# DATA FETCHING — per-ticker, retrying, with global exchange auto-resolve
# ═══════════════════════════════════════════════════════════════════════

def _fetch_one(symbol, start_str, end_str, tries=3):
    """Try to pull a single ticker's close price series with retries and
    a couple of fallback methods, since yfinance intermittently drops or
    rate-limits individual symbols inside batch calls."""
    last_err = None
    for attempt in range(tries):
        try:
            tk = yf.Ticker(symbol)
            hist = tk.history(start=start_str, end=end_str, auto_adjust=True)
            if hist is not None and not hist.empty and "Close" in hist.columns:
                series = pd.to_numeric(hist["Close"], errors="coerce").dropna()
                if len(series) > 5:
                    return series
            # fallback: batch-style download for this one symbol
            raw = yf.download(symbol, start=start_str, end=end_str,
                               progress=False, auto_adjust=True, threads=False)
            if raw is not None and not raw.empty:
                col = "Close" if "Close" in raw.columns else (
                      "Adj Close" if "Adj Close" in raw.columns else None)
                if col:
                    series = pd.to_numeric(raw[col].squeeze(), errors="coerce").dropna()
                    if len(series) > 5:
                        return series
        except Exception as e:
            last_err = e
        time.sleep(0.4 + attempt * 0.6 + random.uniform(0, 0.3))
    return None


def _resolve_ticker(raw_symbol, start_str, end_str):
    """Try the symbol as typed first. If that fails, strip any suffix and
    sweep major global exchange suffixes until one returns real data.
    Returns (series, resolved_symbol) or (None, None)."""
    raw_symbol = raw_symbol.strip().upper()

    # 1) try exactly as given
    series = _fetch_one(raw_symbol, start_str, end_str)
    if series is not None:
        return series, raw_symbol

    # 2) strip an existing suffix (if any) to get the bare root symbol
    if "." in raw_symbol:
        root = raw_symbol.split(".")[0]
    else:
        root = raw_symbol

    # 3) sweep candidate suffixes (skip the one already tried)
    already_tried = raw_symbol
    for suf in EXCHANGE_SUFFIXES:
        candidate = f"{root}{suf}"
        if candidate == already_tried:
            continue
        series = _fetch_one(candidate, start_str, end_str, tries=2)
        if series is not None:
            return series, candidate

    return None, None


@st.cache_data(ttl=3600, show_spinner=False)
def get_clean_data(tickers, start, end):
    today_str = datetime.now().strftime('%Y-%m-%d')
    start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
    end_str   = end.strftime('%Y-%m-%d')   if hasattr(end,   'strftime') else str(end)

    if start_str > today_str: start_str = '2020-01-06'
    if end_str   > today_str: end_str   = today_str
    if start_str >= end_str:
        start_str = '2020-01-06'
        end_str   = today_str

    all_tickers = list(dict.fromkeys(tickers + ["^GSPC"]))
    close_prices = {}
    resolved_map = {}   # original -> actually-used symbol (only when different)
    failed = []

    for t in all_tickers:
        series, resolved = _resolve_ticker(t, start_str, end_str)
        if series is not None:
            close_prices[resolved] = series
            if resolved != t:
                resolved_map[t] = resolved
        else:
            failed.append(t)

    if not close_prices:
        return pd.DataFrame(), pd.Series(), {}, resolved_map, failed

    df = pd.DataFrame(close_prices)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill().dropna(how='all')

    benchmark  = df.pop("^GSPC") if "^GSPC" in df.columns else pd.Series()
    assets_df  = df.dropna(axis=1, how='all')

    if assets_df.empty or len(assets_df) < 10:
        return pd.DataFrame(), pd.Series(), {}, resolved_map, failed

    # Approximate market caps (USD, billions) for the Black-Litterman prior.
    # Falls back to 100 for anything not listed here — still lets the model run.
    fixed_caps = {
        # US Tech
        'AAPL':3000,'MSFT':2800,'GOOGL':1800,'AMZN':1600,'TSLA':600,'NVDA':2200,
        # US Financials & Other
        'JPM':500,'V':500,'MA':400,'JNJ':380,'XOM':400,'WMT':450,'PG':350,
        # Europe
        'MC.PA':400,'ASML':350,'NESN.SW':300,
        # Japan
        '7203.T':280,'6758.T':130,'9984.T':90,'8306.T':110,
        # South Korea
        '005930.KS':380,'000660.KS':110,'035420.KS':30,
        # China / HK
        '0700.HK':400,'9988.HK':220,'0941.HK':230,
        # Gulf
        '2222.SR':1900,'1120.SR':50,
        # India (NSE/BSE)
        'TCS.BO':200,'TCS.NS':200,'INFY.BO':180,'INFY.NS':180,'RELIANCE.BO':250,
        'RELIANCE.NS':250,'HDFCBANK.BO':140,'HDFCBANK.NS':140,'ITC.BO':80,'ITC.NS':80,
        'SBIN.BO':110,'SBIN.NS':110,'BAJAJFINSV.BO':100,'MARUTI.BO':90,'MARUTI.NS':90,
        'WIPRO.BO':70,'WIPRO.NS':70,'AXISBANK.BO':95,'LT.BO':85,'BHARTIARTL.BO':75,
        'BHARTIARTL.NS':75,'SUNPHARMA.BO':65,'ICICIBANK.BO':120,'ICICIBANK.NS':120,
    }
    mcaps = {t: fixed_caps.get(t, 100) * 1e9 for t in assets_df.columns}
    return assets_df, benchmark, mcaps, resolved_map, failed


def apply_geopolitical_overlay(weights, events, intensity):
    if not events or intensity <= 0.5:
        return weights

    sector_risk = {
        'Technology':    {'US-China Tech Tensions':0.8,'Supply Chain Disruption':0.7,'Trade Policy Changes':0.6,'India Policy':0.2},
        'Financials':    {'Currency Volatility':0.6,'Middle East Instability':0.3,'Trade Policy Changes':0.4,'India Policy':0.3},
        'Semiconductors':{'US-China Tech Tensions':0.9,'Supply Chain Disruption':0.8,'Trade Policy Changes':0.7,'India Policy':0.2},
        'Healthcare':    {'EU Regulation Shift':0.5,'Trade Policy Changes':0.3,'India Policy':0.2},
        'Automotive':    {'Supply Chain Disruption':0.9,'Trade Policy Changes':0.7,'India Policy':0.3},
        'Consumer':      {'Supply Chain Disruption':0.5,'Currency Volatility':0.3,'India Policy':0.4},
        'Energy':        {'Middle East Instability':0.8,'Trade Policy Changes':0.6,'India Policy':0.2},
        'IT Services':   {'US-China Tech Tensions':0.4,'India Policy':0.5,'Trade Policy Changes':0.5},
        'Banking':       {'Currency Volatility':0.7,'India Policy':0.4,'Trade Policy Changes':0.3},
        'Pharma':        {'EU Regulation Shift':0.6,'India Policy':0.3,'Trade Policy Changes':0.4},
    }

    ticker_sectors = {
        'AAPL':'Technology','MSFT':'Technology','JPM':'Financials','MC.PA':'Consumer',
        'ASML':'Semiconductors','NESN.SW':'Healthcare','GOOGL':'Technology','AMZN':'Technology',
        'TSLA':'Automotive','NVDA':'Semiconductors','V':'Financials','JNJ':'Healthcare',
        'XOM':'Energy','WMT':'Consumer','PG':'Consumer','MA':'Financials',
        'TCS.BO':'IT Services','TCS.NS':'IT Services','INFY.BO':'IT Services','INFY.NS':'IT Services',
        'RELIANCE.BO':'Energy','RELIANCE.NS':'Energy','HDFCBANK.BO':'Banking','HDFCBANK.NS':'Banking',
        'ICICIBANK.BO':'Banking','ICICIBANK.NS':'Banking','ITC.BO':'Consumer','ITC.NS':'Consumer',
        'SBIN.BO':'Banking','SBIN.NS':'Banking','BAJAJFINSV.BO':'Financials','MARUTI.BO':'Automotive',
        'MARUTI.NS':'Automotive','WIPRO.BO':'IT Services','WIPRO.NS':'IT Services',
        'AXISBANK.BO':'Banking','LT.BO':'Automotive','BHARTIARTL.BO':'Consumer',
        'BHARTIARTL.NS':'Consumer','SUNPHARMA.BO':'Pharma',
        '7203.T':'Automotive','6758.T':'Technology','9984.T':'Technology','8306.T':'Banking',
        '005930.KS':'Semiconductors','000660.KS':'Semiconductors','035420.KS':'Technology',
        '0700.HK':'Technology','9988.HK':'Technology','0941.HK':'Technology',
        '2222.SR':'Energy','1120.SR':'Banking',
    }

    adj = {}
    for ticker, w in weights.items():
        if w == 0:
            adj[ticker] = 0
            continue
        sector = ticker_sectors.get(ticker, 'Technology')
        risk_score = sum(sector_risk.get(sector, {}).get(e, 0.1) for e in events)
        adj[ticker] = max(0.01, w * (1 - risk_score * intensity * 0.15))

    total = sum(adj.values())
    return {k: v / total for k, v in adj.items()} if total > 0 else weights

PLOTLY_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#8b90ab", size=11),
    xaxis=dict(gridcolor="#1e2030", linecolor="#1e2030", zerolinecolor="#1e2030"),
    yaxis=dict(gridcolor="#1e2030", linecolor="#1e2030", zerolinecolor="#1e2030"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1e2030",
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    margin=dict(l=12, r=12, t=28, b=12),
)

def plot_efficient_frontier(mu, S, rf=0.02):
    try:
        ef_mv = EfficientFrontier(mu, S)
        ef_mv.min_volatility()
        min_vol = ef_mv.portfolio_performance(risk_free_rate=rf)

        ef_ms = EfficientFrontier(mu, S)
        ef_ms.max_sharpe()
        max_sharpe = ef_ms.portfolio_performance(risk_free_rate=rf)

        target_returns = np.linspace(min_vol[0], float(mu.max()), 22)
        vols = []
        for tr in target_returns:
            try:
                ef2 = EfficientFrontier(mu, S)
                ef2.efficient_return(target_return=tr)
                _, v, _ = ef2.portfolio_performance(risk_free_rate=rf)
                vols.append(v)
            except:
                vols.append(np.nan)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=vols, y=target_returns, mode='lines', name='Frontier',
            line=dict(color='#4fffb0', width=2.5)
        ))
        asset_vols = np.sqrt(np.diag(S.values) * 252)
        fig.add_trace(go.Scatter(
            x=asset_vols, y=mu, mode='markers+text', name='Assets',
            marker=dict(size=9, color='#00c9ff', symbol='circle',
                        line=dict(color='#0f1117', width=1.5)),
            text=list(S.columns), textposition="top center",
            textfont=dict(size=10, color='#8b90ab')
        ))
        fig.add_trace(go.Scatter(
            x=[min_vol[1]], y=[min_vol[0]], mode='markers', name='Min Vol',
            marker=dict(size=13, color='#4fffb0', symbol='diamond',
                        line=dict(color='#0f1117', width=2))
        ))
        fig.add_trace(go.Scatter(
            x=[max_sharpe[1]], y=[max_sharpe[0]], mode='markers', name='Max Sharpe',
            marker=dict(size=15, color='#ffd166', symbol='star',
                        line=dict(color='#0f1117', width=1.5))
        ))
        fig.update_layout(
            height=420,
            xaxis_title="Annual Volatility",
            yaxis_title="Expected Return",
            xaxis=dict(tickformat=".0%", **PLOTLY_THEME['xaxis']),
            yaxis=dict(tickformat=".0%", **PLOTLY_THEME['yaxis']),
            **{k: v for k, v in PLOTLY_THEME.items() if k not in ('xaxis','yaxis')}
        )
        return fig
    except Exception:
        return None

def plot_performance(p_cum, p_rets, bench_prices):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=p_cum.index, y=p_cum, name="Portfolio",
        line=dict(color='#4fffb0', width=2.5),
        fill='tozeroy', fillcolor='rgba(79,255,176,0.04)'
    ))
    if not bench_prices.empty:
        b_ret    = bench_prices.pct_change().dropna()
        common   = p_cum.index.intersection(b_ret.index)
        if len(common) > 0:
            b_cum = (1 + b_ret.loc[common]).cumprod()
            fig.add_trace(go.Scatter(
                x=b_cum.index, y=b_cum, name="S&P 500",
                line=dict(color='#2a2d42', width=1.8, dash='dot')
            ))
    fig.update_layout(
        height=340,
        xaxis_title=None,
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        xaxis=dict(gridcolor="#1e2030", linecolor="#1e2030"),
        yaxis=dict(gridcolor="#1e2030", linecolor="#1e2030", tickformat=".2f"),
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ('xaxis','yaxis')}
    )
    return fig

def plot_drawdown(p_cum):
    rolling_max = p_cum.expanding().max()
    drawdown    = (p_cum - rolling_max) / rolling_max
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=drawdown.index, y=drawdown, name="Drawdown",
        line=dict(color='#ff6b6b', width=1.5),
        fill='tozeroy', fillcolor='rgba(255,107,107,0.07)'
    ))
    fig.update_layout(
        height=160,
        xaxis_title=None,
        yaxis_title="Drawdown",
        xaxis=dict(gridcolor="#1e2030", linecolor="#1e2030"),
        yaxis=dict(gridcolor="#1e2030", linecolor="#1e2030", tickformat=".0%"),
        margin=dict(l=12, r=12, t=10, b=12),
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ('xaxis','yaxis','margin')}
    )
    return fig

def plot_allocation_donut(final_weights):
    w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
    w_df = w_df[w_df['Weight'] > 0.001].sort_values('Weight', ascending=False)
    colors = ['#4fffb0','#00c9ff','#ffd166','#ef476f','#a29bfe',
              '#74b9ff','#55efc4','#fd79a8','#fdcb6e','#6c5ce7']
    fig = go.Figure(go.Pie(
        labels=w_df.index, values=w_df['Weight'],
        hole=0.62,
        marker=dict(colors=colors[:len(w_df)],
                    line=dict(color='#08090d', width=3)),
        textinfo='label+percent',
        textfont=dict(size=10, family='DM Mono, monospace', color='#8b90ab'),
        hovertemplate='<b>%{label}</b><br>%{percent}<extra></extra>'
    ))
    fig.add_annotation(
        text=f"{len(w_df)}<br><span style='font-size:10px'>assets</span>",
        x=0.5, y=0.5, showarrow=False,
        font=dict(size=22, color='#e8eaf0', family='Syne, sans-serif')
    )
    fig.update_layout(
        height=340,
        showlegend=False,
        **{k: v for k, v in PLOTLY_THEME.items() if k not in ('xaxis','yaxis')}
    )
    return fig

def weight_table_html(final_weights):
    sorted_w = sorted(final_weights.items(), key=lambda x: -x[1])
    rows = ""
    for ticker, w in sorted_w:
        if w < 0.001: continue
        bar_pct = w * 100
        rows += f"""
        <div class="wt-row">
          <div class="wt-ticker">{ticker}</div>
          <div class="wt-bar-wrap"><div class="wt-bar" style="width:{bar_pct:.1f}%"></div></div>
          <div class="wt-pct">{w:.1%}</div>
        </div>"""
    return f'<div class="panel"><div class="panel-title">// Weight Distribution</div>{rows}</div>'

# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════
try:
    if not ticker_list:
        st.info("Enter tickers in the sidebar to begin. Bare symbols are auto-matched to the right global exchange.")
        st.stop()

    with st.spinner("Fetching market data across global exchanges…"):
        prices, bench_prices, market_caps, resolved_map, failed_tickers = get_clean_data(
            ticker_list, start_date, end_date)

    if prices.empty:
        st.error("No data returned for any ticker. Check the symbols and date range, or try again — "
                  "Yahoo Finance occasionally rate-limits requests.")
        st.stop()

    # Tickers whose resolved (post-auto-match) symbol is actually in the data
    resolved_symbols = set(prices.columns)
    available = []
    for t in ticker_list:
        candidate = resolved_map.get(t, t)
        if candidate in resolved_symbols:
            available.append(candidate)

    still_missing = [t for t in ticker_list if t in failed_tickers]
    if resolved_map:
        notes = " · ".join(f"{orig} → {new}" for orig, new in resolved_map.items())
        st.markdown(
            f'<div class="resolve-note">Auto-resolved: <b>{notes}</b></div>',
            unsafe_allow_html=True
        )
    if still_missing:
        st.warning(f"Could not find data for: {', '.join(sorted(still_missing))}. "
                    f"Try the exact Yahoo Finance symbol (e.g. .T for Japan, .KS for South Korea, "
                    f".SR for Saudi Arabia, .HK for Hong Kong, .TA for Israel).")
    if not available:
        st.error("None of the entered tickers returned valid data.")
        st.stop()

    ticker_list = available
    prices      = prices[ticker_list]
    market_caps = {t: market_caps[t] for t in ticker_list if t in market_caps}

    if view_ticker not in ticker_list:
        view_ticker = resolved_map.get(view_ticker, view_ticker)
        if view_ticker not in ticker_list:
            view_ticker = ticker_list[0]

    # Covariance
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)
    tickers_final = list(S.columns)

    # Black-Litterman
    try:
        mcap_series = pd.Series(
            {t: market_caps.get(t, 1e11) for t in tickers_final}, index=tickers_final)
        prior_rets  = black_litterman.market_implied_prior_returns(mcap_series, 2.5, S)
        bl = black_litterman.BlackLittermanModel(
            S, pi=prior_rets,
            absolute_views={view_ticker: view_return},
            omega="idzorek",
            view_confidences=[min(view_conf, 0.99)]
        )
        bl_mu = bl.bl_returns().reindex(tickers_final)
    except Exception as e:
        st.warning(f"Black-Litterman failed ({e}). Using historical returns.")
        ret_tmp = prices.pct_change().dropna()
        bl_mu   = (ret_tmp.mean() * 252).reindex(tickers_final).fillna(0.10)

    # Optimisation
    try:
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        ef.max_sharpe()
        optimized_weights = ef.clean_weights()
        total = sum(optimized_weights.values())
        if abs(total - 1.0) > 0.01:
            optimized_weights = {k: v / total for k, v in optimized_weights.items()}
    except Exception as e:
        st.error(f"Optimisation failed: {e}")
        optimized_weights = {t: 1 / len(tickers_final) for t in tickers_final}

    final_weights = (apply_geopolitical_overlay(optimized_weights, geo_events, geo_intensity)
                     if geo_events and geo_intensity > 0.5 else optimized_weights)

    # Returns & metrics
    weights_arr = np.array([final_weights.get(t, 0) for t in tickers_final])
    returns     = prices.pct_change().dropna().astype(float)
    p_rets      = (returns * weights_arr).sum(axis=1)
    p_cum       = (1 + p_rets).cumprod()

    ann_ret  = p_rets.mean() * 252
    ann_vol  = p_rets.std()  * np.sqrt(252)
    sharpe   = ann_ret / ann_vol if ann_vol > 0 else 0
    rolling_max = p_cum.expanding().max()
    max_dd   = ((p_cum - rolling_max) / rolling_max).min()
    down_vol = p_rets[p_rets < 0].std() * np.sqrt(252)
    sortino  = ann_ret / down_vol if down_vol > 0 else 0

    # KPI strip
    sharpe_cls  = "pos" if sharpe  >= 1    else ("neg" if sharpe  < 0    else "")
    ret_cls     = "pos" if ann_ret >= 0    else "neg"
    dd_cls      = "neg" if max_dd  < -0.15 else ""

    st.markdown(f"""
    <div class="kpi-grid">
      <div class="kpi-cell">
        <div class="kpi-label">Sharpe Ratio</div>
        <div class="kpi-value {sharpe_cls}">{sharpe:.2f}</div>
        <div class="kpi-sub">risk-adjusted return</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Annual Return</div>
        <div class="kpi-value {ret_cls}">{ann_ret:.1%}</div>
        <div class="kpi-sub">252-day annualised</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Annual Volatility</div>
        <div class="kpi-value">{ann_vol:.1%}</div>
        <div class="kpi-sub">1σ annualised</div>
      </div>
      <div class="kpi-cell">
        <div class="kpi-label">Max Drawdown</div>
        <div class="kpi-value {dd_cls}">{max_dd:.1%}</div>
        <div class="kpi-sub">peak-to-trough</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Performance & Sortino
    st.markdown('<div class="sec-label"><span class="dot"></span> Performance Comparison</div>',
                unsafe_allow_html=True)

    perf_col, sortino_col = st.columns([3, 1])
    with perf_col:
        st.plotly_chart(plot_performance(p_cum, p_rets, bench_prices),
                        use_container_width=True, config=dict(displayModeBar=False))
        st.plotly_chart(plot_drawdown(p_cum),
                        use_container_width=True, config=dict(displayModeBar=False))

    with sortino_col:
        sortino_cls = "pos" if sortino >= 1 else ("neg" if sortino < 0 else "")
        active_tickers_html = "".join(
            f'<div class="wt-row"><div class="wt-ticker">{t}</div>'
            f'<div class="wt-pct" style="color:#8b90ab">{bl_mu.get(t, 0):.1%}</div></div>'
            for t in tickers_final
        )
        st.markdown(f"""
        <div class="panel" style="margin-bottom:1rem">
          <div class="panel-title">// Sortino Ratio</div>
          <div class="kpi-value {sortino_cls}" style="font-size:2.4rem">{sortino:.2f}</div>
          <div class="kpi-sub" style="margin-top:0.4rem">downside-risk adjusted</div>
        </div>
        <div class="panel">
          <div class="panel-title">// BL Expected Returns</div>
          {active_tickers_html}
        </div>
        """, unsafe_allow_html=True)

    # Allocation
    st.markdown('<div class="sec-label"><span class="dot"></span> Portfolio Allocation</div>',
                unsafe_allow_html=True)

    alloc_col, wt_col = st.columns([3, 2])
    with alloc_col:
        st.plotly_chart(plot_allocation_donut(final_weights),
                        use_container_width=True, config=dict(displayModeBar=False))
        if geo_events and geo_intensity > 0.5:
            events_str = " · ".join(geo_events)
            st.markdown(
                f'<div class="geo-badge">GEO OVERLAY ACTIVE: {events_str} @ {geo_intensity:.1f}x</div>',
                unsafe_allow_html=True)

    with wt_col:
        st.markdown(weight_table_html(final_weights), unsafe_allow_html=True)

    # Efficient Frontier
    st.markdown('<div class="sec-label"><span class="dot"></span> Efficient Frontier</div>',
                unsafe_allow_html=True)

    fig_ef = plot_efficient_frontier(bl_mu, S)
    if fig_ef:
        st.plotly_chart(fig_ef, use_container_width=True, config=dict(displayModeBar=False))

    # Export
    st.markdown('<div class="sec-label"><span class="dot"></span> Export</div>',
                unsafe_allow_html=True)

    e1, e2, e3 = st.columns(3)
    with e1:
        csv_w = pd.DataFrame.from_dict(final_weights, orient='index',
                                        columns=['Weight']).to_csv().encode()
        st.download_button("↓ Portfolio Weights", csv_w,
                           "portfolio_weights.csv", "text/csv")
    with e2:
        csv_p = pd.DataFrame({'Date': p_cum.index, 'Return': p_rets.values,
                               'Cumulative': p_cum.values}).to_csv(index=False).encode()
        st.download_button("↓ Performance Data", csv_p,
                           "portfolio_performance.csv", "text/csv")
    with e3:
        params_df = pd.DataFrame([{
            'Analysis_Date': datetime.now().strftime('%Y-%m-%d'),
            'Tickers': ', '.join(tickers_final),
            'View_Asset': view_ticker, 'View_Return': f"{view_return:.2%}",
            'View_Confidence': f"{view_conf:.0%}", 'Sharpe': f"{sharpe:.2f}",
            'Annual_Return': f"{ann_ret:.2%}", 'Annual_Vol': f"{ann_vol:.2%}",
            'Max_DD': f"{max_dd:.2%}",
        }])
        st.download_button("↓ Strategy Parameters",
                           params_df.to_csv(index=False).encode(),
                           "strategy_parameters.csv", "text/csv")

    # Disclaimer dropdown
    st.markdown("""
    <div class="disclaimer-dropdown">
      <details>
        <summary>Risk Disclaimer</summary>
        <ul>
          <li>Educational and research purposes only.</li>
          <li>Past performance is not indicative of future results.</li>
          <li>Consult a qualified financial adviser before making investment decisions.</li>
          <li>The Black-Litterman views, geopolitical overlay, and optimisation parameters are hypothetical.</li>
        </ul>
      </details>
    </div>
    """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Engine Error: {e}")
    with st.expander("Traceback"):
        import traceback
        st.code(traceback.format_exc())
