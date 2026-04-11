import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Quant Risk Engine",
    layout="wide",
    page_icon="◈",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

    /* ── Root & global reset ─────────────────────────────── */
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
        --font-head: 'Syne', sans-serif;
        --font-mono: 'DM Mono', monospace;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background: var(--bg) !important; }

    * { font-family: var(--font-mono) !important; }

    /* ── Sidebar ─────────────────────────────────────────── */
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

    /* Slider track */
    [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="progressbar"] {
        background: var(--accent) !important;
    }
    [data-testid="stSidebar"] .stSlider [data-baseweb="thumb"] {
        background: var(--accent) !important;
        box-shadow: 0 0 8px var(--accent) !important;
    }

    /* ── Main content ────────────────────────────────────── */
    .block-container {
        padding: 2rem 2.5rem 4rem !important;
        max-width: 1600px !important;
    }

    /* ── Header ──────────────────────────────────────────── */
    .qre-header {
        display: flex;
        align-items: flex-end;
        gap: 1.2rem;
        margin-bottom: 2rem;
        padding-bottom: 1.2rem;
        border-bottom: 1px solid var(--border);
    }
    .qre-logo {
        font-family: var(--font-head);
        font-size: 2.6rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.03em;
        line-height: 1;
    }
    .qre-logo span { color: var(--accent); }
    .qre-tagline {
        font-size: 0.68rem;
        color: var(--muted);
        letter-spacing: 0.18em;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
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
    }

    /* ── Section labels ──────────────────────────────────── */
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

    /* ── KPI strip ───────────────────────────────────────── */
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

    /* ── Panel cards ─────────────────────────────────────── */
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

    /* ── Weight table ────────────────────────────────────── */
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

    /* ── Alerts / info ───────────────────────────────────── */
    [data-testid="stAlert"] {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text2) !important;
        border-radius: 4px !important;
    }

    /* ── Plotly chart containers ─────────────────────────── */
    .js-plotly-plot .plotly { background: transparent !important; }

    /* ── Download buttons ────────────────────────────────── */
    .stDownloadButton button {
        background: var(--surface2) !important;
        border: 1px solid var(--border2) !important;
        color: var(--text2) !important;
        font-size: 0.68rem !important;
        letter-spacing: 0.1em !important;
        text-transform: uppercase !important;
        border-radius: 3px !important;
        padding: 0.4rem 0.9rem !important;
        width: 100% !important;
        transition: border-color 0.15s !important;
    }
    .stDownloadButton button:hover {
        border-color: var(--accent) !important;
        color: var(--accent) !important;
    }

    /* ── Expander ────────────────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 4px !important;
    }
    [data-testid="stExpander"] summary { color: var(--text2) !important; }

    /* ── Spinner ─────────────────────────────────────────── */
    [data-testid="stSpinner"] { color: var(--accent) !important; }

    /* ── General text ────────────────────────────────────── */
    p, li, span, div { color: var(--text2); }
    h1, h2, h3, h4 { color: var(--text) !important; }
    .stMarkdown a { color: var(--accent2) !important; }

    /* ── Geo overlay badge ───────────────────────────────── */
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

    /* Scrollbar */
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="qre-header">
  <div>
    <div class="qre-logo">◈ QUANT <span>RISK</span> ENGINE</div>
  </div>
  <div style="margin-bottom:0.2rem">
    <div class="qre-tagline">Black-Litterman · Ledoit-Wolf · Geopolitical Overlay</div>
  </div>
  <div class="qre-badge">◉ Live Data</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Tickers")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Comma-separated", default_tickers, label_visibility="collapsed")
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]

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
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
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
    debug_mode = st.checkbox("Debug mode", value=False)


# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start, end, debug=False):
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
    for t in all_tickers:
        for attempt in range(3):
            try:
                raw = yf.download(t, start=start_str, end=end_str,
                                  progress=False, auto_adjust=True)
                time.sleep(0.3)
                if raw.empty: continue
                col = 'Close' if 'Close' in raw.columns else (
                      'Adj Close' if 'Adj Close' in raw.columns else None)
                if col:
                    series = pd.to_numeric(raw[col].squeeze(), errors='coerce').dropna()
                    if len(series) > 5:
                        close_prices[t] = series
                        break
            except Exception as e:
                if debug: st.warning(f"Attempt {attempt+1} for {t}: {e}")
                time.sleep(0.5)

    if not close_prices:
        return pd.DataFrame(), pd.Series(), {}

    df = pd.DataFrame(close_prices)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.ffill().bfill().dropna(how='all')

    benchmark  = df.pop("^GSPC") if "^GSPC" in df.columns else pd.Series()
    assets_df  = df.dropna(axis=1, how='all')

    if assets_df.empty or len(assets_df) < 10:
        return pd.DataFrame(), pd.Series(), {}

    fixed_caps = {
        'AAPL':3000,'MSFT':2800,'JPM':500,'MC.PA':400,'ASML':350,'NESN.SW':300,
        'GOOGL':1800,'AMZN':1600,'TSLA':600,'NVDA':2200,'V':500,'JNJ':380,
        'XOM':400,'WMT':450,'PG':350,'MA':400
    }
    mcaps = {t: fixed_caps.get(t, 100) * 1e9 for t in assets_df.columns}
    return assets_df, benchmark, mcaps


# --- GEOPOLITICAL OVERLAY ---
def apply_geopolitical_overlay(weights, events, intensity):
    if not events or intensity <= 0.5:
        return weights
    sector_risk = {
        'Technology':    {'US-China Tech Tensions':0.8,'Supply Chain Disruption':0.7,'Trade Policy Changes':0.6},
        'Financials':    {'Currency Volatility':0.6,'Middle East Instability':0.3,'Trade Policy Changes':0.4},
        'Semiconductors':{'US-China Tech Tensions':0.9,'Supply Chain Disruption':0.8,'Trade Policy Changes':0.7},
        'Healthcare':    {'EU Regulation Shift':0.5,'Trade Policy Changes':0.3},
        'Automotive':    {'Supply Chain Disruption':0.9,'Trade Policy Changes':0.7},
        'Consumer':      {'Supply Chain Disruption':0.5,'Currency Volatility':0.3},
        'Energy':        {'Middle East Instability':0.8,'Trade Policy Changes':0.6},
    }
    ticker_sectors = {
        'AAPL':'Technology','MSFT':'Technology','JPM':'Financials','MC.PA':'Consumer',
        'ASML':'Semiconductors','NESN.SW':'Healthcare','GOOGL':'Technology','AMZN':'Technology',
        'TSLA':'Automotive','NVDA':'Semiconductors','V':'Financials','JNJ':'Healthcare',
        'XOM':'Energy','WMT':'Consumer','PG':'Consumer','MA':'Financials',
    }
    adj = {}
    for ticker, w in weights.items():
        if w == 0: adj[ticker] = 0; continue
        sector = ticker_sectors.get(ticker, 'Technology')
        risk_score = sum(sector_risk.get(sector, {}).get(e, 0.1) for e in events)
        adj[ticker] = max(0.01, w * (1 - risk_score * intensity * 0.15))
    total = sum(adj.values())
    return {k: v / total for k, v in adj.items()} if total > 0 else weights


# --- CHART HELPERS ---
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
    except Exception as e:
        if debug_mode: st.warning(f"Efficient frontier error: {e}")
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
    return f'<div class="panel"><div class="panel-title">◈ Weight Distribution</div>{rows}</div>'


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
try:
    if not ticker_list:
        st.info("Enter tickers in the sidebar to begin.")
        st.stop()

    with st.spinner("Fetching market data…"):
        prices, bench_prices, market_caps = get_clean_data(
            ticker_list, start_date, end_date, debug=debug_mode)

    if prices.empty:
        st.error("No data returned. Check ticker symbols and date range.")
        st.stop()

    available = [t for t in ticker_list if t in prices.columns]
    missing   = set(ticker_list) - set(available)
    if missing:
        st.warning(f"Tickers not found: {', '.join(sorted(missing))}")
    if not available:
        st.error("None of the entered tickers returned valid data.")
        st.stop()

    ticker_list = available
    prices      = prices[ticker_list]
    market_caps = {t: market_caps[t] for t in ticker_list if t in market_caps}

    if view_ticker not in ticker_list:
        view_ticker = ticker_list[0]

    # ── Covariance ────────────────────────────────────────────────────────
    try:
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    except Exception:
        S = risk_models.sample_cov(prices)
    tickers_final = list(S.columns)

    # ── Black-Litterman ───────────────────────────────────────────────────
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

    # ── Optimisation ──────────────────────────────────────────────────────
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

    # ── Returns & metrics ─────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # LAYOUT  ①  KPI strip
    # ──────────────────────────────────────────────────────────────────────
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

    # ──────────────────────────────────────────────────────────────────────
    # LAYOUT  ②  Performance (left) + Drawdown (stacked) | Sortino card (right)
    # ──────────────────────────────────────────────────────────────────────
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
          <div class="panel-title">◈ Sortino Ratio</div>
          <div class="kpi-value {sortino_cls}" style="font-size:2.4rem">{sortino:.2f}</div>
          <div class="kpi-sub" style="margin-top:0.4rem">downside-risk adjusted</div>
        </div>
        <div class="panel">
          <div class="panel-title">◈ BL Expected Returns</div>
          {active_tickers_html}
        </div>
        """, unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # LAYOUT  ③  Portfolio Allocation (donut) | Weight table
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="dot"></span> Portfolio Allocation</div>',
                unsafe_allow_html=True)

    alloc_col, wt_col = st.columns([3, 2])
    with alloc_col:
        st.plotly_chart(plot_allocation_donut(final_weights),
                        use_container_width=True, config=dict(displayModeBar=False))
        if geo_events and geo_intensity > 0.5:
            events_str = " · ".join(geo_events)
            st.markdown(
                f'<div class="geo-badge">▲ Geo overlay active: {events_str} @ {geo_intensity:.1f}×</div>',
                unsafe_allow_html=True)

    with wt_col:
        st.markdown(weight_table_html(final_weights), unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────
    # LAYOUT  ④  Efficient Frontier (full-width)
    # ──────────────────────────────────────────────────────────────────────
    st.markdown('<div class="sec-label"><span class="dot"></span> Efficient Frontier</div>',
                unsafe_allow_html=True)

    fig_ef = plot_efficient_frontier(bl_mu, S)
    if fig_ef:
        st.plotly_chart(fig_ef, use_container_width=True, config=dict(displayModeBar=False))

    # ──────────────────────────────────────────────────────────────────────
    # LAYOUT  ⑤  Exports
    # ──────────────────────────────────────────────────────────────────────
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

    # ── Disclaimer ────────────────────────────────────────────────────────
    with st.expander("⚠  Risk Disclaimer"):
        st.markdown("""
        Educational and research purposes only.
        Past performance is not indicative of future results.
        Consult a qualified financial adviser before making investment decisions.
        """)

    # ── Debug ─────────────────────────────────────────────────────────────
    if debug_mode:
        with st.expander("Debug"):
            st.write(f"tickers_final: {tickers_final}")
            st.write(f"S.columns: {list(S.columns)}")
            st.write(f"bl_mu.index: {list(bl_mu.index)}")
            st.write(f"prices shape: {prices.shape}")
            st.dataframe(prices.tail(3))

except Exception as e:
    st.error(f"Engine Error: {e}")
    with st.expander("Traceback", expanded=debug_mode):
        import traceback
        st.code(traceback.format_exc())
