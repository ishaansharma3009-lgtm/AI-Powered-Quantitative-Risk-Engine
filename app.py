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
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700;800&family=IBM+Plex+Mono:wght@300;400;500&display=swap');

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
        --font-head: 'Space Grotesk', 'Trebuchet MS', sans-serif;
        --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
    }

    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stMain"], .main { background: var(--bg) !important; }

    * { font-family: var(--font-mono) !important; }

    /* ── Hide Streamlit's default top toolbar/header bar ── */
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

    /* ── SIDEBAR TOGGLE: MINIMAL ARROW ONLY (NO TEXT) ── */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background: rgba(79,255,176,0.07) !important;
        border: 1px solid rgba(79,255,176,0.3) !important;
        border-radius: 4px !important;
        width: 32px !important;
        height: 32px !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        overflow: hidden !important;
    }

    /* Remove the default SVG and any potential text labels */
    [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="collapsedControl"] button * {
        display: none !important;
        visibility: hidden !important;
    }

    /* CSS chevron pointing LEFT (when sidebar is open) */
    [data-testid="stSidebarCollapseButton"] button::before {
        content: '' !important;
        display: block !important;
        visibility: visible !important;
        width: 8px !important;
        height: 8px !important;
        border-left: 2.5px solid var(--accent) !important;
        border-bottom: 2.5px solid var(--accent) !important;
        transform: rotate(45deg) !important;
        margin-left: 3px !important;
    }

    /* CSS chevron pointing RIGHT (when sidebar is collapsed) */
    [data-testid="collapsedControl"] button::before {
        content: '' !important;
        display: block !important;
        visibility: visible !important;
        width: 8px !important;
        height: 8px !important;
        border-left: 2.5px solid var(--accent) !important;
        border-bottom: 2.5px solid var(--accent) !important;
        transform: rotate(225deg) !important;
        margin-left: -3px !important;
    }

    /* ── DISCLAIMER EXPANDER: ARROW ONLY (NO TEXT) ── */
    [data-testid="stExpander"] details summary {
        justify-content: center !important;
        padding: 0.6rem !important;
    }

    /* Hide the default Expander text "Disclaimer" and default SVG */
    [data-testid="stExpander"] details summary p,
    [data-testid="stExpander"] details summary svg {
        display: none !important;
        visibility: hidden !important;
    }

    /* Custom Downward Arrow */
    [data-testid="stExpander"] details summary::after {
        content: '' !important;
        display: inline-block !important;
        width: 9px !important;
        height: 9px !important;
        border-right: 2px solid var(--muted) !important;
        border-bottom: 2px solid var(--muted) !important;
        transform: rotate(45deg) !important;
        transition: transform 0.3s ease, border-color 0.3s !important;
    }

    /* Rotate Arrow Up when open */
    [data-testid="stExpander"] details[open] summary::after {
        transform: rotate(-135deg) !important;
        border-color: var(--accent) !important;
        margin-top: 5px !important;
    }

    /* ── Sidebar ── */
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
    }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stMarkdown p {
        color: var(--text2) !important;
        font-size: 0.72rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── Header ── */
    .qre-header {
        display: flex;
        align-items: center;
        gap: 2rem;
        margin-bottom: 2rem;
        padding-bottom: 1.4rem;
        border-bottom: 1px solid var(--border);
    }
    .qre-logo {
        font-family: 'Space Grotesk' !important;
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text);
        letter-spacing: -0.02em;
    }
    .qre-logo span { color: var(--accent); }

    /* ── KPI Grid ── */
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
    .kpi-cell { background: var(--surface); padding: 1.1rem 1.3rem; }
    .kpi-label { font-size: 0.62rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text2); }
    .kpi-value { font-family: 'Space Grotesk' !important; font-size: 1.9rem; font-weight: 700; color: var(--text); }
    .kpi-value.pos { color: var(--accent); }
    .kpi-value.neg { color: var(--warn); }

    /* ── Section labels ── */
    .sec-label {
        font-family: 'Space Grotesk' !important;
        font-size: 0.65rem;
        letter-spacing: 0.22em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 0.75rem;
        margin-top: 2rem;
        display: flex;
        align-items: center; gap: 0.5rem;
    }
    .sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }

    /* ── Weight table ── */
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1.2rem; }
    .panel-title { font-size: 0.62rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text2); border-bottom: 1px solid var(--border); margin-bottom: 1rem; padding-bottom: 0.6rem; }
    .wt-row { display: flex; align-items: center; justify-content: space-between; padding: 0.45rem 0; border-bottom: 1px solid var(--border); gap: 0.5rem; }
    .wt-ticker { font-size: 0.78rem; font-weight: 500; color: var(--text); min-width: 68px; }
    .wt-bar-wrap { flex: 1; height: 4px; background: var(--border2); border-radius: 2px; overflow: hidden; }
    .wt-bar { height: 100%; background: linear-gradient(90deg, var(--accent), var(--accent2)); }
    .wt-pct { font-size: 0.72rem; color: var(--accent); min-width: 44px; text-align: right; }
</style>
""", unsafe_allow_html=True)

# ── HEADER ──
st.markdown("""
<div class="qre-header">
  <div class="qre-title-block">
    <div class="qre-logo">QUANT <span>RISK</span> ENGINE</div>
    <div style="font-family:'Space Grotesk'; font-size:1rem; font-weight:600; color:var(--text2); letter-spacing:0.04em;">Portfolio Optimiser</div>
  </div>
  <div style="margin-left:auto; font-size:0.65rem; color:var(--muted); letter-spacing:0.16em; text-transform:uppercase;">
    Black-Litterman &nbsp;&mdash;&nbsp; Ledoit-Wolf &nbsp;&mdash;&nbsp; Geopolitical Overlay
  </div>
  <div style="background:rgba(79,255,176,0.07); border:1px solid rgba(79,255,176,0.25); color:var(--accent); font-size:0.65rem; padding:0.3rem 0.7rem; border-radius:3px;">&#9679;&nbsp; Live Data</div>
</div>
""", unsafe_allow_html=True)

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### Tickers")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Comma-separated", default_tickers, label_visibility="collapsed")
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]

    st.markdown("### Date Range")
    start_date = st.date_input("Start", value=datetime(2020, 1, 2), label_visibility="collapsed")
    end_date = st.date_input("End", value=datetime.now(), label_visibility="collapsed")

    st.divider()
    st.markdown("### Geopolitical Overlay")
    geo_events = st.multiselect("Active events", ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", "Supply Chain Disruption"], default=["US-China Tech Tensions"], label_visibility="collapsed")
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### Black-Litterman View")
    view_ticker = st.selectbox("Asset", ticker_list if ticker_list else ["AAPL"], label_visibility="collapsed")
    view_return = st.slider("Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("Confidence (%)", 10, 100, 50) / 100

    st.divider()
    max_cap = st.slider("Max Weight per Asset (%)", 10, 100, 35) / 100
    div_penalty = st.slider("L2 Diversification Penalty", 0.0, 2.0, 0.5)

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start, end):
    all_tickers = list(dict.fromkeys(tickers + ["^GSPC"]))
    try:
        df = yf.download(all_tickers, start=start, end=end, progress=False)['Close']
        df = df.ffill().bfill().dropna(axis=1, how='all')
        bench = df.pop("^GSPC") if "^GSPC" in df.columns else pd.Series()
        # Placeholder for market caps
        mcaps = {t: 500e9 for t in df.columns} 
        return df, bench, mcaps
    except:
        return pd.DataFrame(), pd.Series(), {}

# --- HELPER: GEOPOLITICAL ADJ ---
def apply_geopolitical_overlay(weights, events, intensity):
    if not events: return weights
    adj = {k: v * (1 - len(events) * intensity * 0.05) if any(x in k for x in ["AAPL", "ASML", "MSFT"]) else v for k, v in weights.items()}
    total = sum(adj.values())
    return {k: v/total for k,v in adj.items()}

# --- MAIN LOGIC ---
try:
    prices, bench, mcaps = get_clean_data(ticker_list, start_date, end_date)
    
    if not prices.empty:
        # Covariance & Returns
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        mcap_series = pd.Series(mcaps)
        pi = black_litterman.market_implied_prior_returns(mcap_series, 2.5, S)
        
        bl = black_litterman.BlackLittermanModel(S, pi=pi, absolute_views={view_ticker: view_return}, view_confidences=[view_conf])
        bl_mu = bl.bl_returns()
        
        # Optimization
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        ef.max_sharpe()
        weights = ef.clean_weights()
        final_weights = apply_geopolitical_overlay(weights, geo_events, geo_intensity)
        
        # Returns
        p_rets = prices.pct_change().dropna().dot(pd.Series(final_weights))
        p_cum = (1 + p_rets).cumprod()
        
        # Metrics
        tr, vol = p_cum.iloc[-1]-1, p_rets.std()*np.sqrt(252)
        
        # KPI ROW
        st.markdown(f"""
        <div class="kpi-grid">
          <div class="kpi-cell"><div class="kpi-label">Cumulative Return</div><div class="kpi-value {'pos' if tr>0 else 'neg'}">{tr:.1%}</div></div>
          <div class="kpi-cell"><div class="kpi-label">Ann. Volatility</div><div class="kpi-value">{vol:.1%}</div></div>
          <div class="kpi-cell"><div class="kpi-label">Sharpe Ratio</div><div class="kpi-value">{(tr/vol if vol!=0 else 0):.2f}</div></div>
          <div class="kpi-cell"><div class="kpi-label">Active Assets</div><div class="kpi-value">{len([v for v in final_weights.values() if v > 0.01])}</div></div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown('<div class="sec-label"><span class="dot"></span>Performance Analysis</div>', unsafe_allow_html=True)
            fig = px.line(p_cum, template="plotly_dark")
            fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", margin=dict(t=10), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            st.markdown('<div class="sec-label"><span class="dot"></span>Portfolio Weights</div>', unsafe_allow_html=True)
            wt_rows = "".join([f'<div class="wt-row"><div class="wt-ticker">{k}</div><div class="wt-bar-wrap"><div class="wt-bar" style="width:{v*100}%"></div></div><div class="wt-pct">{v:.1%}</div></div>' 
                               for k,v in sorted(final_weights.items(), key=lambda x:x[1], reverse=True) if v > 0.01])
            st.markdown(f'<div class="panel">{wt_rows}</div>', unsafe_allow_html=True)

    # ── FOOTER EXPANDER (ARROW ONLY) ──
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander(""):
        st.markdown("""
        <div style="font-size:0.65rem; color:var(--muted); text-align:center; font-family:var(--font-mono);">
        FOR INFORMATIONAL PURPOSES ONLY. NOT FINANCIAL ADVICE.<br>
        Models leverage Black-Litterman and Ledoit-Wolf Shrinkage. Past performance is not indicative of future results.
        </div>
        """, unsafe_allow_html=True)

except Exception as e:
    st.error(f"Execution Error: {e}")
