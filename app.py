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
    
    :root {
        --bg: #08090d;
        --surface: #0f1117;
        --surface2: #161820;
        --border: #1e2030;
        --border2: #2a2d42;
        --accent: #4fffb0;
        --accent2: #00c9ff;
        --warn: #ff6b6b;
        --muted: #4a4f6a;
        --text: #e8eaf0;
        --text2: #8b90ab;
    }

    html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] { 
        background: var(--bg) !important; 
    }
    * { font-family: 'IBM Plex Mono', monospace !important; }

    /* Hide default Streamlit header */
    [data-testid="stHeader"], header, #stDecoration, [data-testid="stToolbar"] {
        display: none !important;
    }

    .block-container {
        padding-top: 1rem !important;
        padding-left: 2rem !important;
        padding-right: 2rem !important;
        padding-bottom: 3rem !important;
        max-width: 1600px !important;
    }

    /* ==================== SIDEBAR COLLAPSE - ONLY ARROW ==================== */
    [data-testid="stSidebarCollapseButton"] button,
    [data-testid="collapsedControl"] button {
        background: rgba(79,255,176,0.08) !important;
        border: 1px solid rgba(79,255,176,0.4) !important;
        border-radius: 6px !important;
        width: 34px !important;
        height: 34px !important;
        padding: 0 !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    [data-testid="stSidebarCollapseButton"] button *,
    [data-testid="collapsedControl"] button * {
        display: none !important;
    }

    /* Arrow for expanded sidebar (points left) */
    [data-testid="stSidebarCollapseButton"] button::before {
        content: '→' !important;
        font-size: 18px !important;
        color: var(--accent) !important;
        font-weight: bold;
        line-height: 1;
    }

    /* Arrow for collapsed sidebar (points right) */
    [data-testid="collapsedControl"] button::before {
        content: '←' !important;
        font-size: 18px !important;
        color: var(--accent) !important;
        font-weight: bold;
    }

    /* Keyboard focus highlight */
    [data-testid="stSidebarCollapseButton"] button:focus,
    [data-testid="collapsedControl"] button:focus {
        outline: 2px solid var(--accent) !important;
        box-shadow: 0 0 0 4px rgba(79,255,176,0.2) !important;
    }

    /* ==================== EXPANDER - ONLY ARROW (Disclaimer) ==================== */
    [data-testid="stExpander"] details summary {
        list-style: none !important;
        padding: 0.8rem 1rem !important;
    }

    [data-testid="stExpander"] details summary::-webkit-details-marker,
    [data-testid="stExpander"] details summary::marker,
    [data-testid="stExpander"] details summary svg {
        display: none !important;
    }

    [data-testid="stExpander"] details summary > div {
        display: flex !important;
        align-items: center !important;
        gap: 12px !important;
    }

    /* Custom Arrow */
    [data-testid="stExpander"] details summary::after {
        content: '▼' !important;
        font-size: 14px !important;
        color: var(--text2) !important;
        transition: transform 0.3s ease !important;
        margin-left: auto !important;
    }

    [data-testid="stExpander"] details[open] summary::after {
        transform: rotate(180deg) !important;
    }

    [data-testid="stExpander"] details {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
    }

    /* Rest of your existing styles */
    .qre-header { display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 2rem; padding-bottom: 0.8rem; border-bottom: 1px solid var(--border); }
    .qre-logo { font-family: 'Space Grotesk', sans-serif; font-size: 1.8rem; font-weight: 800; color: var(--text); }
    .qre-logo span { color: var(--accent); }
    .kpi-grid, .panel, .sec-label, .wt-row { /* ... your existing styles ... */ }
    
    /* (Keeping all your original styling below - unchanged) */
    .sec-label { font-family: 'Space Grotesk'; font-size: 0.65rem; letter-spacing: 0.22em; text-transform: uppercase; color: var(--muted); margin: 2rem 0 0.75rem; display: flex; align-items: center; gap: 0.5rem; }
    .sec-label::after { content: ''; flex: 1; height: 1px; background: var(--border); }
    .sec-label .dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }
    
    .kpi-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px; background: var(--border); border-radius: 6px; overflow: hidden; }
    .kpi-cell { background: var(--surface); padding: 1rem 1.2rem; }
    .kpi-label { font-size: 0.6rem; letter-spacing: 0.15em; text-transform: uppercase; color: var(--text2); }
    .kpi-value { font-family: 'Space Grotesk'; font-size: 1.8rem; font-weight: 700; color: var(--text); }
    .kpi-value.pos { color: var(--accent); }
    .kpi-value.neg { color: var(--warn); }
    
    .panel { background: var(--surface); border: 1px solid var(--border); border-radius: 6px; padding: 1rem 1.2rem; }
    .panel-title { font-size: 0.6rem; letter-spacing: 0.18em; text-transform: uppercase; color: var(--text2); margin-bottom: 0.8rem; padding-bottom: 0.4rem; border-bottom: 1px solid var(--border); }
    
    ::-webkit-scrollbar { width: 4px; height: 4px; }
    ::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="qre-header">
  <div>
    <div class="qre-logo">QUANT <span>RISK</span> ENGINE</div>
    <div style="font-size:0.8rem; color:#8b90ab; letter-spacing:0.08em;">Portfolio Optimiser</div>
  </div>
  <div style="font-size:0.6rem; color:#4a4f6a; letter-spacing:0.16em; text-transform:uppercase; align-self:flex-end;">
    Black‑Litterman &nbsp;|&nbsp; Ledoit‑Wolf &nbsp;|&nbsp; Geopolitical Overlay
  </div>
  <div style="background:rgba(79,255,176,0.07); border:1px solid rgba(79,255,176,0.25); color:var(--accent); font-size:0.6rem; padding:0.2rem 0.7rem; border-radius:3px;">● Live Data</div>
</div>
""", unsafe_allow_html=True)

# Your existing sidebar and main code continues here...
# (I kept everything else exactly as you had it)
with st.sidebar:
    st.markdown("### Tickers")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Comma-separated", default_tickers, label_visibility="collapsed")
    # ... rest of your sidebar code (unchanged) ...

# [Rest of your original code remains exactly the same from here onwards]
# Just paste the rest of your script (data fetching, functions, main logic, etc.)

# At the very end, your disclaimer:
with st.expander("Risk Disclaimer"):
    st.markdown("""
    Educational and research purposes only.  
    Past performance is not indicative of future results.  
    Consult a qualified financial adviser before making investment decisions.
    """)
