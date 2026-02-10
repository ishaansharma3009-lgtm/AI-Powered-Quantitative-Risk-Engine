import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import warnings

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(page_title="Asset Management & Risk Engine", layout="wide", page_icon="📊")

# Custom CSS for that professional banking look
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #374151; margin-top: 1.5rem; border-bottom: 2px solid #E5E7EB; }
    .stMetric { background-color: #f8f9fa; padding: 10px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Asset Management & Quantitative Risk Engine</div>', unsafe_allow_html=True)

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.markdown("### 🛠️ Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]
    
    start_date = st.date_input("Analysis Start", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.markdown("### 🌍 Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### 🎯 Black-Litterman View")
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider(f"Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("Confidence (%)", 10, 100, 50) / 100
    
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100

# --- DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start):
    try:
        search_tickers = list(set(tickers + ["^GSPC"]))
        # Download only Close prices to avoid MultiIndex errors
        raw_data = yf.download(search_tickers, start=start, progress=False)["Close"]
        if isinstance(raw_data, pd.Series): raw_data = raw_data.to_frame()
        clean_df = raw_data.ffill().dropna()
        
        benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
        assets_data = clean_df.drop(columns=["^GSPC"]) if "^GSPC" in clean_df.columns else clean_df
        
        # Static Market Caps for BL Model
        fixed_caps = {'AAPL': 3e12, 'MSFT': 2.8e12, 'JPM': 5e11, 'MC.PA': 4e11, 'ASML': 3.5e11, 'NESN.SW': 3e11}
        mcaps = {t: fixed_caps.get(t, 1e11) for t in assets_data.columns}
        return assets_data, benchmark, mcaps
    except Exception:
        return pd.DataFrame(), pd.Series(), {}

# --- OPTIMIZATION LOGIC ---
def apply_geo_overlay(weights, events, intensity):
    if not events: return weights
    # High-level risk mapping
    adj_weights = {}
    for t, w in weights.items():
        # Higher penalty for Tech/Semis if China tensions or Supply Chain is selected
        risk_multiplier = 0.15 if any(x in ["US-China Tech Tensions", "Supply Chain Disruption"] for x in events) else 0.05
        impact = (len(events) * intensity * risk_multiplier)
        adj_weights[t] = max(0.01, w * (1 - impact))
    
    total = sum(adj_weights.values())
    return {k: v/total for k, v in adj_weights.items()}

# --- MAIN EXECUTION ---
prices, bench, mcaps = get_clean_data(ticker_list, start_date)

if not prices.empty:
    # 1. Math Models
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    pi = black_litterman.market_implied_prior_returns(mcaps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(S, pi=pi, absolute_views={view_ticker: view_return}, view_confidences=[view_conf])
    mu = bl.bl_returns()
    
    # 2. Optimization
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_cap))
    raw_weights = ef.max_sharpe()
    final_weights = apply_geo_overlay(raw_weights, geo_events, geo_intensity)
    
    # 3. Metrics
    w_arr = np.array([final_weights[t] for t in prices.columns])
    p_rets = (prices.pct_change().dropna() @ w_arr)
    p_cum = (1 + p_rets).cumprod()

    # --- UI DISPLAY ---
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Ann. Return", f"{p_rets.mean()*252:.1%}")
    col2.metric("Ann. Volatility", f"{p_rets.std()*np.sqrt(252):.1%}")
    col3.metric("Sharpe Ratio", f"{(p_rets.mean()*252)/(p_rets.std()*np.sqrt(252)):.2f}")
    col4.metric("Max Drawdown", f"{(p_cum / p_cum.expanding().max() - 1).min():.1%}")

    st.markdown('<div class="sub-header">Efficient Frontier Analysis</div>', unsafe_allow_html=True)
    
    # Frontier Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.sqrt(np.diag(S)), y=mu, mode='markers+text', text=mu.index, name="Assets", marker=dict(color='red')))
    port_vol = np.sqrt(w_arr.T @ S.values @ w_arr)
    fig.add_trace(go.Scatter(x=[port_vol], y=[w_arr @ mu], mode='markers', name="Optimized Portfolio", marker=dict(size=15, color='gold', symbol='star')))
    fig.update_layout(template="plotly_white", xaxis_title="Annualized Risk (Std Dev)", yaxis_title="Annualized Expected Return")
    st.plotly_chart(fig, use_container_width=True)
    
    

    # DROPDOWN: Risk Information
    with st.expander("🔍 Risk Attribution & Detailed Allocation"):
        st.write("This dropdown provides a granular look at how the geopolitical overlay adjusted your portfolio.")
        c_alt1, c_alt2 = st.columns(2)
        with c_alt1:
            st.write("**Final Weights Table**")
            st.table(pd.Series(final_weights, name="Weight (%)").map(lambda x: f"{x:.2%}"))
        with c_alt2:
            st.write("**Geopolitical Factor Analysis**")
            st.info(f"Total Risk Events: {len(geo_events)}")
            st.info(f"Overlay Intensity: {geo_intensity}x")
            st.write("The current allocation has been 'shrunk' away from high-volatility sectors based on your sidebar selections.")

    # Performance
    st.markdown('<div class="sub-header">Portfolio Performance Growth</div>', unsafe_allow_html=True)
    st.line_chart(p_cum)

    # DISCLAIMER SECTION
    st.divider()
    with st.expander("⚠️ Legal Disclaimer & Risk Warning"):
        st.markdown("""
        **Notice:** This tool is for **informational and educational purposes only**. 
        * **Not Financial Advice:** The quantitative models (Black-Litterman, Mean-Variance Optimization) are based on historical data and mathematical assumptions that may not reflect future market conditions.
        * **Geopolitical Risk:** The 'Geopolitical Risk Overlay' is a qualitative heuristic and should be used as a 'what-if' scenario tool rather than a definitive risk prediction.
        * **Investment Risk:** All trading involves risk. Past performance is no guarantee of future results. 
        * **Consult a Professional:** Always consult with a licensed financial advisor before making significant investment decisions.
        """)

else:
    st.error("Engine failure: Check ticker connectivity or Yahoo Finance rate limits.")
