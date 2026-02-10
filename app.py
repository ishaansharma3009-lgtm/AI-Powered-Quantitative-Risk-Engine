import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings

warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Asset Management & Quantitative Risk Engine", 
    layout="wide",
    page_icon="📊"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: 700; color: #1E3A8A; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.5rem; font-weight: 600; color: #374151; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #E5E7EB; padding-bottom: 0.5rem; }
    .metric-card { background: linear-gradient(135deg, #1E3A8A 0%, #3B82F6 100%); padding: 1.2rem; border-radius: 10px; color: white; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .stButton button { background: #1E3A8A; color: white; border-radius: 5px; width: 100%; }
    .info-box { background-color: #F3F4F6; padding: 1rem; border-radius: 8px; border-left: 4px solid #3B82F6; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Asset Management & Quantitative Risk Engine</div>', unsafe_allow_html=True)
st.caption("Advanced Portfolio Optimization with Geopolitical Risk Integration")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.markdown("### Strategic Parameters")
    # Corrected default tickers to ensure they are high-liquidity recognized symbols
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip().upper() for t in assets.split(",") if t.strip()]
    
    start_date = st.date_input("Analysis Start", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.markdown("### Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### Black-Litterman View")
    # Dynamic selection based on input tickers
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider(f"Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification (L2) Penalty", 0.0, 2.0, 0.5)

# --- IMPROVED DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start):
    """Simplified data fetching to avoid MultiIndex issues"""
    if not tickers:
        return pd.DataFrame(), pd.Series(), {}
    
    search_tickers = list(set(tickers + ["^GSPC"]))
    
    try:
        # Fetching only 'Close' directly avoids the MultiIndex formatting trap
        raw_data = yf.download(search_tickers, start=start, progress=False)["Close"]
        
        if raw_data.empty:
            return pd.DataFrame(), pd.Series(), {}
            
        # Ensure it's a DataFrame (single ticker returns Series)
        if isinstance(raw_data, pd.Series):
            raw_data = raw_data.to_frame()

        # Forward fill and drop any days where data is missing (aligns global exchanges)
        clean_df = raw_data.ffill().dropna()
        
        if clean_df.empty:
            return pd.DataFrame(), pd.Series(), {}
            
        # Separate benchmark and assets
        benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
        assets_data = clean_df.drop(columns=["^GSPC"]) if "^GSPC" in clean_df.columns else clean_df
        
        # Market Cap Estimates (in USD billions)
        fixed_caps = {
            'AAPL': 3000, 'MSFT': 2800, 'JPM': 500, 'MC.PA': 400, 
            'ASML': 350, 'NESN.SW': 300, 'GOOGL': 1800, 'AMZN': 1600,
            'TSLA': 600, 'NVDA': 2200, 'V': 500, 'JNJ': 380,
            'XOM': 400, 'WMT': 450, 'PG': 350, 'MA': 400
        }
        
        # Map caps to final assets in dataframe
        mcaps = {t: fixed_caps.get(t, 100) * 1e9 for t in assets_data.columns}
        
        return assets_data, benchmark, mcaps
        
    except Exception as e:
        st.error(f"Data Fetching Error: {str(e)}")
        return pd.DataFrame(), pd.Series(), {}

# --- OPTIMIZATION LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    if not events or intensity <= 0.5:
        return weights
    
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7, 'Trade Policy Changes': 0.6},
        'Financials': {'Currency Volatility': 0.6, 'Middle East Instability': 0.3, 'Trade Policy Changes': 0.4},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8, 'Trade Policy Changes': 0.7},
        'Healthcare': {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3},
        'Automotive': {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7},
        'Consumer': {'Supply Chain Disruption': 0.5, 'Currency Volatility': 0.3},
        'Energy': {'Middle East Instability': 0.8, 'Trade Policy Changes': 0.6}
    }
    
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Consumer', 'ASML': 'Semiconductors', 'NESN.SW': 'Healthcare'
    }
    
    adjustments = {}
    for ticker, weight in weights.items():
        sector = ticker_sectors.get(ticker, 'Technology')
        risk_score = sum(sector_risk.get(sector, {}).get(event, 0.1) for event in events)
        reduction_factor = 1 - (risk_score * intensity * 0.15)
        adjustments[ticker] = max(0.01, weight * reduction_factor)
    
    total = sum(adjustments.values())
    return {k: v/total for k, v in adjustments.items()} if total > 0 else weights

def plot_efficient_frontier(mu, S, risk_free_rate=0.02):
    try:
        ef = EfficientFrontier(mu, S)
        # Point A: Min Vol
        ef.min_volatility()
        min_vol = ef.portfolio_performance(risk_free_rate=risk_free_rate)
        # Point B: Max Sharpe
        ef_s = EfficientFrontier(mu, S)
        ef_s.max_sharpe()
        max_sharpe = ef_s.portfolio_performance(risk_free_rate=risk_free_rate)
        
        # Plotly implementation
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[min_vol[1]], y=[min_vol[0]], mode='markers', name='Min Vol', marker=dict(size=15, color='green')))
        fig.add_trace(go.Scatter(x=[max_sharpe[1]], y=[max_sharpe[0]], mode='markers', name='Max Sharpe', marker=dict(size=15, color='orange', symbol='star')))
        
        fig.update_layout(title="Efficient Frontier Map", xaxis_title="Volatility", yaxis_title="Return", template="plotly_white")
        return fig
    except:
        return None

# --- MAIN EXECUTION ---
try:
    if not ticker_list:
        st.info("👈 Enter tickers in the sidebar to start.")
        st.stop()
    
    with st.spinner("Crunching market data..."):
        prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    
    if prices.empty:
        st.error("❌ No data found. Try check symbols (e.g., NESN.SW instead of NES).")
        st.stop()

    # 1. Risk Model
    S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
    
    # 2. Black-Litterman
    try:
        pi = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
        bl = black_litterman.BlackLittermanModel(S, pi=pi, absolute_views={view_ticker: view_return}, view_confidences=[view_conf])
        bl_mu = bl.bl_returns()
    except:
        bl_mu = prices.pct_change().mean() * 252

    # 3. MVO Optimization
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    optimized_weights = ef.max_sharpe()
    
    # 4. Overlay & Metrics
    final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)
    
    # Performance Calcs
    weights_arr = np.array([final_weights[t] for t in prices.columns])
    p_rets = (prices.pct_change().dropna() * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    
    # --- UI DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Expected Annual Return", f"{p_rets.mean()*252:.1%}")
    col2.metric("Annual Volatility", f"{p_rets.std()*np.sqrt(252):.1%}")
    col3.metric("Sharpe Ratio", f"{(p_rets.mean()*252)/ (p_rets.std()*np.sqrt(252)):.2f}")

    st.markdown('<div class="sub-header">Portfolio Allocation</div>', unsafe_allow_html=True)
    fig_pie = px.pie(names=list(final_weights.keys()), values=list(final_weights.values()), hole=0.4)
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown('<div class="sub-header">Cumulative Performance</div>', unsafe_allow_html=True)
    st.line_chart(p_cum)

except Exception as e:
    st.error(f"🚨 Engine Error: {str(e)}")
