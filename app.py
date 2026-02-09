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
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">Asset Management & Quantitative Risk Engine</div>', unsafe_allow_html=True)
st.caption("Advanced Portfolio Optimization with Geopolitical Risk Integration")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.markdown("### Strategic Parameters")
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
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider(f"Expected Return (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification (L2) Penalty", 0.0, 2.0, 0.5)

# --- RE-ENGINEERED DATA FETCHING ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start):
    if not tickers:
        return pd.DataFrame(), pd.Series(), {}
    
    all_tickers = list(set(tickers + ["^GSPC"]))
    try:
        # Fetching with group_by='ticker' provides a predictable structure
        raw_data = yf.download(all_tickers, start=start, progress=False, group_by='ticker')
        
        close_prices = pd.DataFrame()
        for t in all_tickers:
            try:
                if len(all_tickers) > 1:
                    close_prices[t] = raw_data[t]['Close']
                else:
                    close_prices[t] = raw_data['Close']
            except Exception:
                continue
                
        if close_prices.empty:
            return pd.DataFrame(), pd.Series(), {}
            
        clean_df = close_prices.ffill().dropna()
        benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
        assets_data = clean_df.drop(columns=["^GSPC"]) if "^GSPC" in clean_df.columns else clean_df
        
        # Static Market Cap Estimates
        fixed_caps = {
            'AAPL': 3.0e12, 'MSFT': 2.8e12, 'JPM': 0.5e12, 'MC.PA': 0.4e12, 
            'ASML': 0.35e12, 'NESN.SW': 0.3e12, 'GOOGL': 1.8e12, 'AMZN': 1.6e12
        }
        mcaps = {t: fixed_caps.get(t, 1e11) for t in tickers if t in assets_data.columns}
        
        return assets_data, benchmark, mcaps
    except Exception as e:
        st.error(f"Market Data Error: {e}")
        return pd.DataFrame(), pd.Series(), {}

# --- OPTIMIZATION LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    if not events or intensity <= 0.5:
        return weights
    
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7},
        'Financials': {'Currency Volatility': 0.6, 'Middle East Instability': 0.3},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8},
        'Healthcare': {'EU Regulation Shift': 0.5}
    }
    
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Consumer', 'ASML': 'Semiconductors', 'NESN.SW': 'Healthcare'
    }
    
    adjustments = {}
    for ticker, weight in weights.items():
        sector = ticker_sectors.get(ticker, 'Technology')
        risk_score = sum(sector_risk.get(sector, {}).get(e, 0.1) for e in events)
        reduction = 1 - (risk_score * intensity * 0.1)
        adjustments[ticker] = max(0.01, weight * reduction)
    
    total = sum(adjustments.values())
    return {k: v/total for k, v in adjustments.items()}

# --- EXECUTION FLOW ---
try:
    prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    
    if not prices.empty:
        # 1. Statistical Priors
        S = risk_models.CovarianceShrinkage(prices).ledoit_wolf()
        delta = 2.5 # Risk aversion coefficient
        
        # Black-Litterman
        try:
            prior_rets = black_litterman.market_implied_prior_returns(market_caps, delta, S)
            bl = black_litterman.BlackLittermanModel(
                S, pi=prior_rets, 
                absolute_views={view_ticker: view_return},
                omega="idzorek",
                view_confidences=[min(view_conf, 0.99)]
            )
            bl_mu = bl.bl_returns()
        except Exception as e:
            bl_mu = prices.pct_change().mean() * 252

        # 2. Mean-Variance Optimization
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        raw_weights = ef.max_sharpe()
        optimized_weights = ef.clean_weights()

        # 3. Geopolitical Overlay
        final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)

        # 4. Results & Performance
        weights_arr = np.array([final_weights.get(t, 0) for t in prices.columns])
        returns = prices.pct_change().dropna()
        p_rets = (returns * weights_arr).sum(axis=1)
        p_cum = (1 + p_rets).cumprod()
        
        # Metrics
        ann_ret = p_rets.mean() * 252
        ann_vol = p_rets.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
        
        # --- UI LAYOUT ---
        col1, col2, col3 = st.columns(3)
        col1.markdown(f'<div class="metric-card">Sharpe Ratio<br><h3>{sharpe:.2f}</h3></div>', unsafe_allow_html=True)
        col2.markdown(f'<div class="metric-card">Annual Return<br><h3>{ann_ret:.1%}</h3></div>', unsafe_allow_html=True)
        col3.markdown(f'<div class="metric-card">Annual Vol<br><h3>{ann_vol:.1%}</h3></div>', unsafe_allow_html=True)

        

[Image of efficient frontier plot]


        # Allocation Plot
        st.markdown('<div class="sub-header">Strategy Allocation</div>', unsafe_allow_html=True)
        fig_alloc = px.pie(
            names=list(final_weights.keys()), 
            values=list(final_weights.values()),
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Prism
        )
        st.plotly_chart(fig_alloc, use_container_width=True)

        # Performance Trace
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Optimized Portfolio", line=dict(color='#1E3A8A')))
        if not bench_prices.empty:
            b_cum = (1 + bench_prices.pct_change()).cumprod()
            fig_perf.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="Benchmark (S&P500)", line=dict(color='#94A3B8', dash='dot')))
        st.plotly_chart(fig_perf, use_container_width=True)

    else:
        st.info("Awaiting input or data fetch...")

except Exception as e:
    st.error(f"Engine Failure: {e}")
