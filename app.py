import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Strategy Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Geopolitical Risk Engine")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.header("Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Tickers", default_tickers)
    ticker_list = [t.strip() for t in assets.split(",")]
    start_date = st.date_input("Analysis Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🌐 Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Geopolitical Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity Multiplier", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.subheader("💡 Black-Litterman View")
    view_ticker = st.selectbox("Select Asset for View", ticker_list)
    view_return = st.slider(f"Expected Return for {view_ticker} (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.subheader("🛡️ Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 2.0, 0.5)

# --- GEOPOLITICAL LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    """Adjust portfolio weights based on geopolitical risk exposure per sector."""
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7},
        'Financials': {'Currency Volatility': 0.6, 'Trade Policy Changes': 0.4},
        'Automotive': {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8},
        'Healthcare': {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3}
    }
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Automotive', 'ASML': 'Semiconductors', 
        'NESN.SW': 'Healthcare', '2330.TW': 'Semiconductors', '7203.T': 'Automotive'
    }
    
    adjustments = {}
    for ticker in tickers:
        sector = ticker_sectors.get(ticker, 'Technology')
        total_risk = sum(sector_risk.get(sector, {}).get(event, 0.1) for event in events)
        # Higher risk = reduce weight based on intensity
        reduction_factor = 1 - (total_risk * intensity * 0.15)
        adjustments[ticker] = max(0.01, weights.get(ticker, 0) * reduction_factor)
    
    # Re-normalize to 100%
    total = sum(adjustments.values())
    return {k: v/total for k, v in adjustments.items()} if total > 0 else weights

# --- DATA ENGINE ---
@st.cache_data
def get_clean_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    raw_data = yf.download(all_tickers, start=start)['Close']
    clean_df = raw_data.ffill().dropna()
    benchmark = clean_df["^GSPC"]
    assets_data = clean_df.drop(columns=["^GSPC"])
    mcaps = {t: yf.Ticker(t).info.get('marketCap', 1e11) for t in tickers}
    return assets_data, benchmark, mcaps

# --- EXECUTION ---
try:
    prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    returns = prices.pct_change().dropna()
    bench_returns = bench_prices.pct_change().dropna()
    
    # 1. Black-Litterman Logic
    S = risk_models.sample_cov(prices)
    prior_rets = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(
        S, pi=prior_rets, absolute_views={view_ticker: view_return}, 
        omega="idzorek", view_confidences=[view_conf]
    )
    bl_mu = bl.bl_returns()

    # 2. Base Optimization
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    optimized_weights = ef.max_sharpe()
    
    # 3. Geopolitical Adjustment
    final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)
    weights_arr = np.array([final_weights[t] for t in ticker_list])

    # 4. Performance Math
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    b_cum = (1 + bench_returns).cumprod()
    
    # Analytics
    ann_ret = p_rets.mean() * 252
    ann_vol = p_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()

    # --- UI: DASHBOARD ---
    st.subheader("📊 Performance & Geopolitical Analysis")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annualized Return", f"{ann_ret:.1%}")
    m2.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m3.metric("Max Drawdown", f"{max_dd:.1%}")
    m4.metric("Annualized Vol", f"{ann_vol:.1%}")

    # --- PERFORMANCE CHART ---
    st.divider()
    st.subheader("📈 Cumulative Growth: Strategy vs. S&P 500")
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Geo-Adjusted Strategy", line=dict(color='#00CC96', width=2.5)))
    fig_perf.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="S&P 500", line=dict(color='white', dash='dash')))
    fig_perf.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- ALLOCATION PIE ---
    st.divider()
    col_pie, col_risk = st.columns(2)
    with col_pie:
        st.subheader("🍕 Final Portfolio Allocation")
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(w_df, values='Weight', names=w_df.index, hole=0.4)
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)
        
    with col_risk:
        st.subheader("🛡️ Weight Shifts (Geo-Adjustment)")
        # Show shift from Base Optimized to Geo-Adjusted
        shift_df = pd.DataFrame({
            "Base Optimized": optimized_weights.values(),
            "Geo-Adjusted": final_weights.values()
        }, index=ticker_list)
        st.bar_chart(shift_df)
        st.caption("How your weights shifted due to Geopolitical Risks.")

except Exception as e:
    st.error(f"Critical Engine Error: {str(e)}")




