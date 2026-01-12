import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, black_litterman
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Asset Lab", layout="wide")

# --- HEADER ---
st.title("🏛️ Strategic Asset Allocation & Black-Litterman Engine")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Portfolio Settings")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Enter Tickers", default_tickers)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("💡 Black-Litterman Views")
    st.caption("Tilt the portfolio away from historical bias using market-implied priors.")
    # Investor view: "I believe Asset 1 will return X% more than the market thinks"
    view_ticker = st.selectbox("Select Asset for View", default_tickers.split(", "))
    view_return = st.slider(f"Expected Annual Return for {view_ticker} (%)", -20, 40, 10) / 100
    conf = st.slider("View Confidence", 0.1, 1.0, 0.5)

    st.divider()
    st.subheader("🛡️ Compliance Constraints")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    
    download_placeholder = st.empty()

# --- DATA FETCHING ---
ticker_list = [t.strip() for t in assets.split(",")]

@st.cache_data
def get_market_data(tickers, start):
    # Fetch assets + S&P 500 for Market Equilibrium calculation
    all_tickers = tickers + ["^GSPC"]
    data = yf.download(all_tickers, start=start)['Close']
    benchmark = data["^GSPC"].ffill()
    assets_data = data.drop(columns=["^GSPC"]).ffill()
    
    # Get Market Caps for Black-Litterman Equilibrium
    mcaps = {}
    for t in tickers:
        try:
            mcaps[t] = yf.Ticker(t).info.get('marketCap', 1e11) # Default to 100B if not found
        except:
            mcaps[t] = 1e11
    return assets_data, benchmark, mcaps

try:
    data, benchmark, market_caps = get_market_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    
    # --- BLACK-LITTERMAN CALCULATION ---
    # 1. Get Market-Implied Prior Returns (Equilibrium)
    S = risk_models.sample_cov(data)
    delta = 2.5 # Risk aversion coefficient
    prior_returns = black_litterman.market_implied_prior_returns(market_caps, delta, S)
    
    # 2. Integrate User Views
    views = {view_ticker: view_return}
    bl = black_litterman.BlackLittermanModel(S, pi=prior_returns, absolute_views=views, omega="Idzorek")
    
    # 3. Calculate Posterior (Updated) Expected Returns
    bl_mu = bl.bl_returns()
    
    # --- OPTIMIZATION ---
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    weights_arr = np.array(list(clean_weights.values()))

    # --- PERFORMANCE ---
    portfolio_daily_returns = (returns * weights_arr).sum(axis=1)
    portfolio_cum = (1 + portfolio_daily_returns).cumprod()
    
    # --- DISPLAY ---
    st.subheader("📊 Strategic Weights (Black-Litterman Optimized)")
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Compare Prior (Historical) vs BL Returns (Future tilt)
        comparison_df = pd.DataFrame({"Market Prior": prior_returns, "Black-Litterman Tilt": bl_mu})
        st.bar_chart(comparison_df)
        st.caption("This chart shows how your 'Views' tilted the market equilibrium returns.")

    with col2:
        weights_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(weights_df[weights_df['Weight']>0], values='Weight', names=weights_df[weights_df['Weight']>0].index, hole=0.4)
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    # --- CORRELATION HEATMAP ---
    st.divider()
    st.subheader("🧩 Diversification Matrix")
    
    fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- EXPORT ---
    csv = weights_df.to_csv().encode('utf-8')
    download_placeholder.download_button("📥 Export BL Strategy", data=csv, file_name='bl_portfolio.csv')

except Exception as e:
    st.error(f"Black-Litterman Engine Error: {e}")








