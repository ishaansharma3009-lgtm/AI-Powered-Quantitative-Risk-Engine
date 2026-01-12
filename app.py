import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions

st.title("🏛️ Professional Strategy Lab")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    assets = st.text_input("Tickers", "AAPL, MSFT, JPM, MC.PA, ASML")
    ticker_list = [t.strip() for t in assets.split(",")]
    
    # Toggle to "un-mess" the model
    use_bl = st.toggle("Use Black-Litterman Logic", value=True)
    
    st.divider()
    view_ticker = st.selectbox("View Ticker", ticker_list)
    view_return = st.slider(f"Expected Return (%)", -20, 40, 10) / 100
    # FIX: Ensure confidence is handled as a list later
    view_conf = st.slider("View Confidence (%)", 10, 100, 40) / 100

# --- DATA ENGINE (FIXED SYNCHRONIZATION) ---
@st.cache_data
def get_synced_data(tickers):
    all_data = yf.download(tickers + ["^GSPC"], period="2y")['Close']
    # FIX: .dropna() here ensures all rows have data for ALL tickers (Length Match)
    synced = all_data.ffill().dropna() 
    bench = synced["^GSPC"]
    prices = synced.drop(columns=["^GSPC"])
    
    mcaps = {t: yf.Ticker(t).info.get('marketCap', 1e11) for t in tickers}
    return prices, bench, mcaps

try:
    prices, benchmark, market_caps = get_synced_data(ticker_list)
    returns = prices.pct_change().dropna()
    S = risk_models.sample_cov(prices) # Covariance matrix

    if use_bl:
        # BLACK-LITTERMAN ROUTE
        prior_rets = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
        bl = black_litterman.BlackLittermanModel(
            S, pi=prior_rets, 
            absolute_views={view_ticker: view_return},
            omega="idzorek",
            view_confidences=[view_conf] # FIX: Must be a list
        )
        mu = bl.bl_returns()
    else:
        # STANDARD ROUTE (Past performance only)
        mu = returns.mean() * 252

    # OPTIMIZATION
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 0.4))
    ef.add_objective(objective_functions.L2_reg, gamma=0.5)
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()

    # --- RESULTS ---
    p_rets = (returns * np.array(list(clean_weights.values()))).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    b_cum = (1 + benchmark.pct_change().dropna()).cumprod()

    # Visuals
    st.subheader("📊 Strategy vs Benchmark")
    chart_data = pd.DataFrame({"Strategy": p_cum, "S&P 500": b_cum})
    st.line_chart(chart_data)

    st.subheader("🍕 Final Allocation")
    st.write(clean_weights)

except Exception as e:
    st.error(f"Engine Stalled: {e}")




