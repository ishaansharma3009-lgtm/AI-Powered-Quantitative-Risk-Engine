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
    ticker_list = [t.strip() for t in assets.split(",")]
    view_ticker = st.selectbox("Select Asset for View", ticker_list)
    view_return = st.slider(f"Expected Annual Return for {view_ticker} (%)", -20, 40, 10) / 100

    st.divider()
    st.subheader("🛡️ Compliance Constraints")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    
    download_placeholder = st.empty()

# --- DATA ENGINE ---
@st.cache_data
def get_market_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    data = yf.download(all_tickers, start=start)['Close']
    benchmark = data["^GSPC"].ffill()
    assets_data = data.drop(columns=["^GSPC"]).ffill()
    
    # Using real-time Market Caps for Equilibrium
    mcaps = {}
    for t in tickers:
        try:
            # Fallback to 100B if API fails to prevent math errors
            mcaps[t] = yf.Ticker(t).info.get('marketCap', 100000000000)
        except:
            mcaps[t] = 100000000000
    return assets_data, benchmark, mcaps

# --- MAIN EXECUTION ---
try:
    data, benchmark, market_caps = get_market_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    
    # --- BLACK-LITTERMAN PROCESS ---
    # Fix for 'self.omega' error:omega must be a square array or string
    S = risk_models.sample_cov(data)
    delta = 2.5  # Standard Risk Aversion
    
    # 1. Market-Implied Prior Returns
    prior_returns = black_litterman.market_implied_prior_returns(market_caps, delta, S)
    
    # 2. Define Absolute Views
    views = {view_ticker: view_return}
    
    # 3. Initialize Black-Litterman with Omega 'Idzorek' method
    # This specifically addresses the error in your screenshot
    bl = black_litterman.BlackLittermanModel(
        S, 
        pi=prior_returns, 
        absolute_views=views, 
        omega="idzorek", # Automated uncertainty matrix calculation
        view_confidences=[0.5] # Default 50% confidence to stabilize math
    )
    
    # 4. Posterior Expected Returns
    bl_mu = bl.bl_returns()
    
    # --- OPTIMIZATION ---
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    weights_arr = np.array(list(clean_weights.values()))

    # --- VISUALIZATION ---
    col1, col2 = st.columns([1, 1.2])
    
    with col1:
        st.subheader("📈 Forward-Looking Allocation")
        weights_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(weights_df[weights_df['Weight']>0], values='Weight', 
                         names=weights_df[weights_df['Weight']>0].index, hole=0.4)
        fig_pie.update_layout(template="plotly_dark")
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        st.subheader("📊 Return Tilt (Prior vs. Posterior)")
        comparison_df = pd.DataFrame({"Market Equilibrium": prior_returns, "Black-Litterman": bl_mu})
        st.bar_chart(comparison_df)
        st.caption("How your active views modified the market's expected returns.")

    # --- RISK ANALYSIS ---
    st.divider()
    st.subheader("🧩 Diversification Matrix")
    fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)

    # --- CSV EXPORT ---
    csv = weights_df.to_csv().encode('utf-8')
    download_placeholder.download_button("📥 Export BL Report", data=csv, file_name='bl_strategy.csv')

except Exception as e:
    # This block captures and explains the errors seen in your screenshot
    st.error(f"Strategy Engine Error: {str(e)}")
    st.info("Check if tickers are valid and if the start date allows for sufficient data.")









