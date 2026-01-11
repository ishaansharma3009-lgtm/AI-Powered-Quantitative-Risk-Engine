import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.express as px

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Asset Lab", layout="wide")

# --- HEADER ---
st.title("🏛️ Strategic Asset Allocation & Risk Engine")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Portfolio Settings")
    
    # Global Asset Management Tickers
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Enter Tickers (Comma separated)", default_tickers)
    
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    
    st.divider()
    
    # Placeholder for Download Button (Defined later in code)
    download_placeholder = st.empty()

    st.divider()
    st.caption("📂 **Institutional Disclosure**")
    st.caption("""
    **FX Normalization:** This engine currently calculates returns 
    using local currency price data. For institutional global 
    portfolios, returns should be normalized to a base currency 
    (e.g., USD).
    """)
    st.caption("Developed by Ishaan Sharma | Asset Management Tool")

# --- DATA FETCHING ---
ticker_list = [t.strip() for t in assets.split(",")]

@st.cache_data
def get_data(tickers, start):
    data = yf.download(tickers, start=start)['Close']
    return data

try:
    data = get_data(ticker_list, start_date)
    returns = data.pct_change().dropna()

    # --- OPTIMIZATION LOGIC ---
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)

    ef = EfficientFrontier(mu, S)
    # Adding a constraint: No single asset more than 35% (Institutional Standard)
    ef.add_constraint(lambda x: x <= 0.35)
    
    # Optimize for Sharpe Ratio
    raw_weights = ef.max_sharpe()
    weights = ef.clean_weights()
    
    # --- CALCULATIONS FOR RISK METER ---
    weights_arr = np.array(list(weights.values()))
    portfolio_return = (weights_arr * returns.mean()).sum()
    portfolio_std = np.sqrt(np.dot(weights_arr.T, np.dot(returns.cov(), weights_arr)))
    
    # 95% Daily Value-at-Risk (VaR)
    var_95 = -(portfolio_return - 1.645 * portfolio_std)

    # --- DISPLAY RESULTS ---
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("📈 Optimal Allocations")
        weights_df = pd.DataFrame.from_dict(weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0] # Filter out 0% weights
        
        fig = px.pie(weights_df, values='Weight', names=weights_df.index, hole=0.4,
                     color_discrete_sequence=px.colors.sequential.RdBu)
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("🛡️ Risk Analysis")
        
        if var_95 < 0.02:
            status, color = "Low Risk", "normal"
        elif var_95 < 0.04:
            status, color = "Moderate Risk", "off"
        else:
            status, color = "High Risk", "inverse"

        st.metric(label="95% Daily Value-at-Risk (VaR)", value=f"{var_95:.2%}", delta=status, delta_color=color)
        st.info("This metric indicates the maximum expected loss over a 1-day period with 95% confidence.")
        
        # Display Weights Table
        st.dataframe(weights_df.style.format("{:.2%}"), use_container_width=True)

    # --- ACTIVATE DOWNLOAD BUTTON ---
    csv = weights_df.to_csv().encode('utf-8')
    download_placeholder.download_button(
        label="📥 Download Portfolio Report",
        data=csv,
        file_name='optimized_portfolio.csv',
        mime='text/csv',
    )

except Exception as e:
    st.error(f"Please check your tickers or date range. Error: {e}")



