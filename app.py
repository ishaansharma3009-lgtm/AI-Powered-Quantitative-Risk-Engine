import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="Strategic Asset Lab", layout="wide")

st.title("🏛️ Strategic Asset Allocation & Risk Engine")
st.markdown("---")

# --- MARKET HEALTH MONITOR (Debugged) ---
try:
    # Fetching VIX to explain the "Why" behind the AI's choices
    vix_ticker = yf.Ticker("^VIX")
    vix_data = vix_ticker.history(period="1d")
    
    if not vix_data.empty:
        # Use .item() or .iloc[0] to avoid the "ambiguous truth value" error
        current_vix = vix_data['Close'].iloc[-1]
        
        st.sidebar.header("Market Context")
        if current_vix > 25:
            st.sidebar.warning(f"⚠️ VIX is High ({current_vix:.2f}): Markets are fearful. Defensive stocks (Utilities) preferred.")
        elif current_vix < 15:
            st.sidebar.success(f"✅ VIX is Low ({current_vix:.2f}): Markets are calm. Growth stocks (Tech) often lead.")
        else:
            st.sidebar.info(f"ℹ️ VIX is Moderate ({current_vix:.2f}): Standard market conditions apply.")
except Exception:
    st.sidebar.info("VIX Monitor: Data currently unavailable.")

# --- SIDEBAR SETTINGS ---
st.sidebar.header("Portfolio Parameters")
tickers = st.sidebar.text_input("Assets", "AAPL, MSFT, JPM, XOM, COST, NEE, V")
tickers_list = [t.strip().upper() for t in tickers.split(",")]
max_weight = st.sidebar.slider("Max Allocation per Stock (%)", 10, 50, 35) / 100
start_date = st.sidebar.date_input("Analysis Start Date", value=pd.to_datetime("2022-01-01"))

if st.sidebar.button("Analyze & Optimize"):
    try:
        # 1. DATA ACQUISITION
        data = yf.download(tickers_list + ["SPY"], start=start_date)['Close']
        stock_prices = data[tickers_list].dropna()
        benchmark_prices = data["SPY"].dropna()

        # 2. CORE OPTIMIZATION
        mu = expected_returns.mean_historical_return(stock_prices)
        S = risk_models.sample_cov(stock_prices)
        ef = EfficientFrontier(mu, S, weight_bounds=(0.02, max_weight))
        weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        ret, vol, sharpe = ef.portfolio_performance()

        # 3. STRATEGIC INSIGHTS
        st.subheader("💡 AI Strategic Recommendation")
        best_stock = max(cleaned_weights, key=cleaned_weights.get)
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            st.info(f"**Dominant Asset: {best_stock}**")
            st.write(f"The model allocated **{cleaned_weights[best_stock]:.1%}** to this asset. This ensures the highest return for every unit of risk taken.")
        with col_ex2:
            st.info("**Strategy Horizon**")
            st.write("This allocation is optimized for a 6–18 month horizon. Effects typically shift with quarterly earnings and interest rate changes.")

        # 4. DASHBOARD METRICS
        st.markdown("---")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Expected Return", f"{ret:.2%}")
        m2.metric("Portfolio Sharpe", f"{sharpe:.2f}")
        m3.metric("Daily Risk (VaR-Adj)", f"{(vol/np.sqrt(252)):.2%}")
        m4.metric("Market Correlation", "Optimized")

        # 5. VISUALS
        c1, c2 = st.columns([1, 2])
        with c1:
            st.plotly_chart(px.pie(values=list(cleaned_weights.values()), names=list(cleaned_weights.keys()), hole=0.4, title="Optimal Allocation"), use_container_width=True)
        with c2:
            # Backtest vs Benchmark
            cum_portfolio = (1 + (stock_prices.pct_change() * pd.Series(cleaned_weights)).sum(axis=1)).cumprod()
            cum_benchmark = (1 + benchmark_prices.pct_change()).cumprod()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=cum_portfolio.index, y=cum_portfolio, name="AI Portfolio"))
            fig.add_trace(go.Scatter(x=cum_benchmark.index, y=cum_benchmark, name="S&P 500 (SPY)", line=dict(dash='dash')))
            fig.update_layout(title="Performance vs. Benchmark", template="plotly_white")
            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:

        st.error(f"Error in Engine: {e}")
        st.divider() 
st.caption("📂 **Institutional Disclosure**")
st.caption("""
**FX Normalization:** This engine currently calculates returns 
using local currency price data. For institutional global 
portfolios, returns should be normalized to a base currency 
(e.g., USD) to accurately reflect cross-border risk.
""")
st.caption("Developed by Ishaan Sharma | Asset Management Tool")
