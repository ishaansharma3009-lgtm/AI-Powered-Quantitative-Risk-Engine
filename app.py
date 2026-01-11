import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models, expected_returns
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Asset Lab", layout="wide")

# --- HEADER ---
st.title("🏛️ Strategic Asset Allocation & Risk Engine")
st.markdown("---")

# --- SIDEBAR SETTINGS ---
with st.sidebar:
    st.header("Portfolio Settings")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Enter Tickers (Comma separated)", default_tickers)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🛡️ Compliance Constraints")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    min_cap = st.slider("Min Weight per Stock (%)", 0, 10, 2) / 100
    
    st.divider()
    download_placeholder = st.empty()
    st.divider()
    st.caption("📂 **Institutional Disclosure**")
    st.caption("Developed by Ishaan Sharma | Asset Management Tool")

# --- DATA FETCHING ---
ticker_list = [t.strip() for t in assets.split(",")]

@st.cache_data
def get_global_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    data = yf.download(all_tickers, start=start)['Close']
    benchmark = data["^GSPC"].ffill()
    assets_data = data.drop(columns=["^GSPC"]).ffill()
    return assets_data, benchmark

try:
    data, benchmark = get_global_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()

    # --- OPTIMIZATION ---
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    ef = EfficientFrontier(mu, S, weight_bounds=(min_cap, max_cap))
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    
    # --- PERFORMANCE CALCULATIONS ---
    weights_arr = np.array(list(clean_weights.values()))
    portfolio_daily_returns = (returns * weights_arr).sum(axis=1)
    
    portfolio_cum = (1 + portfolio_daily_returns).cumprod()
    bench_cum = (1 + bench_returns).cumprod()

    # Yearly Performance Table
    p_daily_df = portfolio_daily_returns.to_frame(name='Returns')
    yearly_perf = p_daily_df.groupby(p_daily_df.index.year).apply(lambda x: (1 + x).prod() - 1)
    yearly_perf.columns = ['Portfolio Return']

    # Max Drawdown
    rolling_max = portfolio_cum.cummax()
    drawdown = (portfolio_cum - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    # Global Metrics
    p_ret = (portfolio_cum.iloc[-1] - 1)
    p_vol = portfolio_daily_returns.std() * np.sqrt(252)
    p_sharpe = (portfolio_daily_returns.mean() * 252) / p_vol
    var_95 = -(portfolio_daily_returns.mean() - 1.645 * portfolio_daily_returns.std())

    # --- DISPLAY ---
    st.subheader("📊 Institutional Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Return", f"{p_ret:.1%}", f"{p_ret - (bench_cum.iloc[-1]-1):.1%} vs S&P")
    m2.metric("Annualized Vol", f"{p_vol:.1%}", delta="Lower is better", delta_color="normal" if p_vol < 0.20 else "inverse")
    m3.metric("Sharpe Ratio", f"{p_sharpe:.2f}", "Risk-Adjusted")
    m4.metric("Max Drawdown", f"{max_drawdown:.1%}", "Worst Peak-to-Trough", delta_color="inverse")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("📈 Optimal Weights")
        weights_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(weights_df[weights_df['Weight']>0], values='Weight', names=weights_df[weights_df['Weight']>0].index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
        
        st.subheader("🗓️ Yearly Backtest")
        st.table(yearly_perf.style.format("{:.2%}"))

    with col_right:
        st.subheader("📊 Growth of $10,000")
        fig_bench = go.Figure()
        fig_bench.add_trace(go.Scatter(x=portfolio_cum.index, y=portfolio_cum*10000, name="Portfolio", line=dict(color='#FFD700', width=3)))
        fig_bench.add_trace(go.Scatter(x=bench_cum.index, y=bench_cum*10000, name="S&P 500", line=dict(color='white', dash='dash')))
        fig_bench.update_layout(template="plotly_dark", height=450)
        st.plotly_chart(fig_bench, use_container_width=True)

    # --- DOWNLOAD ---
    csv = weights_df.to_csv().encode('utf-8')
    download_placeholder.download_button("📥 Export Portfolio Report", data=csv, file_name='portfolio_analysis.csv')

except Exception as e:
    st.error(f"Analysis Error: {e}")






