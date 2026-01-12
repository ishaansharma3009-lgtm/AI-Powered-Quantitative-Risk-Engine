import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier
from scipy.stats import norm, skew, kurtosis
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Asset Management Terminal", layout="wide")
st.title("🏛️ Portfolio Strategy & Risk Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Parameters")
    assets = st.text_input("Tickers (comma separated)", "AAPL, MSFT, JPM, ASML, NESN.SW")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    max_weight = st.slider("Max Stock Weight (%)", 10, 100, 35) / 100

# --- DATA ENGINE ---
@st.cache_data
def get_clean_data(tickers, start):
    # Fetching assets + Benchmark (S&P 500)
    all_t = tickers + ["^GSPC"]
    raw = yf.download(all_t, start=start)['Close'].ffill()
    
    # Calculate daily returns and drop NaNs
    all_rets = raw.pct_change().dropna()
    
    # Portfolio components and Benchmark separation
    portfolio_rets = all_rets[tickers]
    benchmark_rets = all_rets["^GSPC"]
    
    return portfolio_rets, benchmark_rets

def get_cf_var(res, conf=0.95):
    s, k = skew(res), kurtosis(res)
    z = norm.ppf(conf)
    # Cornish-Fisher adjustment for non-normal distributions
    z_cf = (z + (z**2 - 1) * s/6 + (z**3 - 3*z) * k/24 - (2*z**3 - 5*z) * s**2/36)
    return -(res.mean() - z_cf * res.std())

# --- EXECUTION ---
ticker_list = [t.strip() for t in assets.split(",") if t.strip()]

try:
    rets, spy_rets = get_clean_data(ticker_list, start_date)

    # 1. OPTIMIZATION
    mu = expected_returns.mean_historical_return(rets)
    S = risk_models.sample_cov(rets)
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    weights = ef.max_sharpe()
    clean_w = ef.clean_weights()
    weights_arr = np.array(list(clean_w.values()))

    # 2. PERFORMANCE CALCULATIONS
    p_rets = (rets * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    spy_cum = (1 + spy_rets).cumprod()
    
    # Metrics
    sharpe = (p_rets.mean() * 252) / (p_rets.std() * np.sqrt(252))
    cf_var = get_cf_var(p_rets)

    # --- UI: METRICS ---
    st.subheader("📊 Performance vs. S&P 500")
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio Sharpe", f"{sharpe:.2f}")
    m2.metric("Cornish-Fisher VaR", f"{cf_var:.2%}")
    m3.metric("Max Drawdown", f"{((p_cum - p_cum.cummax())/p_cum.cummax()).min():.1%}")

    # --- UI: COMPARISON CHART ---
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Optimized Portfolio"))
    fig_comp.add_trace(go.Scatter(x=spy_cum.index, y=spy_cum, name="S&P 500 (Benchmark)"))
    fig_comp.update_layout(template="plotly_dark", title="Cumulative Return Comparison")
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- UI: CORRELATION HEATMAP ---
    st.divider()
    st.subheader("🧩 Asset Correlation Matrix")
    corr_matrix = rets.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=".2f", aspect="auto", 
                         color_continuous_scale='RdBu_r', template="plotly_dark")
    st.plotly_chart(fig_corr, use_container_width=True)
    

    # --- UI: COVID STRESS TEST ---
    st.divider()
    st.subheader("🔄 COVID-19 Crisis Impact (2020)")
    covid_mask = (p_rets.index >= '2020-02-19') & (p_rets.index <= '2020-03-23')
    if covid_mask.any():
        p_crash = (1 + p_rets[covid_mask]).prod() - 1
        spy_crash = (1 + spy_rets[covid_mask]).prod() - 1
        st.write(f"**Portfolio Return during COVID crash:** {p_crash:.2%}")
        st.write(f"**S&P 500 Return during COVID crash:** {spy_crash:.2%}")

    # --- EXPORT ---
    st.sidebar.download_button("📂 Export Results (CSV)", 
                               data=rets.to_csv().encode('utf-8'), 
                               file_name="portfolio_data.csv")

except Exception as e:
    st.error(f"Engine Error: {e}")






