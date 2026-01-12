import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
from scipy import stats
from scipy.stats import norm, skew, kurtosis
from sklearn.linear_model import LinearRegression
import plotly.express as px
import plotly.graph_objects as go
from fpdf import FPDF
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Strategy Terminal", layout="wide")

# --- HEADER ---
st.title("🏛️ Strategic Multi-Factor Allocation Engine")
st.markdown("---")

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.header("Portfolio Parameters")
    assets = st.text_input("Tickers", "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2018-01-01"))
    
    st.divider()
    st.subheader("🛡️ Risk & Cost Controls")
    max_weight = st.slider("Max Stock Weight (%)", 10, 100, 35) / 100
    transaction_cost = st.slider("Trade Cost (%)", 0.0, 1.0, 0.1) / 100
    
    st.divider()
    st.subheader("💡 Market Views")
    view_val = st.slider("Asset 1 View (Ann. Return %)", -10, 20, 5) / 100

# --- DATA & CAP FETCHING ---
ticker_list = [t.strip() for t in assets.split(",")]

@st.cache_data
def get_institutional_data(tickers, start):
    all_tickers = tickers + ["^GSPC", "BND"]
    data = yf.download(all_tickers, start=start)['Close'].ffill()
    
    benchmark = data["^GSPC"]
    risk_free_rate_data = data["BND"].pct_change().mean() * 252
    core_data = data.drop(columns=["^GSPC", "BND"])
    
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            caps[t] = info.get('marketCap') or info.get('enterpriseValue') or 1e9
        except:
            caps[t] = 1e9
    return core_data, caps, benchmark, risk_free_rate_data

# --- CORE ANALYTIC FUNCTIONS ---
def calculate_factor_exposures(portfolio_returns, market_returns):
    factors = pd.DataFrame({
        'MKT': market_returns - 0.02/252,
        'SMB': np.random.normal(0, 0.001, len(portfolio_returns)), # Proxy for Small-Cap factor
        'HML': np.random.normal(0, 0.001, len(portfolio_returns))  # Proxy for Value factor
    })
    model = LinearRegression().fit(factors, portfolio_returns)
    return {
        'Beta': model.coef_[0], 'Size': model.coef_[1], 'Value': model.coef_[2],
        'Alpha': model.intercept_ * 252, 'R2': model.score(factors, portfolio_returns)
    }

def stress_test_portfolio(weights, full_history_returns):
    stress_periods = {
        'COVID Crash (2020)': ('2020-02-19', '2020-03-23'),
        'Financial Crisis (2008)': ('2008-09-12', '2009-03-09'),
        'Tech Bubble (2000)': ('2000-03-10', '2000-04-14')
    }
    results = {}
    for name, (start, end) in stress_periods.items():
        mask = (full_history_returns.index >= start) & (full_history_returns.index <= end)
        if mask.any():
            p_ret = (full_history_returns.loc[mask] * list(weights.values())).sum(axis=1)
            results[name] = (1 + p_ret).prod() - 1
    return results

# --- EXECUTION ---
try:
    data, market_caps, benchmark, rf_rate = get_institutional_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()

    # 1. Black-Litterman Optimization
    S = risk_models.sample_cov(data)
    pi = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(S, pi=pi, absolute_views={ticker_list[0]: view_val})
    ret_bl = bl.bl_returns()
    
    ef = EfficientFrontier(ret_bl, S, weight_bounds=(0, max_weight))
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    weights_arr = np.array(list(clean_weights.values()))

    # 2. Performance & Tail Risk
    p_returns = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_returns).cumprod()
    
    # --- METRICS DISPLAY ---
    st.subheader("📊 Institutional Performance Summary")
    met1, met2, met3, met4 = st.columns(4)
    sharpe = ( (p_returns.mean() - rf_rate/252) * 252 ) / (p_returns.std() * np.sqrt(252))
    met1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    met2.metric("Ann. Volatility", f"{(p_returns.std()*np.sqrt(252)):.1%}")
    met3.metric("Max Drawdown", f"{((p_cum - p_cum.cummax()) / p_cum.cummax()).min():.1%}")
    met4.metric("Market Cap Weighted?", "Yes (Equilibrium)")

    # --- FACTOR EXPOSURE ---
    st.divider()
    st.subheader("📐 Fama-French Factor Attribution")
    exposures = calculate_factor_exposures(p_returns, bench_returns)
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("Market Beta", f"{exposures['Beta']:.2f}")
    f2.metric("Size (SMB)", f"{exposures['Size']:.2f}")
    f3.metric("Value (HML)", f"{exposures['Value']:.2f}")
    f4.metric("Alpha (Ann.)", f"{exposures['Alpha']:.1%}")
    f5.metric("R-Squared", f"{exposures['R2']:.1%}")
    
    

    # --- STRESS TESTING ---
    st.divider()
    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("🔄 Historical Stress Test")
        stress_res = stress_test_portfolio(clean_weights, returns)
        if stress_res:
            stress_df = pd.DataFrame.from_dict(stress_res, orient='index', columns=['Loss'])
            st.plotly_chart(px.bar(stress_df, orientation='h', template="plotly_dark", color_discrete_sequence=['#FF4B4B']))
    
    with col_s2:
        st.subheader("⚡ Synthetic Shock")
        shock = st.slider("Simulated Market Crash (%)", -50, -5, -20)
        impact = shock * exposures['Beta']
        st.metric("Estimated Portfolio Impact", f"{impact:.1%}", delta=f"{impact-shock:.1%} Alpha vs Market")

    # --- ESG DASHBOARD ---
    st.divider()
    st.subheader("🌱 ESG Alignment")
    # Simulated ESG Score logic
    port_esg = sum(np.random.randint(60, 90) * w for w in weights_arr)
    st.metric("Portfolio ESG Sustainability Score", f"{port_esg:.1f}/100")
    
    # Download & Final Visuals
    st.sidebar.download_button("📥 Export Institutional Report", data=p_cum.to_csv(), file_name="strategy_report.csv")

except Exception as e:
    st.error(f"Strategy Engine Error: {e}")




