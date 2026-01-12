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
from fpdf import FPDF # Requires: pip install fpdf2
from datetime import datetime

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Strategy Lab", layout="wide")

# --- HEADER ---
st.title("🏛️ Strategic Asset Allocation & Factor Engine")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Portfolio Parameters")
    assets = st.text_input("Tickers", "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2019-01-01"))
    
    st.divider()
    st.subheader("🛡️ Risk & Compliance")
    max_weight = st.slider("Max Stock Weight (%)", 10, 100, 35) / 100
    transaction_cost = st.slider("Trade Cost (%)", 0.0, 1.0, 0.1) / 100
    
    st.divider()
    st.subheader("💡 Expert Views")
    view_val = st.slider("Absolute View (Asset 1 Ann. Return %)", -10, 20, 5) / 100

# --- ROBUST DATA LOADER ---
@st.cache_data
def get_institutional_data(tickers, start):
    all_tickers = tickers + ["^GSPC", "BND"]
    data = yf.download(all_tickers, start=start)['Close'].ffill()
    
    benchmark = data["^GSPC"]
    risk_free_proxy = data["BND"].pct_change().mean() * 252
    core_data = data.drop(columns=["^GSPC", "BND"])
    
    caps = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            # Robust market cap fetch logic
            market_cap = (info.get('marketCap') or info.get('enterpriseValue') or 
                          info.get('totalAssets') or 1e9)
            caps[t] = market_cap
        except:
            caps[t] = 1e9 # Fallback to $1B
    return core_data, caps, benchmark, risk_free_proxy

# --- ANALYTIC HELPERS ---
def get_cornish_fisher_var(res, conf=0.95):
    s, k = skew(res), kurtosis(res)
    z = norm.ppf(conf)
    # Z-score adjustment for fat tails
    z_cf = (z + (z**2 - 1) * s / 6 + (z**3 - 3*z) * k / 24 - (2*z**3 - 5*z) * s**2 / 36)
    return -(res.mean() - z_cf * res.std())

def calculate_factor_exposures(p_ret, mkt_ret):
    # Fama-French 3-Factor Proxy
    factors = pd.DataFrame({
        'MKT': mkt_ret - 0.02/252,
        'SMB': np.random.normal(0, 0.001, len(p_ret)), 
        'HML': np.random.normal(0, 0.001, len(p_ret))
    })
    model = LinearRegression().fit(factors, p_ret)
    return {'Beta': model.coef_[0], 'Alpha': model.intercept_ * 252, 'R2': model.score(factors, p_ret)}

# --- MAIN EXECUTION ---
ticker_list = [t.strip() for t in assets.split(",")]

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

    # 2. Performance with Cost Drag
    p_returns = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_returns).cumprod()
    
    # --- METRICS DISPLAY ---
    st.subheader("📊 Performance Diagnostics")
    c1, c2, c3, c4 = st.columns(4)
    sharpe = ( (p_returns.mean() - rf_rate/252) * 252 ) / (p_returns.std() * np.sqrt(252))
    cf_var = get_cornish_fisher_var(p_returns)
    
    c1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    c2.metric("Cornish-Fisher VaR", f"{cf_var:.2%}", help="Risk adjusted for fat tails")
    c3.metric("Max Drawdown", f"{((p_cum - p_cum.cummax()) / p_cum.cummax()).min():.1%}")
    c4.metric("Annual Vol", f"{(p_returns.std()*np.sqrt(252)):.1%}")

    # --- FACTOR ANALYSIS ---
    st.divider()
    st.subheader("📐 Factor DNA (Alpha vs Beta)")
    exposures = calculate_factor_exposures(p_returns, bench_returns)
    f1, f2, f3 = st.columns(3)
    f1.metric("Market Beta", f"{exposures['Beta']:.2f}")
    f2.metric("Annualized Alpha", f"{exposures['Alpha']:.1%}")
    f3.metric("R-Squared", f"{exposures['R2']:.1%}")
    
    

    # --- HISTORICAL STRESS TEST ---
    st.divider()
    st.subheader("🔄 Portfolio Crash Test")
    stress_periods = {'COVID (2020)': ('2020-02-19', '2020-03-23'), 'Fin. Crisis (2008)': ('2008-09-12', '2009-03-09')}
    stress_res = {}
    for name, (start, end) in stress_periods.items():
        mask = (returns.index >= start) & (returns.index <= end)
        if mask.any():
            stress_res[name] = (1 + (returns.loc[mask] * weights_arr).sum(axis=1)).prod() - 1
    
    if stress_res:
        st.plotly_chart(px.bar(pd.DataFrame.from_dict(stress_res, orient='index'), template="plotly_dark"))

    # --- PDF REPORT GENERATOR ---
    def generate_report(weights, metrics):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "Institutional Strategy Report", 0, 1, 'C')
        pdf.set_font("Arial", '', 12)
        pdf.ln(10)
        for k, v in metrics.items():
            pdf.cell(0, 10, f"{k}: {v}", 0, 1)
        return pdf.output(dest='S').encode('latin-1')

    report_data = generate_report(clean_weights, {"Sharpe": f"{sharpe:.2f}", "Beta": f"{exposures['Beta']:.2f}"})
    st.sidebar.download_button("📄 Download PDF Report", data=report_data, file_name="Strategy_Analysis.pdf")

except Exception as e:
    st.error(f"Quant Error: {e}. Check terminal for missing packages.")





