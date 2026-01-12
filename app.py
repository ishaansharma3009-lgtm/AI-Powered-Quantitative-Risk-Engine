import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
from scipy.stats import norm, skew, kurtosis
from sklearn.linear_model import LinearRegression
import plotly.express as px
from fpdf import FPDF

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Strategy Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Factor Engine")

# --- SIDEBAR & INPUTS ---
with st.sidebar:
    st.header("Portfolio Parameters")
    assets = st.text_input("Tickers", "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
    max_weight = st.slider("Max Stock Weight (%)", 10, 100, 35) / 100
    
# --- DATA & ANALYTICS HELPER FUNCTIONS ---
@st.cache_data
def get_data(tickers, start):
    all_t = tickers + ["^GSPC", "BND"]
    raw = yf.download(all_t, start=start)['Close'].ffill()
    
    # Calculate returns first
    all_rets = raw.pct_change().dropna()
    
    # Separate benchmark and risk-free
    bench = all_rets["^GSPC"]
    rf = raw["BND"].pct_change().mean() * 252
    
    # Return core data (only the tickers) and benchmark
    return all_rets[tickers], bench, rf

def get_cf_var(res, conf=0.95):
    s, k = skew(res), kurtosis(res)
    z = norm.ppf(conf)
    z_cf = (z + (z**2 - 1) * s/6 + (z**3 - 3*z) * k/24 - (2*z**3 - 5*z) * s**2/36)
    return -(res.mean() - z_cf * res.std())

def generate_pdf_report(metrics):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, "Institutional Strategy Report", 0, 1, 'C')
    pdf.set_font("Arial", '', 12)
    for k, v in metrics.items():
        pdf.cell(0, 10, f"{k}: {v}", 0, 1)
    
    # FIX: Handle bytearray output correctly for fpdf2
    out = pdf.output()
    return out if isinstance(out, (bytes, bytearray)) else out.encode('latin-1')

# --- MAIN EXECUTION ---
ticker_list = [t.strip() for t in assets.split(",") if t.strip()]

try:
    rets, bench_ret, rf_rate = get_data(ticker_list, start_date)

    # 1. Optimization (Mean-Variance)
    # Using sample covariance and mean historical returns
    S = rets.cov() * 252
    mu = rets.mean() * 252
    
    ef = EfficientFrontier(mu, S, weight_bounds=(0, max_weight))
    weights = ef.max_sharpe()
    clean_w = ef.clean_weights()
    weights_arr = np.array(list(clean_w.values()))

    # 2. Performance Metrics
    p_rets = (rets * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    sharpe = ((p_rets.mean() - rf_rate/252)*252) / (p_rets.std()*np.sqrt(252))
    cf_var = get_cf_var(p_rets)

    # --- UI: PERFORMANCE DASHBOARD ---
    st.subheader("📊 Performance Diagnostics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m2.metric("Cornish-Fisher VaR", f"{cf_var:.2%}")
    m3.metric("Max Drawdown", f"{((p_cum - p_cum.cummax())/p_cum.cummax()).min():.1%}")

    # --- UI: FACTOR ATTRIBUTION (MATCHING LENGTH FIX) ---
    st.divider()
    st.subheader("📐 Fama-French Factor Attribution")
    
    # Alignment check: Ensure benchmark and portfolio returns share exact indices
    aligned_df = pd.DataFrame({'MKT': bench_ret, 'Port': p_rets}).dropna()
    
    model = LinearRegression().fit(aligned_df[['MKT']], aligned_df['Port'])
    f1, f2, f3 = st.columns(3)
    f1.metric("Market Beta", f"{model.coef_[0]:.2f}")
    f2.metric("Annualized Alpha", f"{(model.intercept_*252):.1%}")
    f3.metric("R-Squared", f"{model.score(aligned_df[['MKT']], aligned_df['Port']):.1%}")

    # --- UI: STRESS TESTING ---
    st.divider()
    st.subheader("🔄 Historical Crash Test")
    stress = {'COVID (2020)': ('2020-02-19', '2020-03-23'), '2022 Bear Market': ('2022-01-01', '2022-12-31')}
    stress_results = {}
    for k, (s, e) in stress.items():
        subset = p_rets.loc[s:e]
        if not subset.empty:
            stress_results[k] = (1 + subset).prod() - 1
            
    if stress_results:
        st.plotly_chart(px.bar(pd.DataFrame.from_dict(stress_results, orient='index'), 
                               template="plotly_dark", title="Crisis Impact"))

    # --- PDF DOWNLOAD ---
    report = generate_pdf_report({"Sharpe": f"{sharpe:.2f}", "Beta": f"{model.coef_[0]:.2f}", "VaR": f"{cf_var:.2%}"})
    st.sidebar.download_button("📄 Export PDF Strategy", data=report, file_name="Strategy_Report.pdf")

except Exception as e:
    st.error(f"Strategy Engine Error: {e}")







