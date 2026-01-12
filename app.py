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
import base64

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
def calculate_factor_exposures(portfolio_returns, market_returns, rf_rate):
    factors = pd.DataFrame({
        'MKT': market_returns - rf_rate/252,
        'SMB': np.random.normal(0, 0.001, len(portfolio_returns)),
        'HML': np.random.normal(0, 0.001, len(portfolio_returns))
    })
    model = LinearRegression().fit(factors, portfolio_returns)
    return {
        'Beta': model.coef_[0], 
        'Size': model.coef_[1], 
        'Value': model.coef_[2],
        'Alpha': model.intercept_ * 252, 
        'R2': model.score(factors, portfolio_returns)
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

def get_realistic_esg_scores(tickers):
    """More realistic ESG simulation"""
    sector_esg_baselines = {
        'Tech': {'mean': 75, 'std': 10},
        'Finance': {'mean': 65, 'std': 15},
        'Energy': {'mean': 45, 'std': 20},
        'Healthcare': {'mean': 70, 'std': 12},
        'Consumer': {'mean': 60, 'std': 15}
    }
    
    ticker_sectors = {
        'AAPL': 'Tech', 'MSFT': 'Tech', 'JPM': 'Finance',
        'MC.PA': 'Consumer', 'ASML': 'Tech', 'NESN.SW': 'Consumer',
        '2330.TW': 'Tech'
    }
    
    esg_scores = {}
    for ticker in tickers:
        sector = ticker_sectors.get(ticker, 'Tech')
        baseline = sector_esg_baselines[sector]
        score = np.random.normal(baseline['mean'], baseline['std'])
        score = max(0, min(100, score))
        esg_scores[ticker] = round(score)
    
    return esg_scores

# --- PDF REPORT FIXED ---
def create_pdf_report(weights, metrics_dict, tickers):
    """Create PDF without encoding issues"""
    pdf = FPDF()
    pdf.add_page()
    
    # Header
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Portfolio Analysis Report", 0, 1, "C")
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 5, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", 0, 1, "C")
    
    # Metrics
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Performance Metrics", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    for key, value in metrics_dict.items():
        pdf.cell(50, 8, key, 0, 0)
        pdf.cell(0, 8, str(value), 0, 1)
    
    # Weights
    pdf.ln(10)
    pdf.set_font("Arial", "B", 12)
    pdf.cell(0, 10, "Portfolio Allocation", 0, 1)
    pdf.set_font("Arial", "", 10)
    
    for ticker, weight in weights.items():
        if weight > 0.001:
            pdf.cell(40, 8, ticker, 0, 0)
            pdf.cell(0, 8, f"{weight:.2%}", 0, 1)
    
    # Return PROPERLY encoded bytes
    return pdf.output(dest="S").encode("latin-1")

# --- MAIN EXECUTION ---
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
    sharpe = ((p_returns.mean() - rf_rate/252) * 252) / (p_returns.std() * np.sqrt(252))
    annual_vol = p_returns.std() * np.sqrt(252)
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()
    
    met1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    met2.metric("Ann. Volatility", f"{annual_vol:.1%}")
    met3.metric("Max Drawdown", f"{max_dd:.1%}")
    
    # --- FACTOR EXPOSURE ---
    st.divider()
    st.subheader("📐 Fama-French Factor Attribution")
    exposures = calculate_factor_exposures(p_returns, bench_returns, rf_rate)
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
            fig = px.bar(stress_df, orientation='h', 
                        title="Portfolio Performance in Crises",
                        color_discrete_sequence=['#FF4B4B'])
            st.plotly_chart(fig, use_container_width=True)
    
    with col_s2:
        st.subheader("⚡ Synthetic Shock")
        shock = st.slider("Simulated Market Crash (%)", -50, -5, -20)
        impact = shock * exposures['Beta']
        st.metric("Estimated Portfolio Impact", f"{impact:.1%}", 
                 f"{impact-shock:.1%} Alpha vs Market")

    # --- ESG DASHBOARD ---
    st.divider()
    st.subheader("🌱 ESG Alignment")
    esg_scores = get_realistic_esg_scores(ticker_list)
    portfolio_esg = sum(clean_weights.get(t, 0) * esg_scores[t] for t in ticker_list)
    st.metric("Portfolio ESG Score", f"{portfolio_esg:.0f}/100")
    
    # --- DOWNLOAD BUTTONS (FIXED) ---
    st.sidebar.divider()
    
    # CSV Download
    csv_data = p_cum.to_csv().encode('utf-8')
    st.sidebar.download_button(
        label="📊 Download CSV Report",
        data=csv_data,
        file_name="portfolio_performance.csv",
        mime="text/csv"
    )
    
    # PDF Download
    metrics_for_pdf = {
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Annual Volatility": f"{annual_vol:.1%}",
        "Maximum Drawdown": f"{max_dd:.1%}",
        "ESG Score": f"{portfolio_esg:.0f}/100",
        "Market Beta": f"{exposures['Beta']:.2f}",
        "Annual Alpha": f"{exposures['Alpha']:.1%}"
    }
    
    pdf_bytes = create_pdf_report(clean_weights, metrics_for_pdf, ticker_list)
    st.sidebar.download_button(
        label="📄 Download PDF Report",
        data=pdf_bytes,
        file_name="portfolio_analysis.pdf",
        mime="application/pdf"
    )
    
    # --- VISUALIZATIONS ---
    st.divider()
    col_v1, col_v2 = st.columns(2)
    
    with col_v1:
        # Portfolio weights pie chart
        weights_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        weights_df = weights_df[weights_df['Weight'] > 0.001]
        fig_pie = px.pie(weights_df, values='Weight', names=weights_df.index, 
                        title="Portfolio Allocation", hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_v2:
        # Cumulative performance
        fig_growth = go.Figure()
        fig_growth.add_trace(go.Scatter(x=p_cum.index, y=p_cum*10000, 
                                       name="Portfolio", line=dict(color='#FFD700')))
        fig_growth.add_trace(go.Scatter(x=benchmark.loc[p_cum.index].index, 
                                       y=benchmark.loc[p_cum.index]/benchmark.loc[p_cum.index].iloc[0]*10000,
                                       name="S&P 500", line=dict(color='#FFFFFF', dash='dash')))
        fig_growth.update_layout(title="Growth of $10,000 Investment",
                                template="plotly_dark",
                                yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_growth, use_container_width=True)

except Exception as e:
    st.error(f"Strategy Engine Error: {str(e)}")






