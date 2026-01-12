import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, black_litterman, objective_functions
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Strategy Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Risk Engine")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.header("Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Tickers", default_tickers)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🛡️ Compliance & Diversification")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 2.0, 0.5, help="Higher values force the model to spread wealth across more assets.")
    
    st.divider()
    st.subheader("💡 Black-Litterman View")
    ticker_list = [t.strip() for t in assets.split(",")]
    view_ticker = st.selectbox("Select Asset", ticker_list)
    view_return = st.slider(f"Expected Return for {view_ticker} (%)", -20, 40, 10) / 100

# --- DATA ENGINE ---
@st.cache_data
def get_market_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    data = yf.download(all_tickers, start=start)['Close']
    benchmark = data["^GSPC"].ffill()
    assets_data = data.drop(columns=["^GSPC"]).ffill()
    
    mcaps = {}
    for t in tickers:
        try:
            mcaps[t] = yf.Ticker(t).info.get('marketCap', 100000000000)
        except:
            mcaps[t] = 100000000000
    return assets_data, benchmark, mcaps

# --- ANALYTICS ENGINE ---
def calculate_risk_metrics(returns, weights):
    p_rets = (returns * weights).sum(axis=1)
    cum_rets = (1 + p_rets).cumprod()
    
    # Sharpe Ratio (Annualized)
    sharpe = (p_rets.mean() * 252) / (p_rets.std() * np.sqrt(252))
    
    # Value at Risk (VaR) 95% Confidence
    var_95 = np.percentile(p_rets, 5)
    
    # Peak-to-Trough (Max Drawdown)
    running_max = cum_rets.cummax()
    drawdown = (cum_rets - running_max) / running_max
    max_dd = drawdown.min()
    
    return p_rets, cum_rets, sharpe, var_95, max_dd

# --- EXECUTION ---
try:
    data, benchmark, market_caps = get_market_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    bench_returns = benchmark.pct_change().dropna()
    
    # 1. Black-Litterman Setup
    S = risk_models.sample_cov(data)
    prior_returns = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(S, pi=prior_returns, absolute_views={view_ticker: view_return}, omega="idzorek")
    bl_mu = bl.bl_returns()
    
    # 2. Optimization with Diversification Penalty
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    # Forces the optimizer to avoid concentrating in one stock
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty) 
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    weights_arr = np.array(list(clean_weights.values()))

    # 3. Calculate Performance Metrics
    p_rets, p_cum, sharpe, var_95, max_dd = calculate_risk_metrics(returns, weights_arr)
    b_cum = (1 + bench_returns).cumprod()

    # --- UI: PERFORMANCE DASHBOARD ---
    st.subheader("📊 Institutional Risk & Return Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Sharpe Ratio", f"{sharpe:.2f}", "Risk-Adjusted")
    m2.metric("Value at Risk (95%)", f"{var_95:.2%}", "Daily Threshold")
    m3.metric("Max Drawdown", f"{max_dd:.1%}", "Peak-to-Trough")
    m4.metric("Annualized Vol", f"{(p_rets.std() * np.sqrt(252)):.1%}")

    # --- UI: WEALTH CURVE ---
    st.subheader("📈 Performance vs. S&P 500 (Growth of $1)")
    fig_comp = go.Figure()
    fig_comp.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Optimized BL Strategy", line=dict(color='#00CC96', width=3)))
    fig_comp.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="S&P 500 Benchmark", line=dict(color='white', dash='dash')))
    fig_comp.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- UI: COMPLIANCE AUDIT ---
    st.divider()
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("🛡️ Compliance & Allocation")
        w_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(w_df[w_df['Weight']>0], values='Weight', names=w_df[w_df['Weight']>0].index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("🧩 Asset Correlation")
        fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

except Exception as e:
    st.error(f"Engine Error: {e}")








