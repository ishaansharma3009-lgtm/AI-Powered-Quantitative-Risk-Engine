import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Asset Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Risk Engine")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Portfolio Settings")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Enter Tickers", default_tickers)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("💡 Black-Litterman View")
    ticker_list = [t.strip() for t in assets.split(",")]
    view_ticker = st.selectbox("Select Asset for View", ticker_list)
    view_return = st.slider(f"Expected Return for {view_ticker} (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.subheader("🛡️ Compliance & Diversification")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 2.0, 0.5)

# --- DATA ENGINE ---
@st.cache_data
def get_clean_market_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    # Ensure data is clean and synchronized
    data = yf.download(all_tickers, start=start)['Close'].ffill().dropna()
    benchmark = data["^GSPC"]
    assets_data = data.drop(columns=["^GSPC"])
    
    # Get Market Caps
    mcaps = {}
    for t in tickers:
        try:
            mcaps[t] = yf.Ticker(t).info.get('marketCap', 1e11)
        except:
            mcaps[t] = 1e11
    return assets_data, benchmark, mcaps

# --- MAIN ENGINE ---
try:
    assets_df, bench_df, market_caps = get_clean_market_data(ticker_list, start_date)
    returns = assets_df.pct_change().dropna()
    bench_returns = bench_df.pct_change().dropna()
    
    # 1. ALIGNMENT FIX: Ensure lengths match exactly
    common_index = returns.index.intersection(bench_returns.index)
    returns = returns.loc[common_index]
    bench_returns = bench_returns.loc[common_index]

    # 2. BLACK-LITTERMAN (Fixed Idzorek Confidence)
    S = risk_models.sample_cov(assets_df)
    prior_returns = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    
    bl = black_litterman.BlackLittermanModel(
        S, 
        pi=prior_returns, 
        absolute_views={view_ticker: view_return}, 
        omega="idzorek",
        view_confidences=[view_conf]  # FIX: Vector of confidence for views
    )
    bl_mu = bl.bl_returns()

    # 3. OPTIMIZATION WITH CONSTRAINTS
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    weights = ef.max_sharpe()
    clean_weights = ef.clean_weights()
    weights_arr = np.array(list(clean_weights.values()))

    # 4. PERFORMANCE METRICS
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    b_cum = (1 + bench_returns).cumprod()
    
    # Risk Stats
    sharpe = (p_rets.mean() * 252) / (p_rets.std() * np.sqrt(252))
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()
    var_95 = np.percentile(p_rets, 5)

    # --- DASHBOARD ---
    st.subheader("📊 Performance vs. Benchmark")
    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col_m2.metric("Max Drawdown", f"{max_dd:.1%}")
    col_m3.metric("Value at Risk (95%)", f"{var_95:.2%}")

    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(x=p_cum.index, y=p_cum, name="Portfolio Strategy", line=dict(color='#00CC96', width=2)))
    fig_perf.add_trace(go.Scatter(x=b_cum.index, y=b_cum, name="S&P 500", line=dict(color='white', dash='dash')))
    fig_perf.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- ALLOCATION & CORRELATION ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("🛡️ Strategic Weights")
        w_df = pd.DataFrame.from_dict(clean_weights, orient='index', columns=['Weight'])
        fig_pie = px.pie(w_df[w_df['Weight']>0], values='Weight', names=w_df[w_df['Weight']>0].index, hole=0.4)
        st.plotly_chart(fig_pie, use_container_width=True)
    with c2:
        st.subheader("🧩 Asset Correlation")
        fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)
        
    # --- EXPORT FIX ---
    st.sidebar.divider()
    csv_data = w_df.to_csv().encode('utf-8') # FIX: Encoding for bytearray
    st.sidebar.download_button("📥 Export Analysis", data=csv_data, file_name='portfolio_report.csv')

except Exception as e:
    st.error(f"Engine Error: {str(e)}")








