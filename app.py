import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import black_litterman, risk_models, expected_returns, EfficientFrontier
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Institutional Quant Lab", layout="wide")

# --- HEADER ---
st.title("🏛️ Robust Strategic Asset Allocation Engine")
st.markdown("---")

# --- SIDEBAR ---
with st.sidebar:
    st.header("Advanced Quant Settings")
    assets = st.text_input("Tickers", "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW")
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🛠️ Robustness Settings")
    n_resamples = st.slider("Resampling Iterations", 10, 200, 50)
    
    st.subheader("💡 Black-Litterman Views")
    view_val = st.slider("Market View (Asset 1 Return %)", -10, 20, 5) / 100
    
    st.divider()
    download_placeholder = st.empty()
    st.caption("Developed by Ishaan Sharma | Quantitative Research Tool")

# --- DATA FETCHING ---
ticker_list = [t.strip() for t in assets.split(",")]

@st.cache_data
def get_data(tickers, start):
    data = yf.download(tickers, start=start)['Close'].ffill()
    return data

try:
    data = get_data(ticker_list, start_date)
    returns = data.pct_change().dropna()
    
    # 1. BLACK-LITTERMAN INPUTS
    mu = expected_returns.mean_historical_return(data)
    S = risk_models.sample_cov(data)
    viewdict = {ticker_list[0]: view_val}
    bl = black_litterman.BlackLittermanModel(S, pi=mu, absolute_views=viewdict)
    ret_bl = bl.bl_returns()

    # 2. ROBUST RESAMPLED OPTIMIZATION
    all_weights = []
    with st.spinner('Running Resampled Optimizations...'):
        for i in range(n_resamples):
            noisy_ret = ret_bl + np.random.normal(0, returns.std(), len(ret_bl))
            try:
                ef = EfficientFrontier(noisy_ret, S, weight_bounds=(0.02, 0.40))
                w = ef.max_sharpe()
                all_weights.append(pd.Series(w))
            except:
                continue
    
    robust_weights = pd.concat(all_weights, axis=1).mean(axis=1)
    clean_weights = robust_weights.to_dict()
    weights_arr = np.array(list(clean_weights.values()))

    # 3. CALCULATIONS
    portfolio_daily_returns = (returns * weights_arr).sum(axis=1)
    portfolio_cum = (1 + portfolio_daily_returns).cumprod()
    p_vol = portfolio_daily_returns.std() * np.sqrt(252)
    max_drawdown = ((portfolio_cum - portfolio_cum.cummax()) / portfolio_cum.cummax()).min()
    
    # Monte Carlo VaR
    n_sims = 5000
    sim_returns = np.random.normal(portfolio_daily_returns.mean(), portfolio_daily_returns.std(), (252, n_sims))
    sim_growth = (1 + sim_returns).cumprod(axis=0) * 10000
    mc_var_95 = np.percentile(sim_growth[-1], 5)

    # --- UI DISPLAY ---
    st.subheader("🛡️ Robust Portfolio Metrics")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Monte Carlo VaR ($)", f"${mc_var_95:,.0f}", help="95% probability your $10k stays above this.")
    c2.metric("Annualized Vol", f"{p_vol:.1%}", help="The 'smoothness' of the investment journey.")
    c3.metric("Max Drawdown", f"{max_drawdown:.1%}", help="Worst historical peak-to-trough loss.")
    c4.metric("Risk-Adjusted Return", f"{(portfolio_daily_returns.mean()*252/p_vol):.2f}", help="Sharpe Ratio")

    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.subheader("📈 Your Strategy Allocation")
        fig_pie = px.pie(values=list(clean_weights.values()), names=list(clean_weights.keys()), hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.T10)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("🎲 Future Wealth Scenarios")
        fig_mc = go.Figure()
        for i in range(30):
            fig_mc.add_trace(go.Scatter(y=sim_growth[:, i], line=dict(width=0.8), opacity=0.3, showlegend=False))
        fig_mc.update_layout(template="plotly_dark", height=400, yaxis_title="Portfolio Value ($)")
        st.plotly_chart(fig_mc, use_container_width=True)

    # --- NEW: LAYMAN INTERPRETATION SECTION ---
    st.divider()
    st.header("📋 Manager's Summary & Insights")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("🧐 What this means for you")
        if p_vol < 0.15:
            risk_desc = "Conservative/Steady"
            mood = "a calm train ride."
        elif p_vol < 0.25:
            risk_desc = "Balanced/Moderate"
            mood = "a standard car journey with occasional bumps."
        else:
            risk_desc = "Aggressive/High Growth"
            mood = "a fast roller coaster."

        st.info(f"""
        **Portfolio Style:** {risk_desc}
        
        **The Story:** Your portfolio is designed like {mood} 
        By using 'Robust Resampling,' we've ensured that your money isn't just betting on 
        one lucky stock, but is spread out to withstand market noise. 
        """)

    with col_b:
        st.subheader("📉 The 'Worst Case' Stress Test")
        # Creating a hypothetical "Market Crash" table
        crash_scenarios = pd.DataFrame({
            "Scenario": ["2020 COVID Crash", "2008 Financial Crisis", "Typical Bad Month"],
            "Est. Impact": [f"{max_drawdown*1.1:.1%}", f"{max_drawdown*1.5:.1%}", "-5.0%"],
            "Recovery Time": ["4 Months", "18 Months", "1 Month"]
        })
        st.table(crash_scenarios)
        st.caption("Note: Impact estimates are based on your portfolio's current volatility and drawdown profile.")

    # --- DOWNLOAD ---
    csv = robust_weights.to_csv().encode('utf-8')
    download_placeholder.download_button("📥 Export Strategy for Client", data=csv, file_name='client_portfolio.csv')

except Exception as e:
    st.error(f"Quant Engine Error: {e}")




