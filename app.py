import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, expected_returns, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="Strategic Asset Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Geopolitical Risk Engine")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.header("Portfolio Settings")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW, 2330.TW, 7203.T"
    assets = st.text_input("Tickers", default_tickers)
    start_date = st.date_input("Start Date", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🌐 Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Geopolitical Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity Multiplier", 0.5, 3.0, 1.0, 0.1)

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

# --- DATA ENGINE (WITH SYNCHRONIZATION) ---
@st.cache_data
def get_clean_synchronized_data(tickers, start):
    all_tickers = tickers + ["^GSPC"]
    # Download and drop any rows where ANY ticker has missing data (Syncs holidays)
    data = yf.download(all_tickers, start=start)['Close'].ffill().dropna()
    benchmark = data["^GSPC"]
    assets_data = data.drop(columns=["^GSPC"])
    
    mcaps = {}
    for t in tickers:
        try:
            mcaps[t] = yf.Ticker(t).info.get('marketCap', 1e11)
        except:
            mcaps[t] = 1e11
    return assets_data, benchmark, mcaps

# --- GEOPOLITICAL LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7},
        'Financials': {'Currency Volatility': 0.6, 'Trade Policy Changes': 0.4},
        'Automotive': {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8},
        'Healthcare': {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3}
    }
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Automotive', 'ASML': 'Semiconductors', 
        'NESN.SW': 'Healthcare', '2330.TW': 'Semiconductors', '7203.T': 'Automotive'
    }
    adjustments = {}
    for ticker in tickers:
        sector = ticker_sectors.get(ticker, 'Technology')
        risk = sum(sector_risk.get(sector, {}).get(e, 0.1) for e in events)
        adj_factor = 1 - (risk * intensity * 0.2)
        adjustments[ticker] = max(0.01, weights.get(ticker, 0) * adj_factor)
    
    total = sum(adjustments.values())
    return {k: v/total for k, v in adjustments.items()}

# --- MAIN ENGINE ---
try:
    assets_df, bench_df, market_caps = get_clean_synchronized_data(ticker_list, start_date)
    returns = assets_df.pct_change().dropna()
    
    # 1. Black-Litterman (Fixed Confidence Error)
    S = risk_models.sample_cov(assets_df)
    prior_returns = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(
        S, pi=prior_returns, absolute_views={view_ticker: view_return}, 
        omega="idzorek", 
        view_confidences=[view_conf] # Fixed: Must be a list for Idzorek
    )
    bl_mu = bl.bl_returns()

    # 2. Optimization with L2 Diversification
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    base_weights = ef.max_sharpe()
    clean_base = ef.clean_weights()
    
    # 3. Apply Geopolitical Overlay
    final_weights = apply_geopolitical_overlay(clean_base, ticker_list, geo_events, geo_intensity)
    weights_arr = np.array([final_weights[t] for t in ticker_list])

    # 4. Annualized Performance Suite
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    
    # Proper order of Annualization
    ann_return = p_rets.mean() * 252
    ann_vol = p_rets.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol
    var_95 = np.percentile(p_rets, 5)
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()

    # --- UI: ANALYTICS ---
    st.subheader("📈 Institutional Risk & Performance")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Annualized Return", f"{ann_return:.1%}")
    m2.metric("Annualized Vol", f"{ann_vol:.1%}")
    m3.metric("Sharpe Ratio", f"{sharpe:.2f}")
    m4.metric("Max Drawdown", f"{max_dd:.1%}")

    # --- UI: GEOPOLITICAL IMPACT ---
    st.divider()
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("🛡️ Weight Shift Analysis")
        comp_df = pd.DataFrame({"Optimized": clean_base.values(), "Geo-Adjusted": final_weights.values()}, index=ticker_list)
        st.bar_chart(comp_df)
    with col_g2:
        st.subheader("🧩 Correlation Heatmap")
        fig_corr = px.imshow(returns.corr(), text_auto=".2f", color_continuous_scale='RdBu_r', template="plotly_dark")
        st.plotly_chart(fig_corr, use_container_width=True)

    # --- UI: EXPORT FIX ---
    csv_data = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight']).to_csv().encode('utf-8')
    st.sidebar.download_button("📥 Download Trade Report", data=csv_data, file_name='portfolio_weights.csv')

except Exception as e:
    st.error(f"Engine Error: {str(e)}")







