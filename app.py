import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time

# --- PAGE SETUP ---
st.set_page_config(page_title="Institutional Strategy Lab", layout="wide")
st.title("🏛️ Institutional Strategy & Geopolitical Risk Engine")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.header("Strategic Parameters")
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip() for t in assets.split(",") if t.strip()]
    start_date = st.date_input("Analysis Start", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.subheader("🌐 Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.subheader("💡 Black-Litterman View")
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider(f"Return for {view_ticker} (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.subheader("🛡️ Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 2.0, 0.5)

# --- GEOPOLITICAL LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    if not events or intensity <= 0.5:
        return weights
    
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
        if ticker not in weights: continue
        sector = ticker_sectors.get(ticker, 'Technology')
        total_risk = sum(sector_risk.get(sector, {}).get(event, 0.1) for event in events)
        reduction_factor = 1 - (total_risk * intensity * 0.15)
        adjustments[ticker] = max(0.01, weights[ticker] * reduction_factor)
    
    total = sum(adjustments.values())
    return {k: v/total for k, v in adjustments.items()} if total > 0 else weights

# --- DATA ENGINE ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start):
    if not tickers: return pd.DataFrame(), pd.Series(), {}
    tickers = tickers[:8]
    all_tickers = tickers + ["^GSPC"]
    try:
        raw_data = yf.download(all_tickers, start=start, progress=False)['Close']
        time.sleep(1)
    except:
        return pd.DataFrame(), pd.Series(), {}
    
    clean_df = raw_data.ffill().dropna()
    if clean_df.empty: return pd.DataFrame(), pd.Series(), {}
    
    benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
    assets_data = clean_df.drop(columns=["^GSPC"]) if "^GSPC" in clean_df.columns else clean_df
    
    fixed_caps = {'AAPL': 2.8e12, 'MSFT': 2.5e12, 'JPM': 0.5e12, 'MC.PA': 0.07e12, 'ASML': 0.3e12, 'NESN.SW': 0.3e12}
    mcaps = {t: fixed_caps.get(t, 1e11) for t in tickers if t in assets_data.columns}
    
    return assets_data, benchmark, mcaps

# --- EXECUTION ---
try:
    with st.spinner("📡 Fetching market data..."):
        prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    
    if prices.empty:
        st.error("No data available.")
        st.stop()
    
    available_tickers = [t for t in ticker_list if t in prices.columns]
    returns = prices[available_tickers].pct_change().dropna()
    
    # 1. Black-Litterman
    S = risk_models.sample_cov(prices[available_tickers])
    prior_rets = black_litterman.market_implied_prior_returns(market_caps, 2.5, S)
    bl = black_litterman.BlackLittermanModel(S, pi=prior_rets, absolute_views={view_ticker: view_return}, omega="idzorek", view_confidences=[view_conf])
    bl_mu = bl.bl_returns()

    # 2. Optimization
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    optimized_weights = ef.clean_weights()
    
    # 3. Geo Adjustment
    final_weights = apply_geopolitical_overlay(optimized_weights, available_tickers, geo_events, geo_intensity)
    weights_arr = np.array([final_weights.get(t, 0) for t in available_tickers])
    
    # 4. Analytics
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    
    # --- UI: DASHBOARD ---
    st.subheader("📊 Performance & Geopolitical Analysis")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sharpe Ratio", f"{(p_rets.mean()*252)/(p_rets.std()*np.sqrt(252)):.2f}")
    c2.metric("Annual Return", f"{p_rets.mean()*252:.1%}")
    c3.metric("Annual Volatility", f"{p_rets.std()*np.sqrt(252):.1%}")
    c4.metric("Max Drawdown", f"{((p_cum - p_cum.cummax()) / p_cum.cummax()).min():.1%}")

    st.divider()
    st.subheader("📈 Cumulative Growth")
    fig_perf = px.line(p_cum, labels={'value': 'Growth of $1', 'Date': ''}, template="plotly_dark")
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- NEW VISUALIZATION SECTION ---
    st.divider()
    st.subheader("🔍 Deep Risk & Allocation Analysis")
    v_col1, v_col2, v_col3 = st.columns(3)

    with v_col1:
        st.markdown("**🍕 Final Allocation**")
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight']).reset_index()
        fig_pie = px.pie(w_df, values='Weight', names='index', hole=0.4, color_discrete_sequence=px.colors.qualitative.Set3)
        fig_pie.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with v_col2:
        st.markdown("**🛡️ Geopolitical Impact (pp Change)**")
        changes = {t: (final_weights.get(t,0) - optimized_weights.get(t,0))*100 for t in available_tickers}
        changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Change']).sort_values('Change')
        fig_bar = px.bar(changes_df, orientation='h', color=changes_df['Change'] > 0, 
                         color_discrete_map={True: '#FF4B4B', False: '#00CC96'}, template="plotly_dark")
        fig_bar.update_layout(showlegend=False, height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_bar, use_container_width=True)

    with v_col3:
        st.markdown("**🧩 Correlation Heatmap**")
        corr_matrix = returns.corr()
        fig_heat = px.imshow(corr_matrix, text_auto=".2f", aspect="auto",
                             color_continuous_scale='RdBu_r', range_color=[-1, 1],
                             template="plotly_dark")
        fig_heat.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20))
        st.plotly_chart(fig_heat, use_container_width=True)

except Exception as e:
    st.error(f"🚨 Engine Error: {str(e)}")




