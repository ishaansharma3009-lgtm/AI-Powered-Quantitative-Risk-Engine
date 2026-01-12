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
    """Adjust portfolio weights based on geopolitical risk exposure per sector."""
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
        if ticker not in weights:
            continue
        sector = ticker_sectors.get(ticker, 'Technology')
        total_risk = sum(sector_risk.get(sector, {}).get(event, 0.1) for event in events)
        reduction_factor = 1 - (total_risk * intensity * 0.15)
        adjustments[ticker] = max(0.01, weights[ticker] * reduction_factor)
    
    if not adjustments:
        return weights
    
    # Re-normalize to 100%
    total = sum(adjustments.values())
    if total > 0:
        return {k: v/total for k, v in adjustments.items()}
    return weights

# --- DATA ENGINE WITH RATE LIMIT PROTECTION ---
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_clean_data(tickers, start):
    """Safe data fetching with rate limit protection"""
    if not tickers:
        return pd.DataFrame(), pd.Series(), {}
    
    # Limit tickers to avoid rate limits
    tickers = tickers[:8]  # Max 8 tickers
    all_tickers = tickers + ["^GSPC"]
    
    try:
        # Try bulk download first
        raw_data = yf.download(all_tickers, start=start, progress=False)['Close']
        time.sleep(1)  # Prevent rate limiting
    except Exception as e:
        st.warning(f"Bulk download failed, trying individual tickers...")
        # Fallback: download individually
        raw_data = pd.DataFrame()
        for t in all_tickers:
            try:
                ticker_data = yf.download(t, start=start, progress=False, interval="1d")
                raw_data[t] = ticker_data['Close']
                time.sleep(0.5)  # Wait between requests
            except:
                continue
    
    if raw_data.empty:
        st.error("Could not fetch data. Please check ticker symbols and try again.")
        return pd.DataFrame(), pd.Series(), {}
    
    # Clean and separate
    clean_df = raw_data.ffill().dropna()
    if clean_df.empty:
        return pd.DataFrame(), pd.Series(), {}
    
    benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
    assets_data = clean_df.drop(columns=["^GSPC"]) if "^GSPC" in clean_df.columns else clean_df
    
    # Get market caps (with fallback values)
    fixed_caps = {
        'AAPL': 2.8e12, 'MSFT': 2.5e12, 'JPM': 0.5e12,
        'MC.PA': 0.07e12, 'ASML': 0.3e12, 'NESN.SW': 0.3e12,
        '2330.TW': 0.5e12, '7203.T': 0.03e12,
        'GOOGL': 1.8e12, 'AMZN': 1.6e12, 'TSLA': 0.6e12
    }
    
    mcaps = {}
    for t in tickers:
        if t in assets_data.columns:
            mcaps[t] = fixed_caps.get(t, 1e11)  # Use fixed caps to avoid API calls
    
    return assets_data, benchmark, mcaps

# --- MAIN EXECUTION ---
try:
    if not ticker_list:
        st.warning("Please enter at least one ticker symbol.")
        st.stop()
    
    # Load data
    with st.spinner("📡 Fetching market data..."):
        prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    
    if prices.empty:
        st.error("No data available for the selected tickers/period.")
        st.stop()
    
    # Ensure ticker_list matches available data
    available_tickers = [t for t in ticker_list if t in prices.columns]
    if not available_tickers:
        st.error("None of the entered tickers returned valid data.")
        st.stop()
    
    ticker_list = available_tickers  # Update to only available tickers
    returns = prices[ticker_list].pct_change().dropna()
    
    if not bench_prices.empty:
        bench_returns = bench_prices.pct_change().dropna()
        # Align dates
        common_idx = returns.index.intersection(bench_returns.index)
        returns = returns.loc[common_idx]
        bench_returns = bench_returns.loc[common_idx]
    else:
        bench_returns = pd.Series()
    
    if returns.empty:
        st.error("Insufficient data for analysis. Try a longer time period.")
        st.stop()
    
    # 1. Black-Litterman Optimization
    with st.spinner("⚙️ Optimizing portfolio..."):
        S = risk_models.sample_cov(prices[ticker_list])
        
        # Create market caps dict with only available tickers
        available_caps = {t: market_caps.get(t, 1e11) for t in ticker_list}
        prior_rets = black_litterman.market_implied_prior_returns(available_caps, 2.5, S)
        
        # Ensure view ticker is in our list
        if view_ticker not in ticker_list:
            view_ticker = ticker_list[0]
        
        bl = black_litterman.BlackLittermanModel(
            S, 
            pi=prior_rets, 
            absolute_views={view_ticker: view_return}, 
            omega="idzorek", 
            view_confidences=[view_conf]
        )
        bl_mu = bl.bl_returns()
    
    # 2. Base Optimization
    ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
    ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
    optimized_weights = ef.max_sharpe()
    optimized_weights = ef.clean_weights()  # Clean small weights
    
    # 3. Geopolitical Adjustment
    if geo_events and geo_intensity > 0.5:
        final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)
    else:
        final_weights = optimized_weights
    
    # Convert to array for calculations
    weights_arr = np.array([final_weights.get(t, 0) for t in ticker_list])
    
    # 4. Performance Calculations
    p_rets = (returns * weights_arr).sum(axis=1)
    p_cum = (1 + p_rets).cumprod()
    
    if not bench_returns.empty:
        b_cum = (1 + bench_returns).cumprod()
    
    # Analytics
    ann_ret = p_rets.mean() * 252
    ann_vol = p_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()

    # --- UI: DASHBOARD ---
    st.subheader("📊 Performance & Geopolitical Analysis")
    
    # Metrics in columns
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Sharpe Ratio", f"{sharpe:.2f}")
    col2.metric("Annual Return", f"{ann_ret:.1%}")
    col3.metric("Annual Volatility", f"{ann_vol:.1%}")
    col4.metric("Max Drawdown", f"{max_dd:.1%}")

    # --- PERFORMANCE CHART ---
    st.divider()
    st.subheader("📈 Cumulative Growth")
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=p_cum.index, 
        y=p_cum, 
        name="Strategy", 
        line=dict(color='#00CC96', width=2.5),
        fill='tozeroy',
        fillcolor='rgba(0, 204, 150, 0.1)'
    ))
    
    if not bench_returns.empty:
        fig_perf.add_trace(go.Scatter(
            x=b_cum.index, 
            y=b_cum, 
            name="S&P 500", 
            line=dict(color='#636EFA', dash='dash')
        ))
    
    fig_perf.update_layout(
        template="plotly_dark", 
        height=400,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        hovermode='x unified'
    )
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- CORRELATION HEATMAP SECTION ---
    st.divider()
    st.subheader("🔥 Asset Correlation Matrix")
    
    # Calculate correlation matrix
    corr_matrix = returns.corr()
    
    # Create interactive heatmap
    fig_corr = px.imshow(
        corr_matrix,
        text_auto='.2f',  # Show correlation values with 2 decimals
        aspect="auto",     # Auto-adjust aspect ratio
        color_continuous_scale='RdBu_r',  # Red-Blue reversed scale
        title="Asset Correlation Heatmap",
        labels=dict(color="Correlation"),
        zmin=-1,  # Fix scale from -1 to 1
        zmax=1
    )
    
    # Add custom hover text
    fig_corr.update_traces(
        hovertemplate="<b>%{y} vs %{x}</b><br>Correlation: %{z:.3f}<extra></extra>"
    )
    
    # Update layout
    fig_corr.update_layout(
        height=500,
        xaxis_title="Assets",
        yaxis_title="Assets",
        coloraxis_colorbar=dict(
            title="Correlation",
            titleside="right",
            tickvals=[-1, -0.5, 0, 0.5, 1],
            ticktext=["-1.0 (Perfect Negative)", "-0.5", "0.0 (Uncorrelated)", "0.5", "1.0 (Perfect Positive)"]
        )
    )
    
    # Display heatmap
    st.plotly_chart(fig_corr, use_container_width=True)
    
    # Correlation insights
    col_insight1, col_insight2, col_insight3 = st.columns(3)
    
    with col_insight1:
        # Find strongest positive correlation
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        corr_values = corr_matrix.where(mask).stack()
        if not corr_values.empty:
            max_corr_pair = corr_values.idxmax()
            max_corr_value = corr_values.max()
            st.metric("Strongest Correlation", 
                     f"{max_corr_value:.2f}",
                     f"{max_corr_pair[0]} & {max_corr_pair[1]}")
    
    with col_insight2:
        # Find strongest negative correlation (best for diversification)
        if not corr_values.empty:
            min_corr_pair = corr_values.idxmin()
            min_corr_value = corr_values.min()
            st.metric("Best Diversification", 
                     f"{min_corr_value:.2f}",
                     f"{min_corr_pair[0]} & {min_corr_pair[1]}")
    
    with col_insight3:
        # Average correlation
        avg_corr = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
        st.metric("Average Portfolio Correlation", f"{avg_corr:.2f}")
    
    # Interpretation guide
    with st.expander("📖 How to Read Correlation Matrix"):
        st.markdown("""
        **Correlation measures how assets move together:**
        
        | Value Range | Interpretation | Diversification Benefit |
        |-------------|----------------|------------------------|
        | **1.0** | Perfect positive correlation | ❌ None (move exactly together) |
        | **0.7 to 0.9** | Strong positive correlation | ⚠️ Limited |
        | **0.3 to 0.7** | Moderate correlation | ✅ Some |
        | **0.0 to 0.3** | Weak correlation | ✅ Good |
        | **Negative** | Inverse relationship | ✅ Excellent |
        
        **Ideal portfolio**: Contains assets with **low or negative correlations** to reduce overall risk.
        """)

    # --- ALLOCATION & ANALYSIS ---
    st.divider()
    st.subheader("📦 Portfolio Construction")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🍕 Portfolio Allocation")
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        w_df = w_df[w_df['Weight'] > 0.001]  # Only show significant weights
        
        if not w_df.empty:
            # Sort by weight descending
            w_df = w_df.sort_values('Weight', ascending=False)
            
            fig_pie = px.pie(
                w_df, 
                values='Weight', 
                names=w_df.index, 
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Set3,
                category_orders={"index": w_df.index.tolist()}
            )
            fig_pie.update_layout(
                showlegend=True, 
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5)
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # Show weight table
            st.dataframe(
                w_df.style.format({'Weight': '{:.1%}'}),
                use_container_width=True
            )
        else:
            st.info("No significant allocations found.")
    
    with col_right:
        st.markdown("#### 🛡️ Geopolitical Impact")
        
        if geo_events and geo_intensity > 0.5:
            # Calculate changes
            changes = {}
            for t in ticker_list:
                orig = optimized_weights.get(t, 0)
                new = final_weights.get(t, 0)
                change = new - orig
                if abs(change) > 0.001:  # Only show meaningful changes
                    changes[t] = change * 100  # Convert to percentage points
            
            if changes:
                changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Change (%)'])
                changes_df = changes_df.sort_values('Change (%)')
                
                fig_changes = px.bar(
                    changes_df,
                    orientation='h',
                    title="Geopolitical Weight Adjustments",
                    color=changes_df['Change (%)'] > 0,
                    color_discrete_map={True: '#FF4B4B', False: '#00CC96'},
                    labels={'index': 'Ticker', 'Change (%)': 'Change (percentage points)'}
                )
                fig_changes.update_layout(
                    showlegend=False, 
                    height=400,
                    xaxis_title="Change (percentage points)",
                    yaxis_title="Ticker"
                )
                st.plotly_chart(fig_changes, use_container_width=True)
                
                # Summary stats
                total_reduction = sum(v for v in changes.values() if v < 0)
                total_increase = sum(v for v in changes.values() if v > 0)
                
                st.metric("Total Risk Reduction", f"{abs(total_reduction):.1f} pp")
                st.metric("Risk-Averse Increase", f"{total_increase:.1f} pp")
            else:
                st.info("Geopolitical events had minimal impact on current allocation.")
        else:
            st.info("Enable geopolitical risk overlay in sidebar to see adjustments.")

    # --- EXPORT ---
    st.divider()
    st.subheader("📥 Export Results")
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Export weights
        export_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        csv = export_df.to_csv().encode('utf-8')
        st.download_button(
            label="📊 Portfolio Weights",
            data=csv,
            file_name='portfolio_weights.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp2:
        # Export performance
        perf_df = pd.DataFrame({
            'Date': p_cum.index,
            'Strategy': p_cum.values,
            'Returns': p_rets.values
        })
        if not bench_returns.empty:
            perf_df['S&P_500'] = b_cum.values
        
        csv_perf = perf_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📈 Performance Data",
            data=csv_perf,
            file_name='performance_data.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp3:
        # Export correlation matrix
        corr_export = corr_matrix.reset_index()
        corr_csv = corr_export.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="🔥 Correlation Matrix",
            data=corr_csv,
            file_name='correlation_matrix.csv',
            mime='text/csv',
            use_container_width=True
        )

except Exception as e:
    st.error(f"🚨 Engine Error: {str(e)}")
    
    # Helpful error messages for common issues
    if "rate limit" in str(e).lower() or "too many" in str(e).lower():
        st.info("""
        **Rate Limit Issue Detected**
        
        Yahoo Finance has rate limits. Try:
        1. Wait 1-2 minutes and refresh
        2. Use fewer tickers (4-6 instead of 8+)
        3. Use the default tickers provided
        """)
    
    elif "ticker" in str(e).lower() or "symbol" in str(e).lower():
        st.info("""
        **Ticker Issue Detected**
        
        Please check:
        1. Ticker symbols are correct (e.g., AAPL, not APPL)
        2. International tickers include exchange (e.g., MC.PA for Paris)
        3. Separate tickers with commas only
        """)




