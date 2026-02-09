import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from pypfopt import risk_models, EfficientFrontier, black_litterman, objective_functions
import plotly.express as px
import plotly.graph_objects as go
import time
import warnings
warnings.filterwarnings('ignore')

# --- PAGE SETUP ---
st.set_page_config(
    page_title="Asset Management & Quantitative Risk Engine", 
    layout="wide",
    page_icon="📊"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .stButton button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 5px;
        font-weight: 600;
    }
    .stSelectbox, .stMultiselect, .stSlider, .stTextInput {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Title with professional styling
st.markdown('<div class="main-header">Asset Management & Quantitative Risk Engine</div>', unsafe_allow_html=True)
st.caption("Advanced Portfolio Optimization with Geopolitical Risk Integration")

# --- SIDEBAR: STRATEGIC CONTROLS ---
with st.sidebar:
    st.markdown("### Strategic Parameters")
    
    default_tickers = "AAPL, MSFT, JPM, MC.PA, ASML, NESN.SW"
    assets = st.text_input("Tickers (comma-separated)", default_tickers)
    ticker_list = [t.strip() for t in assets.split(",") if t.strip()]
    
    # SINGLE DATE INPUT AS ORIGINAL
    start_date = st.date_input("Analysis Start", value=pd.to_datetime("2021-01-01"))
    
    st.divider()
    st.markdown("### Geopolitical Risk Overlay")
    geo_events = st.multiselect(
        "Active Events",
        ["US-China Tech Tensions", "EU Regulation Shift", "Middle East Instability", 
         "Supply Chain Disruption", "Currency Volatility", "Trade Policy Changes"],
        default=["US-China Tech Tensions"]
    )
    geo_intensity = st.slider("Risk Intensity", 0.5, 3.0, 1.0, 0.1)

    st.divider()
    st.markdown("### Black-Litterman View")
    view_ticker = st.selectbox("Asset for View", ticker_list if ticker_list else ["AAPL"])
    view_return = st.slider(f"Return for {view_ticker} (%)", -20, 40, 10) / 100
    view_conf = st.slider("View Confidence (%)", 10, 100, 50) / 100

    st.divider()
    st.markdown("### Compliance & Risk")
    max_cap = st.slider("Max Weight per Stock (%)", 10, 100, 35) / 100
    div_penalty = st.slider("Diversification Penalty", 0.0, 2.0, 0.5)

# --- CURRENCY CONVERSION SYSTEM ---
def convert_to_base_currency(prices_df, base_currency='USD'):
    """Convert all prices to base currency using FX rates"""
    
    # Map tickers to their native currencies
    ticker_currency_map = {
        'AAPL': 'USD', 'MSFT': 'USD', 'JPM': 'USD', 
        'MC.PA': 'EUR', 'ASML': 'EUR', 'NESN.SW': 'CHF',
        '2330.TW': 'TWD', '7203.T': 'JPY', '^GSPC': 'USD',
        'GOOGL': 'USD', 'AMZN': 'USD', 'TSLA': 'USD',
        'XOM': 'USD', 'CVX': 'USD', 'WMT': 'USD'
    }
    
    # Get FX rates relative to USD
    fx_pairs = {
        'EUR': 'EURUSD=X',
        'CHF': 'CHFUSD=X',
        'JPY': 'JPYUSD=X',
        'TWD': 'TWDUSD=X',
        'GBP': 'GBPUSD=X',
        'CAD': 'CADUSD=X',
        'AUD': 'AUDUSD=X'
    }
    
    # Get FX data for all needed currencies
    needed_currencies = set()
    for ticker in prices_df.columns:
        if ticker in ticker_currency_map:
            currency = ticker_currency_map[ticker]
            if currency != 'USD':
                needed_currencies.add(currency)
    
    fx_rates = {}
    for currency in needed_currencies:
        if currency in fx_pairs:
            try:
                fx_data = yf.download(fx_pairs[currency], start=start_date, progress=False)['Close']
                if not fx_data.empty:
                    # Align indices
                    fx_rates[currency] = fx_data.reindex(prices_df.index, method='ffill').ffill().bfill()
            except:
                # Fallback rates if API fails
                fallback_rates = {
                    'EUR': 1.08, 'CHF': 1.12, 'JPY': 0.0068, 
                    'TWD': 0.031, 'GBP': 1.27, 'CAD': 0.74, 'AUD': 0.66
                }
                fx_rates[currency] = pd.Series(fallback_rates.get(currency, 1.0), index=prices_df.index)
    
    # Convert prices
    converted_prices = pd.DataFrame(index=prices_df.index)
    
    for ticker in prices_df.columns:
        if ticker in ticker_currency_map:
            currency = ticker_currency_map[ticker]
            if currency == 'USD' or currency not in fx_rates:
                converted_prices[ticker] = prices_df[ticker]
            else:
                # If we have EUR stock and EUR/USD rate, multiply: EUR * (EUR/USD) = USD
                converted_prices[ticker] = prices_df[ticker] * fx_rates[currency]
        else:
            # Default to USD if unknown
            converted_prices[ticker] = prices_df[ticker]
    
    return converted_prices

# --- GEOPOLITICAL LOGIC ---
def apply_geopolitical_overlay(weights, tickers, events, intensity):
    """Adjust portfolio weights based on geopolitical risk exposure per sector."""
    if not events or intensity <= 0.5:
        return weights
    
    sector_risk = {
        'Technology': {'US-China Tech Tensions': 0.8, 'Supply Chain Disruption': 0.7, 'Trade Policy Changes': 0.6},
        'Financials': {'Currency Volatility': 0.6, 'Trade Policy Changes': 0.4, 'Middle East Instability': 0.3},
        'Automotive': {'Supply Chain Disruption': 0.9, 'Trade Policy Changes': 0.7, 'EU Regulation Shift': 0.5},
        'Semiconductors': {'US-China Tech Tensions': 0.9, 'Supply Chain Disruption': 0.8, 'Trade Policy Changes': 0.7},
        'Healthcare': {'EU Regulation Shift': 0.5, 'Trade Policy Changes': 0.3, 'Currency Volatility': 0.2},
        'Energy': {'Middle East Instability': 0.8, 'Trade Policy Changes': 0.6, 'Currency Volatility': 0.4},
        'Consumer': {'Supply Chain Disruption': 0.5, 'Trade Policy Changes': 0.4, 'Currency Volatility': 0.3}
    }
    
    ticker_sectors = {
        'AAPL': 'Technology', 'MSFT': 'Technology', 'JPM': 'Financials',
        'MC.PA': 'Automotive', 'ASML': 'Semiconductors', 
        'NESN.SW': 'Healthcare', '2330.TW': 'Semiconductors', 
        '7203.T': 'Automotive', 'GOOGL': 'Technology', 
        'AMZN': 'Technology', 'TSLA': 'Automotive',
        'XOM': 'Energy', 'CVX': 'Energy', 'WMT': 'Consumer'
    }
    
    adjustments = {}
    
    for ticker in tickers:
        if ticker not in weights:
            continue
        sector = ticker_sectors.get(ticker, 'Technology')
        total_risk = 0
        for event in events:
            total_risk += sector_risk.get(sector, {}).get(event, 0.1)
        reduction_factor = 1 - (total_risk * intensity * 0.15)
        adjustments[ticker] = max(0.01, weights[ticker] * reduction_factor)
    
    if not adjustments:
        return weights
    
    # Re-normalize to 100%
    total = sum(adjustments.values())
    if total > 0:
        normalized = {k: v/total for k, v in adjustments.items()}
        return normalized
    return weights

# --- DATA ENGINE WITH CURRENCY CONVERSION ---
@st.cache_data(ttl=3600)
def get_clean_data(tickers, start):
    """Safe data fetching with currency conversion - FIXED VERSION"""
    if not tickers:
        return pd.DataFrame(), pd.Series(), {}
    
    tickers = tickers[:10]
    all_tickers = tickers + ["^GSPC"]
    
    try:
        # Download all data at once - FIX: Use proper parameter names
        raw_data = yf.download(all_tickers, start=start, progress=False, group_by='ticker')
        
        # DEBUG: Check structure
        if isinstance(raw_data.columns, pd.MultiIndex):
            # MultiIndex structure - extract Close prices for each ticker
            close_prices = pd.DataFrame()
            for ticker in all_tickers:
                if ticker in raw_data.columns.get_level_values(0):
                    close_prices[ticker] = raw_data[ticker]['Close']
                else:
                    # Try alternative approach
                    try:
                        # Download individually if not found
                        ticker_data = yf.download(ticker, start=start, progress=False)
                        if not ticker_data.empty:
                            close_prices[ticker] = ticker_data['Close']
                    except:
                        continue
            raw_prices = close_prices
        else:
            # Single level column structure
            if 'Close' in raw_data.columns:
                raw_prices = raw_data['Close'].copy()
            else:
                raw_prices = raw_data.copy()
        
        time.sleep(1)
    except Exception as e:
        st.warning(f"Bulk download failed: {str(e)}. Trying individual tickers...")
        raw_prices = pd.DataFrame()
        for t in all_tickers:
            try:
                ticker_data = yf.download(t, start=start, progress=False)
                if not ticker_data.empty and 'Close' in ticker_data.columns:
                    raw_prices[t] = ticker_data['Close']
                time.sleep(0.3)
            except Exception as e:
                st.warning(f"Failed to download {t}: {str(e)}")
                continue
    
    if raw_prices.empty:
        st.error("Could not fetch data. Please check ticker symbols and try again.")
        return pd.DataFrame(), pd.Series(), {}
    
    # Ensure we have a DataFrame
    if isinstance(raw_prices, pd.Series):
        raw_prices = raw_prices.to_frame()
    
    # Convert all prices to USD
    converted_data = convert_to_base_currency(raw_prices)
    
    # Clean and separate
    clean_df = converted_data.ffill().dropna()
    if clean_df.empty:
        return pd.DataFrame(), pd.Series(), {}
    
    benchmark = clean_df["^GSPC"] if "^GSPC" in clean_df.columns else pd.Series()
    
    # Remove benchmark from assets data
    asset_columns = [col for col in clean_df.columns if col != "^GSPC"]
    assets_data = clean_df[asset_columns].copy()
    
    # Fixed market caps (in USD)
    fixed_caps = {
        'AAPL': 2.8e12, 'MSFT': 2.5e12, 'JPM': 0.5e12,
        'MC.PA': 0.07e12, 'ASML': 0.3e12, 'NESN.SW': 0.3e12,
        '2330.TW': 0.5e12, '7203.T': 0.03e12,
        'GOOGL': 1.8e12, 'AMZN': 1.6e12, 'TSLA': 0.6e12,
        'XOM': 0.4e12, 'CVX': 0.3e12, 'WMT': 0.45e12
    }
    
    mcaps = {}
    for t in tickers:
        if t in assets_data.columns:
            mcaps[t] = fixed_caps.get(t, 1e11)
    
    return assets_data, benchmark, mcaps

# --- MAIN EXECUTION ---
try:
    if not ticker_list:
        st.warning("Please enter at least one ticker symbol.")
        st.stop()
    
    # Load data - FIXED: Use only start_date as parameter
    with st.spinner("Fetching market data and converting currencies..."):
        prices, bench_prices, market_caps = get_clean_data(ticker_list, start_date)
    
    if prices.empty:
        st.error("No data available for the selected tickers/period.")
        st.stop()
    
    # Ensure ticker_list matches available data
    available_tickers = [t for t in ticker_list if t in prices.columns]
    if not available_tickers:
        st.error("None of the entered tickers returned valid data.")
        st.stop()
    
    ticker_list = available_tickers
    
    # Calculate returns - Handle both single and multiple tickers
    if len(ticker_list) == 1:
        # Single ticker case
        returns = prices[ticker_list[0]].pct_change().dropna().to_frame()
        returns.columns = ticker_list
    else:
        # Multiple tickers case
        returns = prices[ticker_list].pct_change().dropna()
    
    if not bench_prices.empty:
        bench_returns = bench_prices.pct_change().dropna()
        # Align dates
        common_idx = returns.index.intersection(bench_returns.index)
        if len(common_idx) > 0:
            returns = returns.loc[common_idx]
            bench_returns = bench_returns.loc[common_idx]
        else:
            bench_returns = pd.Series()
    else:
        bench_returns = pd.Series()
    
    if returns.empty or len(returns) < 10:
        st.error("Insufficient data for analysis. Try a longer time period or different tickers.")
        st.stop()
    
    # 1. Black-Litterman Optimization
    with st.spinner("Optimizing portfolio..."):
        # Ensure we have enough data for covariance matrix
        if len(ticker_list) > 1:
            S = risk_models.sample_cov(prices[ticker_list])
        else:
            # Single asset case - create a dummy covariance matrix
            S = pd.DataFrame({ticker_list[0]: [0.04]}, index=ticker_list)
        
        available_caps = {t: market_caps.get(t, 1e11) for t in ticker_list}
        
        # Calculate prior returns
        try:
            prior_rets = black_litterman.market_implied_prior_returns(available_caps, 2.5, S)
        except Exception as e:
            st.warning(f"Using fallback prior returns: {str(e)}")
            # Fallback: Use historical returns
            prior_rets = returns.mean() * 252
        
        # Ensure view ticker is in our list
        if view_ticker not in ticker_list:
            view_ticker = ticker_list[0]
        
        # Create Black-Litterman model
        try:
            bl = black_litterman.BlackLittermanModel(
                S, 
                pi=prior_rets, 
                absolute_views={view_ticker: view_return}, 
                omega="idzorek", 
                view_confidences=[view_conf]
            )
            bl_mu = bl.bl_returns()
        except Exception as e:
            st.warning(f"Black-Litterman failed, using mean-variance optimization: {str(e)}")
            bl_mu = prior_rets
    
    # 2. Base Optimization
    try:
        ef = EfficientFrontier(bl_mu, S, weight_bounds=(0, max_cap))
        ef.add_objective(objective_functions.L2_reg, gamma=div_penalty)
        optimized_weights = ef.max_sharpe()
        optimized_weights = ef.clean_weights()
    except Exception as e:
        st.error(f"Optimization failed: {str(e)}. Using equal weights as fallback.")
        optimized_weights = {t: 1/len(ticker_list) for t in ticker_list}
    
    # 3. Geopolitical Adjustment
    if geo_events and geo_intensity > 0.5:
        final_weights = apply_geopolitical_overlay(optimized_weights, ticker_list, geo_events, geo_intensity)
    else:
        final_weights = optimized_weights
    
    # Convert to array for calculations
    weights_arr = np.array([final_weights.get(t, 0) for t in ticker_list])
    
    # 4. Performance Calculations
    if len(ticker_list) == 1:
        p_rets = returns[ticker_list[0]] * weights_arr[0]
    else:
        p_rets = (returns * weights_arr).sum(axis=1)
    
    p_cum = (1 + p_rets).cumprod()
    
    if not bench_returns.empty:
        b_cum = (1 + bench_returns).cumprod()
    
    # Analytics
    ann_ret = p_rets.mean() * 252
    ann_vol = p_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    max_dd = ((p_cum - p_cum.cummax()) / p_cum.cummax()).min()
    sortino = ann_ret / (p_rets[p_rets < 0].std() * np.sqrt(252)) if len(p_rets[p_rets < 0]) > 0 else 0

    # --- PERFORMANCE DASHBOARD ---
    st.markdown('<div class="sub-header">Performance & Risk Analytics</div>', unsafe_allow_html=True)
    
    # Metrics in professional cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f'<div class="metric-card">Sharpe Ratio<br><span style="font-size: 2rem;">{sharpe:.2f}</span></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card">Annual Return<br><span style="font-size: 2rem;">{ann_ret:.1%}</span></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card">Annual Volatility<br><span style="font-size: 2rem;">{ann_vol:.1%}</span></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card">Max Drawdown<br><span style="font-size: 2rem;">{max_dd:.1%}</span></div>', unsafe_allow_html=True)

    # Performance Chart
    st.divider()
    st.markdown('<div class="sub-header">Cumulative Returns</div>', unsafe_allow_html=True)
    
    fig_perf = go.Figure()
    fig_perf.add_trace(go.Scatter(
        x=p_cum.index, 
        y=p_cum, 
        name="Strategy", 
        line=dict(color='#1E3A8A', width=3),
        fill='tozeroy',
        fillcolor='rgba(30, 58, 138, 0.1)'
    ))
    
    if not bench_returns.empty:
        fig_perf.add_trace(go.Scatter(
            x=b_cum.index, 
            y=b_cum, 
            name="S&P 500", 
            line=dict(color='#6B7280', dash='dash', width=2)
        ))
    
    fig_perf.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=400,
        xaxis_title="",
        yaxis_title="Cumulative Return",
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial, sans-serif")
    )
    
    # Add grid
    fig_perf.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
    fig_perf.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
    
    st.plotly_chart(fig_perf, use_container_width=True)

    # --- PORTFOLIO ALLOCATION ---
    st.divider()
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown('<div class="sub-header">Portfolio Allocation</div>', unsafe_allow_html=True)
        
        w_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        w_df = w_df[w_df['Weight'] > 0.001]
        
        if not w_df.empty:
            # Create a professional horizontal bar chart
            w_df = w_df.sort_values('Weight', ascending=True)
            fig_alloc = go.Figure(go.Bar(
                y=w_df.index,
                x=w_df['Weight'] * 100,
                orientation='h',
                marker=dict(
                    color='#1E3A8A',
                    line=dict(color='#1D4ED8', width=1)
                ),
                text=[f'{w*100:.1f}%' for w in w_df['Weight']],
                textposition='auto',
            ))
            
            fig_alloc.update_layout(
                plot_bgcolor='white',
                paper_bgcolor='white',
                height=400,
                xaxis_title="Weight (%)",
                yaxis_title="",
                font=dict(family="Arial, sans-serif")
            )
            fig_alloc.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
            
            st.plotly_chart(fig_alloc, use_container_width=True)
        else:
            st.info("No significant allocations found.")
    
    with col_right:
        st.markdown('<div class="sub-header">Geopolitical Impact Analysis</div>', unsafe_allow_html=True)
        
        if geo_events and geo_intensity > 0.5:
            changes = {}
            for t in ticker_list:
                orig = optimized_weights.get(t, 0)
                new = final_weights.get(t, 0)
                change = (new - orig) * 100
                if abs(change) > 0.05:
                    changes[t] = change
            
            if changes:
                changes_df = pd.DataFrame.from_dict(changes, orient='index', columns=['Change (%)'])
                changes_df = changes_df.sort_values('Change (%)', ascending=True)
                
                fig_changes = go.Figure(go.Bar(
                    y=changes_df.index,
                    x=changes_df['Change (%)'],
                    orientation='h',
                    marker=dict(
                        color=['#DC2626' if x < 0 else '#059669' for x in changes_df['Change (%)']],
                        line=dict(color='white', width=0.5)
                    ),
                    text=[f'{x:+.1f}pp' for x in changes_df['Change (%)']],
                    textposition='auto',
                ))
                
                fig_changes.update_layout(
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    height=400,
                    xaxis_title="Weight Change (percentage points)",
                    yaxis_title="",
                    font=dict(family="Arial, sans-serif"),
                    showlegend=False
                )
                fig_changes.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
                
                st.plotly_chart(fig_changes, use_container_width=True)
                
                # Summary metrics
                total_reduction = sum(v for v in changes.values() if v < 0)
                total_increase = sum(v for v in changes.values() if v > 0)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Total Risk Reduction", f"{abs(total_reduction):.1f} pp", delta="-")
                with col2:
                    st.metric("Risk-Averse Increase", f"{total_increase:.1f} pp", delta="+")
            else:
                st.info("Geopolitical events had minimal impact on current allocation.")
        else:
            st.info("Enable geopolitical risk overlay in sidebar to see adjustments.")

    # --- RISK ANALYSIS ---
    st.divider()
    st.markdown('<div class="sub-header">Risk Decomposition</div>', unsafe_allow_html=True)
    
    # Calculate risk contributions
    if len(ticker_list) > 1:
        # Calculate covariance matrix
        cov_matrix = returns.cov() * 252
        
        # Calculate risk contributions
        portfolio_variance = weights_arr.T @ cov_matrix.values @ weights_arr
        marginal_contributions = (cov_matrix.values @ weights_arr) / np.sqrt(portfolio_variance) if portfolio_variance > 0 else weights_arr * 0
        risk_contributions = weights_arr * marginal_contributions
        
        risk_df = pd.DataFrame({
            'Asset': ticker_list,
            'Weight': [final_weights.get(t, 0) * 100 for t in ticker_list],
            'Risk Contribution (%)': risk_contributions * 100 / risk_contributions.sum() if risk_contributions.sum() > 0 else [0] * len(ticker_list)
        })
        
        fig_risk = go.Figure(go.Bar(
            x=risk_df['Asset'],
            y=risk_df['Risk Contribution (%)'],
            marker_color='#DC2626',
            text=risk_df['Risk Contribution (%)'].apply(lambda x: f'{x:.1f}%'),
            textposition='auto'
        ))
        
        fig_risk.update_layout(
            plot_bgcolor='white',
            paper_bgcolor='white',
            height=300,
            xaxis_title="Asset",
            yaxis_title="Risk Contribution (%)",
            font=dict(family="Arial, sans-serif")
        )
        fig_risk.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
        fig_risk.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#F3F4F6')
        
        st.plotly_chart(fig_risk, use_container_width=True)
    else:
        st.info("Risk decomposition requires multiple assets.")

    # --- EXPORT SECTION ---
    st.divider()
    st.markdown('<div class="sub-header">Export Results</div>', unsafe_allow_html=True)
    
    col_exp1, col_exp2, col_exp3 = st.columns(3)
    
    with col_exp1:
        # Export weights
        export_df = pd.DataFrame.from_dict(final_weights, orient='index', columns=['Weight'])
        export_df['Weight'] = export_df['Weight'].apply(lambda x: f'{x:.3%}')
        csv = export_df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Portfolio Weights",
            data=csv,
            file_name='portfolio_weights.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp2:
        # Export performance
        perf_df = pd.DataFrame({
            'Date': p_cum.index,
            'Strategy_Cumulative': p_cum.values,
            'Strategy_Daily_Return': p_rets.values
        })
        if not bench_returns.empty:
            perf_df['SP500_Cumulative'] = b_cum.values
        
        csv_perf = perf_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Performance Data",
            data=csv_perf,
            file_name='performance_data.csv',
            mime='text/csv',
            use_container_width=True
        )
    
    with col_exp3:
        # Export optimization parameters
        params_df = pd.DataFrame({
            'Parameter': ['View Asset', 'View Return', 'View Confidence', 'Max Weight', 'Diversification Penalty', 
                         'Geopolitical Events', 'Risk Intensity', 'Analysis Period'],
            'Value': [view_ticker, f'{view_return:.1%}', f'{view_conf:.0%}', f'{max_cap:.0%}', div_penalty,
                     ', '.join(geo_events) if geo_events else 'None', geo_intensity, 
                     f'From {start_date}']
        })
        csv_params = params_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Strategy Parameters",
            data=csv_params,
            file_name='strategy_parameters.csv',
            mime='text/csv',
            use_container_width=True
        )

except Exception as e:
    st.error(f"Execution Error: {str(e)}")
    
    import traceback
    st.code(traceback.format_exc())
    
    if "rate limit" in str(e).lower():
        st.info("""
        **Rate Limit Issue Detected**
        
        Yahoo Finance has rate limits. Please try:
        1. Wait 1-2 minutes and refresh the application
        2. Use fewer tickers (4-6 recommended)
        3. Use the default ticker set provided
        """)
    
    elif "ticker" in str(e).lower():
        st.info("""
        **Ticker Symbol Issue**
        
        Please verify:
        1. Ticker symbols are correct (e.g., AAPL for Apple)
        2. International tickers include exchange suffix (e.g., MC.PA for LVMH on Paris exchange)
        3. Symbols are comma-separated without spaces between commas
        """)
