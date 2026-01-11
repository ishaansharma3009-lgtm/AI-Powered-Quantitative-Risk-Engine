# 🏛️ Institutional Portfolio Strategy & Risk Engine
**Developed by Ishaan Sharma**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_svg)](https://ai-powered-quantitative-risk-engine-jwyqn5b4hxxjbkhgdoscw5.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ishaansharma3009-lgtm/AI-Powered-Quantitative-Risk-Engine/blob/main/LICENSE)

## 📌 Project Overview
This tool is an Asset Management solution designed to automate strategic asset allocation and optimize portfolio weights based on risk-return objectives.". It automates the identification of "optimal" asset allocations by balancing historical returns against systemic risk factors using live market data.

## 📊 Core Features
* **Mean-Variance Optimization (MVO):** Identifies the "Max Sharpe" portfolio to provide the highest return per unit of risk.
* **Market Regime Monitor:** Integrates live **VIX (Volatility Index)** data to adjust strategy recommendations based on current market "fear."
* **Institutional Risk Controls:** Implements a 35% concentration cap to ensure professional-grade diversification.
* **Risk Analytics:** Provides **95% Value-at-Risk (VaR)** metrics and automated benchmarking against the **S&P 500 (SPY)**.

## 🧮 Mathematical Framework
The engine seeks to maximize the **Sharpe Ratio** ($S$):

$$S = \frac{R_p - R_f}{\sigma_p}$$

* **$R_p$**: Expected Portfolio Return
* **$R_f$**: Risk-Free Rate
* **$\sigma_p$**: Portfolio Volatility (Standard Deviation)

## 🛠️ Technical Stack
- **Python 3.9+**
- **yfinance:** Real-time financial data acquisition.
- **PyPortfolioOpt:** Quadratic programming for convex optimization.
- **Streamlit:** Interactive dashboard and UI.
- **Plotly:** Dynamic financial time-series visualization.

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone [https://github.com/ishaansharma3009-lgtm/AI-Powered-Quantitative-Risk-Engine.git](https://github.com/ishaansharma3009-lgtm/AI-Powered-Quantitative-Risk-Engine.git)
   ---

## 👤 Author

**Ishaan Sharma**
*Aspiring Asset Management Professional*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishaan-sharma-128694374/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ishaansharma3009-lgtm)

**Disclaimer:** *This project is a technical demonstration of Asset Management principles. It uses unhedged local currency data. All investment strategies involve a risk of loss, and past performance is not indicative of future results.*
---
