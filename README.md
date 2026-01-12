# 🏛️ Institutional Strategy & Geopolitical Risk Engine  
**Developed by Ishaan Sharma**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://ai-powered-quantitative-risk-engine-jwyqn5b4hxxjbkhgdoscw5.streamlit.app)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/ishaansharma3009-lgtm/AI-Powered-Quantitative-Risk-Engine/blob/main/LICENSE)

## 📌 Project Overview
An advanced portfolio optimization platform that combines **Black-Litterman quantitative methods** with **geopolitical risk assessment** to deliver institutionally robust asset allocation. The engine balances mathematical optimization with real-world strategic adaptation for professional asset managers.

## 📊 Core Features
* **Black-Litterman Optimization:** Implements market-implied equilibrium returns with Bayesian investor views to overcome estimation errors in traditional mean-variance optimization.
* **Geopolitical Risk Overlay:** Dynamically adjusts portfolio weights based on active global events (US-China tensions, supply chain risks, regulatory shifts) with sector-specific vulnerability scoring.
* **Institutional Risk Controls:** Configurable position limits (1-35% per stock) and diversification penalties to enforce professional compliance standards.
* **Performance Analytics:** Sharpe ratio, maximum drawdown, annualized volatility, and S&P 500 benchmarking with interactive growth visualization.
* **Strategic Allocation:** Outputs geopolitically-adjusted optimal weights with transparent adjustment reporting.

## 🧮 Mathematical Framework
The engine implements the **Black-Litterman model** which combines market equilibrium returns with investor views:

$$\Pi = \delta \Sigma w_{mkt}$$
$$E(R) = [(\tau\Sigma)^{-1} + P'\Omega^{-1}P]^{-1}[(\tau\Sigma)^{-1}\Pi + P'\Omega^{-1}Q]$$

* **$\Pi$**: Market-implied equilibrium returns
* **$P$**: Investor view matrix  
* **$\Omega$**: View uncertainty matrix
* **$Q$**: Expected returns from views

## 🛠️ Technical Stack
- **Python 3.9+**
- **yfinance:** Real-time market data via Yahoo Finance API
- **PyPortfolioOpt:** Black-Litterman implementation and portfolio optimization
- **Streamlit:** Interactive institutional dashboard
- **Plotly:** Professional financial visualizations
- **pandas/numpy:** Data processing and quantitative calculations

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ishaansharma3009-lgtm/AI-Powered-Quantitative-Risk-Engine.git
   cd AI-Powered-Quantitative-Risk-Engine
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Launch the application:
   ```bash
   streamlit run app.py
   ```

## 📈 Example Usage
1. Input asset tickers (e.g., `AAPL, MSFT, JPM, MC.PA, ASML`)
2. Set geopolitical risk parameters (events, intensity)
3. Define Black-Litterman view (asset forecast + confidence)
4. Adjust compliance constraints (max weights, diversification)
5. Analyze optimized allocation and download results

## 🎓 Academic & Professional Relevance
This implementation bridges **theoretical finance** (Black & Litterman, 1992) with **practical portfolio management** by incorporating geopolitical intelligence—a critical dimension in modern asset allocation. The tool demonstrates graduate-level quantitative skills while addressing real-world investment challenges.

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Important Disclaimer
*This tool is for educational and research purposes only. It demonstrates quantitative finance concepts and is not financial advice. Past performance does not guarantee future results. All investment strategies involve risk of loss.*

---

## 👤 Author

**Ishaan Sharma**  
*Aspiring Asset Management & Quantitative Finance Professional*

[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/ishaan-sharma-128694374/)
[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/ishaansharma3009-lgtm)

**Research Focus:** Portfolio optimization, geopolitical risk integration, quantitative asset allocation strategies.

