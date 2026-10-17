# Master Portfolio Engine Documentation

## System Architecture

The `master_portfolio_engine.py` script implements an end-to-end quantitative portfolio construction and validation framework divided into 7 distinct stages:

### 1. Data Ingestion & Time-Series Alignment
- Downloads up to 10 years of historical daily price series for a target asset universe plus a benchmark index (`SPY`).
- Cleans multi-index data structures, normalizes timezones, and handles missing exchange trading days via forward/backward filling.

### 2. CAPM Diagnostics & Risk Radiography
Calculates single-asset risk-adjusted performance metrics against the market benchmark:
- **CAGR & Volatility**: Annualized return and standard deviation.
- **CAPM Alpha & Beta**: Regression of excess returns over risk-free rate ($R_f = 4.2\%$).
- **Sharpe & Sortino Ratios**: Measure total risk-adjusted and downside risk-adjusted return.
- **Maximum Drawdown (MDD) & Calmar Ratio**: Worst peak-to-trough decline and return-to-MDD ratio.

### 3. Portfolio Optimization Engine
Combines three advanced econometric methods:
- **Ledoit-Wolf Covariance Shrinkage**: Replaces standard sample covariance to minimize estimation error and overfitting.
- **Black-Litterman Model**: Blends market equilibrium returns (implied by market cap) with subjective analyst views (annual return expectations).
- **Monte Carlo Resampled Markowitz**: Simulates 500 multivariate normal return vectors to compute mean optimal weights under bounds ($2\% - 22\%$) and hard sector allocation caps.

### 4. Out-of-Sample Walk-Forward Backtesting
- Splits price data into 70% in-sample training and 30% out-of-sample testing periods.
- Applies optimal weights directly to the out-of-sample window to evaluate real historical tracking performance vs. `SPY`.

### 5. Wealth Projection Simulation (DCA Engine)
- Simulates a 30-year Dollar-Cost Averaging strategy (€1,000 initial + €100/month) using 1,000 historical bootstrap iterations.
- Outputs P10, P50 (median), and P90 cumulative capital growth boundaries.

---

## Technical Limitations & Future Improvements

1. **Static Weight Backtest Assumption**:
   - The current out-of-sample test holds fixed asset weights over the entire evaluation period. Implementing dynamic quarterly or semi-annual rebalancing loops during the backtest window will yield even more realistic performance tracking.

2. **Data Provider Dependency**:
   - The extraction pipeline relies on `yfinance`, which can be subject to rate limits or API structure updates by Yahoo Finance.

3. **Covariance Stationary Assumption**:
   - Ledoit-Wolf shrinkage improves covariance stability, but asset correlations can shift rapidly during systemic market crashes.