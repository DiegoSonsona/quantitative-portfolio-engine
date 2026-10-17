::: {align="center"}
# Quantitative Portfolio Optimization Engine

**A Python framework for portfolio construction, risk analysis,
backtesting, and long-term investment projections.**
:::

------------------------------------------------------------------------

## Overview

The **Quantitative Portfolio Optimization Engine** is an end-to-end
framework for researching portfolio allocations and evaluating their
historical performance. It combines Bayesian return estimation,
covariance shrinkage, Monte Carlo resampling, and out-of-sample
backtesting to help address the estimation error that can make
traditional mean-variance optimization unstable.

The pipeline retrieves and prepares market data, calculates asset-level
risk metrics, estimates portfolio allocations under configured
constraints, and produces reports for further analysis.

> **Disclaimer:** This project is for educational and research purposes
> only. Historical performance and simulated projections do not
> guarantee future results. Outputs depend on data quality, modeling
> assumptions, and configuration; they should not be treated as
> investment advice.

## Features

-   **Market data preparation** --- Retrieves time-series data with
    `yfinance`, handles MultiIndex columns, normalizes time zones, and
    resamples prices to a weekly frequency.
-   **Asset-level risk analysis** --- Calculates CAPM alpha and beta,
    Sharpe ratio, Sortino ratio, Calmar ratio, and maximum drawdown.
-   **Bayesian return estimation** --- Uses a Black--Litterman approach
    to combine market-implied returns with configured views.
-   **Covariance shrinkage** --- Applies Ledoit--Wolf shrinkage to
    estimate a more stable covariance matrix.
-   **Resampled portfolio optimization** --- Runs Monte Carlo
    simulations (configured for 500 draws) to explore portfolio weights
    under configured constraints, including sector caps.
-   **Walk-forward evaluation** --- Evaluates portfolio performance out
    of sample using a 70/30 train-test split and compares results with
    `SPY`.
-   **Long-term contribution projections** --- Generates 30-year
    dollar-cost averaging (DCA) scenarios using 1,000
    historical-bootstrap paths.
-   **Automated reporting** --- Exports CSV diagnostics, a five-panel
    PNG dashboard, and an interactive QuantStats HTML report.

## Requirements

-   Python 3
-   The packages listed in `requirements.txt`
-   A working internet connection for market data retrieval through
    `yfinance`

> Confirm the Python version and dependency versions supported by your
> implementation in `requirements.txt` before publishing.

## Getting Started

### 1. Clone the repository

Replace `YOUR_USERNAME` with your GitHub username or organization, and
make sure the repository URL matches the repository you create.

``` bash
git clone https://github.com/YOUR_USERNAME/quantitative-portfolio-engine.git
cd quantitative-portfolio-engine
```

### 2. Install dependencies

It is recommended to use a virtual environment.

**macOS / Linux**

``` bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**Windows (PowerShell)**

``` powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Run the pipeline

``` bash
python master_portfolio_engine.py
```

The script is expected to write the output files listed below to the
project root. Actual filenames and behavior depend on the implementation
and configuration.

## Generated Artifacts

  -----------------------------------------------------------------------
  File                                Description
  ----------------------------------- -----------------------------------
  `01_Pesos_Optimos_Cartera.csv`      Estimated optimal portfolio
                                      allocation weights.

  `02_Matriz_Correlacion.csv`         Asset return correlation matrix.

  `03_Metricas_Activos.csv`           Asset-level CAPM and risk-adjusted
                                      performance metrics.

  `Master_Portfolio_Dashboard.png`    Five-panel portfolio analysis
                                      dashboard.

  `Portfolio_Master_vs_SPY.html`      Interactive portfolio-versus-`SPY`
                                      performance report generated with
                                      QuantStats.
  -----------------------------------------------------------------------

## Project Structure

The repository should contain at least the following files:

``` 
quantitative-portfolio-engine/
├── master_portfolio_engine.py
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```

Add any additional source modules, configuration files, tests, or data
directories that your implementation requires.

## Reproducibility and Interpretation

-   Historical market data may be revised or unavailable, and downloads
    can fail.
-   Optimization results are sensitive to the asset universe, estimation
    window, constraints, and return assumptions.
-   Backtest results depend on the test period, rebalancing rules,
    transaction-cost assumptions, and treatment of missing data.
-   Bootstrap projections are scenarios based on historical
    observations, not reliable predictions of future returns.
-   Before relying on results, verify that the implementation avoids
    look-ahead bias and accounts for relevant trading costs, dividends,
    and corporate actions.

