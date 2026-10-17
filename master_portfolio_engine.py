# -*- coding: utf-8 -*-

import os
import time
import warnings
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import yfinance as yf
from scipy import stats
import scipy.optimize as sco
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel

try:
    import quantstats as qs
    qs.extend_pandas()
    HAS_QUANTSTATS = True
except ImportError:
    HAS_QUANTSTATS = False

warnings.filterwarnings("ignore")

PORTFOLIO_TICKERS = [
    'TSM', 'NKE', 'META', 'PYPL', 'CROX',
    'NVDA', 'DUOL', 'HIMS', 'COPX', 'GDX',
    'NVTS', 'ADBE', 'SFM', 'TTE', 'NVO',
    'PFE', 'IWM', 'URTH', 'XBI', 'BOTZ'
]

BENCHMARK_TICKER = 'SPY'
LOOKBACK_YEARS = 10
RISK_FREE_RATE = 0.042

SUBJECTIVE_VIEWS = {
    'NVDA': 0.15,
    'GDX': 0.08,
    'NKE': 0.06,
    'PYPL': 0.12
}

SECTORS = {
    'IA_TECH' : ['NVDA', 'NVTS', 'META', 'ADBE', 'TSM', 'DUOL', 'BOTZ'],
    'SALUD'   : ['NVO', 'PFE', 'HIMS', 'XBI'],
    'CONSUMO' : ['NKE', 'CROX', 'SFM', 'PYPL'],
    'MATERIAS': ['COPX', 'GDX', 'TTE'],
    'OTROS'   : ['IWM', 'URTH']
}

SECTOR_LIMITS = {
    'IA_TECH' : 0.40,
    'SALUD'   : 0.30,
    'CONSUMO' : 0.30,
    'MATERIAS': 0.325,
    'OTROS'   : 0.20
}

INITIAL_CAPITAL = 1000
MONTHLY_CONTRIBUTION = 100
PROJECTION_YEARS = 30
TRAIN_RATIO = 0.70


def download_market_data(tickers, benchmark='SPY', years=10):
    end_date = datetime.datetime.today()
    start_date = end_date - datetime.timedelta(days=365 * years)
    all_tickers = list(set(tickers + [benchmark]))

    df_raw = yf.download(all_tickers, start=start_date, end=end_date, progress=False)

    data_dict = {}
    if isinstance(df_raw.columns, pd.MultiIndex):
        for t in all_tickers:
            if ('Adj Close', t) in df_raw.columns:
                data_dict[t] = df_raw[('Adj Close', t)]
            elif ('Close', t) in df_raw.columns:
                data_dict[t] = df_raw[('Close', t)]
            elif t in df_raw.columns.get_level_values(0):
                if 'Adj Close' in df_raw[t].columns:
                    data_dict[t] = df_raw[t]['Adj Close']
                elif 'Close' in df_raw[t].columns:
                    data_dict[t] = df_raw[t]['Close']
    else:
        for t in all_tickers:
            if t in df_raw.columns:
                data_dict[t] = df_raw[t]

    prices = pd.DataFrame(data_dict)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    prices = prices.ffill().bfill().dropna(how='all')
    return prices


def calculate_asset_metrics(prices, benchmark='SPY', rf_rate=0.042):
    daily_rets = prices.pct_change().dropna(how='all')
    rf_daily = rf_rate / 252
    metrics = []

    for ticker in daily_rets.columns:
        if ticker == benchmark: continue
        valid_data = daily_rets[[ticker, benchmark]].dropna()
        if len(valid_data) < 252: continue

        r_act, r_ben = valid_data[ticker], valid_data[benchmark]
        years = len(valid_data) / 252
        ann_ret = (1 + r_act).prod() ** (1 / years) - 1
        ann_vol = r_act.std() * np.sqrt(252)

        ex_act, ex_ben = r_act - rf_daily, r_ben - rf_daily
        beta, alpha_d, _, _, _ = stats.linregress(ex_ben, ex_act)
        alpha = alpha_d * 252

        sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else 0
        downside_vol = r_act[r_act < 0].std() * np.sqrt(252)
        sortino = (ann_ret - rf_rate) / downside_vol if downside_vol > 0 else 0

        cum_series = (1 + r_act).cumprod()
        mdd = ((cum_series - cum_series.cummax()) / cum_series.cummax()).min()
        calmar = ann_ret / abs(mdd) if mdd < 0 else 0

        metrics.append({
            'Ticker': ticker, 'Retorno_Anual': ann_ret, 'Volatilidad': ann_vol,
            'Alpha': alpha, 'Beta': beta, 'Sharpe': sharpe, 'Sortino': sortino,
            'Max_Drawdown': mdd, 'Calmar': calmar
        })
    df_metrics = pd.DataFrame(metrics).set_index('Ticker')
    return df_metrics


def optimize_portfolio(prices, views, sectors, sector_limits, benchmark='SPY', n_sims=500):
    weekly_prices = prices.resample('W-FRI').last()
    weekly_rets = weekly_prices.pct_change().dropna(how='all')
    tickers = [c for c in weekly_rets.columns if c != benchmark]
    weekly_rets = weekly_rets[tickers]

    S_lw = risk_models.CovarianceShrinkage(weekly_prices[tickers]).ledoit_wolf()

    market_caps = {}
    for t in tickers:
        try:
            mc = getattr(yf.Ticker(t).fast_info, 'market_cap', None)
            market_caps[t] = mc if mc and mc > 0 else 1e9
        except:
            market_caps[t] = 1e9

    mc_series = pd.Series(market_caps)
    mc_series = mc_series / mc_series.sum()
    valid_views = {k: v for k, v in views.items() if k in tickers}

    try:
        bl = BlackLittermanModel(S_lw, pi='market', market_caps=mc_series, risk_aversion=2.5, absolute_views=valid_views)
        ret_bl = bl.bl_returns()
    except Exception as e:
        ret_bl = expected_returns.mean_historical_return(weekly_prices[tickers])

    n_assets = len(tickers)

    def neg_sharpe(w, ret, cov):
        r_p = np.sum(ret * w)
        v_p = np.sqrt(np.dot(w.T, np.dot(cov, w)))
        return -(r_p / v_p) if v_p > 0 else 0

    bounds = tuple((0.02, 0.22) for _ in range(n_assets))
    constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

    ticker_to_sector = {}
    for sec, t_list in sectors.items():
        for t in t_list:
            if t in tickers: ticker_to_sector[t] = sec

    for sec, limit in sector_limits.items():
        sec_indices = [i for i, t in enumerate(tickers) if ticker_to_sector.get(t) == sec]
        if sec_indices:
            constraints.append({'type': 'ineq', 'fun': lambda x, idx=sec_indices, lim=limit: lim - np.sum(x[idx])})

    w0 = np.ones(n_assets) / n_assets
    res_clasico = sco.minimize(neg_sharpe, w0, args=(ret_bl.values, S_lw.values), method='SLSQP', bounds=bounds, constraints=constraints)
    w_classic = res_clasico.x if res_clasico.success else w0

    simulated_weights = []
    for _ in range(n_sims):
        sim_rets = np.random.multivariate_normal(ret_bl.values, S_lw.values / 52)
        res_sim = sco.minimize(neg_sharpe, w0, args=(sim_rets, S_lw.values), method='SLSQP', bounds=bounds, constraints=constraints)
        if res_sim.success:
            simulated_weights.append(res_sim.x)

    w_robusto = np.mean(simulated_weights, axis=0) if simulated_weights else w_classic

    df_weights = pd.DataFrame({
        'Activo': tickers,
        'Markowitz_Clasico': w_classic,
        'Markowitz_Robusto_MC': w_robusto
    }).set_index('Activo').sort_values(by='Markowitz_Robusto_MC', ascending=False)

    corr_matrix = weekly_rets.corr()
    return df_weights, corr_matrix, ret_bl, S_lw


def run_backtest(prices, weights_dict, benchmark='SPY', train_ratio=0.70):
    daily_rets = prices.pct_change().dropna(how='all')
    split_idx = int(len(daily_rets) * train_ratio)

    test_rets = daily_rets.iloc[split_idx:]
    tickers = [t for t in weights_dict.keys() if t in test_rets.columns]

    w_arr = np.array([weights_dict[t] for t in tickers])
    w_arr = w_arr / w_arr.sum()

    port_test_ret = test_rets[tickers].dot(w_arr)
    bench_test_ret = test_rets[benchmark] if benchmark in test_rets.columns else None

    return port_test_ret, bench_test_ret, test_rets.index[0].date()


def simulate_dca(port_returns, initial=1000, monthly=100, years=30, sims=1000):
    months = years * 12
    monthly_rets = port_returns.resample('ME').apply(lambda x: (1 + x).prod() - 1).dropna().values

    boot_paths = np.zeros((sims, months + 1))
    boot_paths[:, 0] = initial

    for s in range(sims):
        sampled_rets = np.random.choice(monthly_rets, size=months, replace=True)
        cap = initial
        for m_idx, r in enumerate(sampled_rets):
            cap = cap * (1 + r) + monthly
            boot_paths[s, m_idx + 1] = cap

    p50 = np.percentile(boot_paths, 50, axis=0)
    p10 = np.percentile(boot_paths, 10, axis=0)
    p90 = np.percentile(boot_paths, 90, axis=0)

    return p50, p10, p90, (initial + monthly * months)


def generate_reports(df_metrics, corr_matrix, df_weights, test_port, test_bench, dca_p50, dca_p10, dca_p90, total_contrib):
    df_weights.to_csv("01_Pesos_Optimos_Cartera.csv")
    corr_matrix.to_csv("02_Matriz_Correlacion.csv")
    df_metrics.to_csv("03_Metricas_Activos.csv")

    fig = plt.figure(figsize=(22, 18), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    ax1.set_facecolor('#161b22')
    top_w = df_weights['Markowitz_Robusto_MC'].sort_values(ascending=True)
    ax1.barh(top_w.index, top_w.values * 100, color='#00ff88', edgecolor='#0d1117')
    ax1.set_title('Resampled Monte Carlo Optimal Weights (%)', color='white', fontsize=11, fontweight='bold')
    ax1.tick_params(colors='white')
    ax1.grid(True, alpha=0.15, color='white')

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.set_facecolor('#161b22')
    ax2.axis('off')
    ax2.set_title('Top Risk-Adjusted Assets (Sharpe)', color='white', fontsize=11, fontweight='bold')
    top_sharpe = df_metrics.sort_values(by='Sharpe', ascending=False).head(6)
    y_pos = 0.85
    for idx, row in top_sharpe.iterrows():
        ax2.text(0.05, y_pos, f"{idx:<6} | Sharpe: {row['Sharpe']:.2f} | Alpha: {row['Alpha']:.1%}", color='white', fontsize=9)
        y_pos -= 0.13

    ax3 = fig.add_subplot(gs[1, :2])
    ax3.set_facecolor('#161b22')
    cum_port = (1 + test_port).cumprod() - 1
    ax3.plot(cum_port.index, cum_port.values * 100, color='#00ff88', label='Robust Portfolio (TEST)', linewidth=2)
    if test_bench is not None:
        cum_bench = (1 + test_bench.reindex(test_port.index).dropna()).cumprod() - 1
        ax3.plot(cum_bench.index, cum_bench.values * 100, color='#ff6b35', label='S&P 500 (SPY)', linestyle='--', linewidth=1.5)
    ax3.set_title('Out-of-Sample Performance (Test Period)', color='white', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Cumulative Return (%)', color='white')
    ax3.tick_params(colors='white')
    ax3.legend(facecolor='#161b22', labelcolor='white')
    ax3.grid(True, alpha=0.15, color='white')

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor('#161b22')
    x_years = np.arange(len(dca_p50)) / 12
    ax4.plot(x_years, dca_p50 / 1000, color='#00d4ff', label='Median (P50)', linewidth=2)
    ax4.fill_between(x_years, dca_p10 / 1000, dca_p90 / 1000, color='#00d4ff', alpha=0.15, label='P10-P90 Range')
    ax4.axhline(total_contrib / 1000, color='white', linestyle=':', label='Total Contributed')
    ax4.set_title(f'{PROJECTION_YEARS}-Year DCA Projection (€100/mo)', color='white', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Capital (€ Thousands)', color='white')
    ax4.tick_params(colors='white')
    ax4.legend(facecolor='#161b22', labelcolor='white', fontsize=8)
    ax4.grid(True, alpha=0.15, color='white')

    ax5 = fig.add_subplot(gs[2, :])
    ax5.set_facecolor('#161b22')
    cax = ax5.matshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax, ax=ax5, fraction=0.02, pad=0.04)
    ax5.set_xticks(range(len(corr_matrix.columns)))
    ax5.set_yticks(range(len(corr_matrix.columns)))
    ax5.set_xticklabels(corr_matrix.columns, color='white', rotation=45, fontsize=8)
    ax5.set_yticklabels(corr_matrix.columns, color='white', fontsize=8)
    ax5.set_title('Asset Cross-Correlation Matrix (Weekly)', color='white', fontsize=11, fontweight='bold', pad=12)

    plt.suptitle("Master Quantitative Portfolio Dashboard", color='white', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig("Master_Portfolio_Dashboard.png", dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.close()

    if HAS_QUANTSTATS:
        try:
            qs.reports.html(test_port, benchmark=test_bench, output='Portfolio_Master_vs_SPY.html', title='Master Portfolio vs SPY')
        except Exception as e:
            pass


def main():
    prices = download_market_data(PORTFOLIO_TICKERS, benchmark=BENCHMARK_TICKER, years=LOOKBACK_YEARS)
    df_metrics = calculate_asset_metrics(prices, benchmark=BENCHMARK_TICKER, rf_rate=RISK_FREE_RATE)
    df_weights, corr_matrix, ret_bl, S_lw = optimize_portfolio(
        prices, SUBJECTIVE_VIEWS, SECTORS, SECTOR_LIMITS, benchmark=BENCHMARK_TICKER, n_sims=500
    )
    best_weights_dict = df_weights['Markowitz_Robusto_MC'].to_dict()
    test_port, test_bench, start_test_date = run_backtest(prices, best_weights_dict, benchmark=BENCHMARK_TICKER, train_ratio=TRAIN_RATIO)
    dca_p50, dca_p10, dca_p90, total_contrib = simulate_dca(test_port, initial=INITIAL_CAPITAL, monthly=MONTHLY_CONTRIBUTION, years=PROJECTION_YEARS)
    generate_reports(df_metrics, corr_matrix, df_weights, test_port, test_bench, dca_p50, dca_p10, dca_p90, total_contrib)

if __name__ == "__main__":
    main()