import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
from sqlalchemy import create_engine

FMP_API_KEY = 'Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT'
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'Naura')
DB_USER = os.getenv('DB_USER', 'nauraai')
DB_PASSWORD = os.getenv('DB_PASSWORD', '')

ENGINE = create_engine(f'postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

CURRENT_DATE = pd.to_datetime("2025-09-20")

def fetch_peers(symbol):
    query = """
    SELECT DISTINCT peer_symbol
    FROM etfs_peers
    WHERE symbol = %s
    """
    df = pd.read_sql(query, ENGINE, params=(symbol,))
    return df['peer_symbol'].tolist()

def fetch_historical(symbol, table='etfs_quotes'):
    query = f"""
    SELECT date, close
    FROM {table}
    WHERE symbol = %s
    ORDER BY date ASC
    """
    df = pd.read_sql(query, ENGINE, params=(symbol,), parse_dates=['date'])
    if df.empty:
        raise ValueError(f"No data for {symbol}")
    return symbol, df.set_index('date')['close'].sort_index()

def fetch_treasury():
    query = """
    SELECT date, month3
    FROM treasury
    WHERE date BETWEEN '2010-01-01' AND '2025-09-20'
    ORDER BY date ASC
    """
    df = pd.read_sql(query, ENGINE, parse_dates=['date'])
    if df.empty:
        raise ValueError("No treasury data")
    return df.set_index('date')['month3'].sort_index() / 100 / 252

def fetch_expense(symbol):
    query = """
    SELECT "expenseRatio"
    FROM etfs_profile
    WHERE symbol = %s
    """
    df = pd.read_sql(query, ENGINE, params=(symbol,))
    if df.empty:
        raise ValueError(f"No info for {symbol}")
    return df['expenseRatio'].iloc[0]

def compute_metrics(symbol, df_fund, df_bench, df_rf):
    three_yr_ago = CURRENT_DATE - pd.Timedelta(days=3*365 + 1)
    five_yr_ago = CURRENT_DATE - pd.Timedelta(days=5*365 + 1)
    df_fund_5yr = df_fund[five_yr_ago:CURRENT_DATE]
    if len(df_fund_5yr) < 900:
        return None
    days_5 = (df_fund_5yr.index[-1] - df_fund_5yr.index[0]).days
    cagr5 = (df_fund_5yr.iloc[-1] / df_fund_5yr.iloc[0]) ** (365 / days_5) - 1
    df_fund_3yr = df_fund[three_yr_ago:CURRENT_DATE]
    days_3 = (df_fund_3yr.index[-1] - df_fund_3yr.index[0]).days
    cagr3 = (df_fund_3yr.iloc[-1] / df_fund_3yr.iloc[0]) ** (365 / days_3) - 1
    ret_fund_d = df_fund_3yr.pct_change().dropna()
    ret_bench_d = df_bench[ret_fund_d.index].pct_change().dropna()
    ret_fund_d, ret_bench_d = ret_fund_d.align(ret_bench_d, join='inner')
    rf_series = df_rf.asof(ret_fund_d.index)
    rf_d = rf_series.mean()
    std = ret_fund_d.std() * np.sqrt(252)
    slope, _, _, _, _ = linregress(ret_bench_d, ret_fund_d)
    beta = slope
    excess = ret_fund_d - rf_d
    sharpe = excess.mean() / ret_fund_d.std() * np.sqrt(252)
    excess_mkt = ret_bench_d - rf_d
    alpha = ((ret_fund_d.mean() - rf_d) - beta * excess_mkt.mean()) * 252
    df_fund_m = df_fund.resample('ME').last()
    ret_fund_m = df_fund_m.pct_change().dropna()
    df_bench_m = df_bench.resample('ME').last()
    ret_bench_m = df_bench_m.pct_change().dropna()
    ret_fund_m, ret_bench_m = ret_fund_m.align(ret_bench_m, join='inner')
    ret_fund_m3 = ret_fund_m[-36:]
    ret_bench_m3 = ret_bench_m[-36:]
    batting = (ret_fund_m3 > ret_bench_m3).mean() * 100
    up = ret_bench_m3 > 0
    upside = (ret_fund_m3[up].mean() / ret_bench_m3[up].mean() * 100) if up.any() else 0
    down = ret_bench_m3 < 0
    downside = (ret_fund_m3[down].mean() / ret_bench_m3[down].mean() * 100) if down.any() else 0
    expense = fetch_expense(symbol)
    return {
        'symbol': symbol, '5yr_return': cagr5, '3yr_return': cagr3, 'batting_avg': batting,
        'std_dev': std, 'beta': beta, 'sharpe': sharpe, 'alpha': alpha,
        'mer': expense, 'upside': upside, 'downside': downside, 'ter': expense
    }

def calculate_scores(metrics_list):
    if not metrics_list:
        return pd.DataFrame()
    df = pd.DataFrame(metrics_list).set_index('symbol')
    weights = {
        '5yr_return': 12, '3yr_return': 20, 'batting_avg': 8,
        'std_dev': 7.5, 'beta': 7.5, 'sharpe': 7.5, 'alpha': 7.5,
        'mer': 15, 'upside': 5, 'downside': 5, 'ter': 5
    }
    direction = {
        '5yr_return': True, '3yr_return': True, 'batting_avg': True,
        'std_dev': False, 'beta': False, 'sharpe': True, 'alpha': True,
        'mer': False, 'upside': True, 'downside': False, 'ter': False
    }
    for col, w in weights.items():
        vals = df[col].dropna()
        if len(vals) < 2: continue
        minv, maxv = vals.min(), vals.max()
        if minv == maxv:
            norm = pd.Series(0.5, index=df.index)
        else:
            norm = (df[col] - minv) / (maxv - minv)
            if not direction[col]: norm = 1 - norm
        df[f'score_{col}'] = norm * w
    perf_metrics = ['5yr_return', '3yr_return', 'batting_avg']
    risk_metrics = ['std_dev', 'beta', 'sharpe', 'alpha']
    mgmt_metrics = ['mer', 'upside', 'downside', 'ter']
    df['performance'] = df[[f'score_{m}' for m in perf_metrics]].sum(axis=1) / 40 * 100
    df['risk'] = df[[f'score_{m}' for m in risk_metrics]].sum(axis=1) / 30 * 100
    df['management'] = df[[f'score_{m}' for m in mgmt_metrics]].sum(axis=1) / 30 * 100
    df['total_score'] = df.filter(like='score_').sum(axis=1)
    for col in ['performance', 'risk', 'management', 'total_score']:
        minv, maxv = df[col].min(), df[col].max()
        if maxv > minv:
            df[col] = (df[col] - minv) / (maxv - minv) * 100
        else:
            df[col] = 50.0
    result_cols = ['performance', 'risk', 'management', 'total_score']
    return df[result_cols].sort_values('total_score', ascending=False)

def main(benchmark='SPY'):
    input_symbol = input("Enter symbol: ").strip().upper()
    peers = fetch_peers(input_symbol)
    symbols = list(set([input_symbol] + peers))
    df_rf = fetch_treasury()
    _, df_bench = fetch_historical(benchmark, table='market_and_sector_quotes')
    funds = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(fetch_historical, s) for s in symbols]
        for f in tqdm(as_completed(futures), total=len(symbols), desc="Fetching fund data"):
            try:
                s, d = f.result()
                funds[s] = d
            except Exception as e:
                print(e)
    metrics = []
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = [ex.submit(compute_metrics, s, funds[s], df_bench, df_rf) for s in symbols if s in funds]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Computing metrics"):
            m = f.result()
            if m: metrics.append(m)
    return calculate_scores(metrics)

if __name__ == "__main__":
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.float_format', '{:.2f}'.format)
    result = main()
    print(result.to_string())