import pandas as pd
import numpy as np
from datetime import datetime
from scipy.stats import linregress
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import os
import sys
from sqlalchemy import create_engine, text, Table, MetaData
from sqlalchemy.dialects.postgresql import insert

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

FMP_API_KEY = 'Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT'

ENGINE = create_engine(config.get_connection_string())

CURRENT_DATE = pd.to_datetime("2025-09-20")

def fetch_peers(symbol):
    query = """
    SELECT DISTINCT peer_symbol
    FROM etfs_peers
    WHERE symbol = %s AND peer_symbol IS NOT NULL
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
        return symbol, pd.Series(dtype=float)
    return symbol, df.set_index('date')['close'].sort_index()

def fetch_treasury():
    query = """
    SELECT date, month3
    FROM treasury
    WHERE date >= '2010-01-01'
    ORDER BY date ASC
    """
    df = pd.read_sql(query, ENGINE, parse_dates=['date'])
    if df.empty:
        raise ValueError("No treasury data")
    return df.set_index('date')['month3'].sort_index() / 100 / 252

def fetch_expense(symbol):
    query = """
    SELECT expenseratio
    FROM etfs_profile
    WHERE symbol = %s
    """
    df = pd.read_sql(query, ENGINE, params=(symbol,))
    if df.empty:
        return np.nan
    return df['expenseratio'].iloc[0]

def compute_metrics(symbol, df_fund, df_bench, df_rf, current_date):
    if df_fund is None or df_fund.empty:
        return None
    five_yr_ago = current_date - pd.Timedelta(days=5*365 + 1)
    three_yr_ago = current_date - pd.Timedelta(days=3*365 + 1)
    df_fund_slice = df_fund[:current_date]
    if df_fund_slice.empty:
        return None
    df_fund_5yr = df_fund_slice[five_yr_ago:]
    if len(df_fund_5yr) < 900:
        return None
    days_5 = (df_fund_5yr.index[-1] - df_fund_5yr.index[0]).days
    cagr5 = (df_fund_5yr.iloc[-1] / df_fund_5yr.iloc[0]) ** (365 / days_5) - 1
    df_fund_3yr = df_fund_slice[three_yr_ago:]
    if len(df_fund_3yr) < 252:
        return None
    days_3 = (df_fund_3yr.index[-1] - df_fund_3yr.index[0]).days
    cagr3 = (df_fund_3yr.iloc[-1] / df_fund_3yr.iloc[0]) ** (365 / days_3) - 1
    ret_fund_d = df_fund_3yr.pct_change().dropna()
    df_bench_slice = df_bench[:current_date]
    ret_bench_d = df_bench_slice.pct_change().dropna()
    ret_fund_d, ret_bench_d = ret_fund_d.align(ret_bench_d, join='inner')
    if len(ret_fund_d) < 2:
        return None
    df_rf_up_to = df_rf[:current_date]
    rf_series = df_rf_up_to.asof(ret_fund_d.index)
    rf_d = rf_series.mean()
    std = ret_fund_d.std() * np.sqrt(252)
    slope, _, _, _, _ = linregress(ret_bench_d, ret_fund_d)
    beta = slope
    excess = ret_fund_d - rf_d
    fund_std = ret_fund_d.std()
    sharpe = excess.mean() / fund_std * np.sqrt(252) if fund_std > 0 else 0.0
    excess_mkt = ret_bench_d - rf_d
    alpha = ((ret_fund_d.mean() - rf_d) - beta * excess_mkt.mean()) * 252
    df_fund_m = df_fund_slice.resample('ME').last()
    ret_fund_m = df_fund_m.pct_change().dropna()
    df_bench_m = df_bench_slice.resample('ME').last()
    ret_bench_m = df_bench_m.pct_change().dropna()
    ret_fund_m, ret_bench_m = ret_fund_m.align(ret_bench_m, join='inner')
    if len(ret_fund_m) < 12:
        return None
    ret_fund_m3 = ret_fund_m.tail(36)
    ret_bench_m3 = ret_bench_m.tail(36)
    batting = (ret_fund_m3 > ret_bench_m3).mean() * 100
    up = ret_bench_m3 > 0
    upside = (ret_fund_m3[up].mean() / ret_bench_m3[up].mean() * 100) if up.any() else 50.0
    down = ret_bench_m3 < 0
    downside = (ret_fund_m3[down].mean() / ret_bench_m3[down].mean() * 100) if down.any() else 50.0
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
        if col not in df.columns:
            df[f'score_{col}'] = 0
            continue
        vals = df[col].dropna()
        if len(vals) < 2:
            norm = pd.Series(0.5, index=df.index)
        else:
            minv, maxv = vals.min(), vals.max()
            if pd.isna(minv) or pd.isna(maxv) or minv == maxv:
                norm = pd.Series(0.5, index=df.index)
            else:
                norm = (df[col] - minv) / (maxv - minv)
                if not direction[col]:
                    norm = 1 - norm
        df[f'score_{col}'] = norm.fillna(0.5) * w
    perf_metrics = ['5yr_return', '3yr_return', 'batting_avg']
    risk_metrics = ['std_dev', 'beta', 'sharpe', 'alpha']
    mgmt_metrics = ['mer', 'upside', 'downside', 'ter']
    df['performance'] = df[[f'score_{m}' for m in perf_metrics]].sum(axis=1) / 40 * 100
    df['risk'] = df[[f'score_{m}' for m in risk_metrics]].sum(axis=1) / 30 * 100
    df['management'] = df[[f'score_{m}' for m in mgmt_metrics]].sum(axis=1) / 30 * 100
    df['total_score'] = df.filter(like='score_').sum(axis=1)
    for col in ['performance', 'risk', 'management', 'total_score']:
        minv, maxv = df[col].min(), df[col].max()
        if pd.isna(minv) or pd.isna(maxv) or maxv <= minv:
            df[col] = 50.0
        else:
            df[col] = (df[col] - minv) / (maxv - minv) * 100
    result_cols = ['performance', 'risk', 'management', 'total_score']
    return df[result_cols]

def upsert_scores(scores_df, engine):
    metadata = MetaData()
    table = Table('etfs_scores', metadata, autoload_with=engine)
    with engine.connect() as conn:
        ins = insert(table).values(scores_df.to_dict('records'))
        stmt = ins.on_conflict_do_update(
            index_elements=['symbol', 'date'],
            set_={col: ins.excluded[col] for col in ['performance', 'risk', 'management', 'total_score']}
        )
        conn.execute(stmt)
        conn.commit()

def main(benchmark='SPY'):
    sql = """
        CREATE TABLE IF NOT EXISTS etfs_scores (
            symbol VARCHAR(10),
            date DATE,
            performance FLOAT,
            risk FLOAT,
            management FLOAT,
            total_score FLOAT,
            PRIMARY KEY (symbol, date)
        )
    """
    with ENGINE.connect() as conn:
        conn.execute(text(sql))
        conn.commit()
    symbols = pd.read_sql("SELECT symbol FROM etfs_profile WHERE symbol IS NOT NULL", ENGINE)['symbol'].tolist()
    print(f"Processing {len(symbols)} symbols")
    df_rf = fetch_treasury()
    _, df_bench = fetch_historical(benchmark, table='market_and_sector_quotes')
    funds = {}
    with ThreadPoolExecutor(max_workers=20) as ex:
        futures = [ex.submit(fetch_historical, s) for s in symbols]
        for f in tqdm(as_completed(futures), total=len(symbols), desc="Fetching fund data"):
            try:
                s, d = f.result()
                funds[s] = d
            except Exception as e:
                print(f"Error fetching {s}: {e}")
    dates = pd.date_range(end=CURRENT_DATE, periods=12, freq='ME')
    for current_date in tqdm(dates, desc="Computing historical scores"):
        scores_this_date = []
        for symbol in tqdm(symbols, desc=f"Symbols for {current_date.strftime('%Y-%m')}", leave=False):
            peers = fetch_peers(symbol)
            group_symbols = list(set([symbol] + peers))
            metrics_group = []
            with ThreadPoolExecutor(max_workers=10) as ex:
                futures = [ex.submit(compute_metrics, s, funds.get(s), df_bench, df_rf, current_date) for s in group_symbols]
                for future in as_completed(futures):
                    m = future.result()
                    if m:
                        metrics_group.append(m)
            if metrics_group and symbol in [m['symbol'] for m in metrics_group]:
                scores_group = calculate_scores(metrics_group)
                if symbol in scores_group.index:
                    score_row = scores_group.loc[symbol]
                    scores_this_date.append({
                        'symbol': symbol,
                        'performance': score_row['performance'],
                        'risk': score_row['risk'],
                        'management': score_row['management'],
                        'total_score': score_row['total_score']
                    })
        if scores_this_date:
            scores_df = pd.DataFrame(scores_this_date)
            scores_df['date'] = current_date.date()
            scores_df = scores_df[['symbol', 'date', 'performance', 'risk', 'management', 'total_score']]
            upsert_scores(scores_df, ENGINE)

if __name__ == "__main__":
    main()