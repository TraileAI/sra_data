import psycopg2
import pandas as pd
import numpy as np
from datetime import datetime
from dateutil.relativedelta import relativedelta
import calendar
from collections import defaultdict
from tqdm import tqdm
from pandas.tseries.offsets import DateOffset
import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration from environment variables
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def get_connection_string():
    """Get database connection string from environment variables."""
    return f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_symbols():
    conn_str = get_connection_string()
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM equity_profile")
    symbols = [row[0] for row in cur.fetchall()]
    cur.close()
    conn.close()
    return symbols


def get_peers(symbols):
    if not symbols:
        return {}
    conn_str = get_connection_string()
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    chunk_size = 1000
    all_peers = []
    for i in range(0, len(symbols), chunk_size):
        chunk = symbols[i:i + chunk_size]
        symbols_tuple = tuple(chunk)
        cur.execute("SELECT symbol, peer_symbol FROM equity_peers WHERE symbol IN %s", (symbols_tuple,))
        rows = cur.fetchall()
        all_peers.extend(rows)
    cur.close()
    conn.close()
    peers_dict = defaultdict(list)
    for row in all_peers:
        peers_dict[row[0]].append(row[1])
    return dict(peers_dict)


def generate_dates(start_date, num_months=12):
    dates = []
    date = start_date
    for _ in range(num_months):
        dates.append(date)
        date = date - relativedelta(months=1)
        days_in_month = calendar.monthrange(date.year, date.month)[1]
        date = date.replace(day=days_in_month)
    return sorted(dates)


def get_raw_growth(sym, fg_dict, fr_dict, cutoff_date):
    fg = fg_dict.get(sym, pd.DataFrame(columns=['date']))
    fg = fg[fg['date'] <= cutoff_date].sort_values('date')
    fg_latest = fg.iloc[-1] if not fg.empty else pd.Series()
    rev_growth_val = fg_latest.get('revenueGrowth', np.nan)
    rev_growth = rev_growth_val * 100 if not pd.isna(rev_growth_val) else np.nan
    fr = fr_dict.get(sym, pd.DataFrame(columns=['date']))
    fr = fr[fr['date'] <= cutoff_date].sort_values('date')
    fr_latest = fr.iloc[-1] if not fr.empty else pd.Series()
    peg_val = fr_latest.get('priceEarningsToGrowthRatio', np.nan)
    peg = peg_val if not pd.isna(peg_val) else np.nan
    if pd.isna(rev_growth) and pd.isna(peg):
        return np.nan
    return np.nanmean([rev_growth, -peg])


def get_raw_valuation(sym, fr_dict, cutoff_date):
    fr = fr_dict.get(sym, pd.DataFrame(columns=['date']))
    fr = fr[fr['date'] <= cutoff_date].sort_values('date')
    if fr.empty:
        return np.nan
    fr_latest = fr.iloc[-1]
    pe = fr_latest.get('priceEarningsRatio', np.nan)
    pb = fr_latest.get('priceToBookRatio', np.nan)
    peg = fr_latest.get('priceEarningsToGrowthRatio', np.nan)
    values = [-v for v in [pe, pb, peg] if not np.isnan(v)]
    return np.nanmean(values) if values else np.nan


def get_raw_strength(sym, fr_dict, cutoff_date):
    fr = fr_dict.get(sym, pd.DataFrame(columns=['date']))
    fr = fr[fr['date'] <= cutoff_date].sort_values('date')
    if fr.empty:
        return np.nan
    fr_latest = fr.iloc[-1]
    d2e = fr_latest.get('debtEquityRatio', np.nan)
    current = fr_latest.get('currentRatio', np.nan)
    interest = fr_latest.get('interestCoverage', np.nan)
    values = [-d2e] + [v for v in [current, interest] if not np.isnan(v)]
    return np.nanmean(values) if values else np.nan


def get_raw_profitability(sym, km_dict, fr_dict, cutoff_date):
    km = km_dict.get(sym, pd.DataFrame(columns=['date']))
    km = km[km['date'] <= cutoff_date].sort_values('date')
    if km.empty:
        return np.nan
    km_latest = km.iloc[-1]
    roe = km_latest.get('roe', np.nan) * 100
    fr = fr_dict.get(sym, pd.DataFrame(columns=['date']))
    fr = fr[fr['date'] <= cutoff_date].sort_values('date')
    if fr.empty:
        return np.nan
    fr_latest = fr.iloc[-1]
    npm = fr_latest.get('netProfitMargin', np.nan) * 100
    opm = fr_latest.get('operatingProfitMargin', np.nan) * 100
    values = [v for v in [roe, npm, opm] if not np.isnan(v)]
    return np.mean(values) if values else np.nan


def get_raw_performance(sym, q_dict, cutoff_date):
    sym_data = q_dict.get(sym, pd.DataFrame(columns=['date']))
    current_data = sym_data[sym_data['date'] <= cutoff_date]
    if current_data.empty:
        return np.nan
    current_close = current_data.loc[current_data['date'].idxmax(), 'adjClose']
    if np.isnan(current_close):
        return np.nan
    rets = []
    past_date = cutoff_date - DateOffset(months=1)
    past_data = sym_data[sym_data['date'] <= past_date]
    if not past_data.empty:
        past_close = past_data.loc[past_data['date'].idxmax(), 'adjClose']
        if not np.isnan(past_close) and past_close != 0:
            rets.append((current_close - past_close) / past_close * 100)
    past_date = datetime(cutoff_date.year, 1, 1)
    past_data = sym_data[sym_data['date'] < past_date]
    if not past_data.empty:
        past_close = past_data.loc[past_data['date'].idxmax(), 'adjClose']
        if not np.isnan(past_close) and past_close != 0:
            rets.append((current_close - past_close) / past_close * 100)
    past_date = cutoff_date - DateOffset(years=1)
    past_data = sym_data[sym_data['date'] <= past_date]
    if not past_data.empty:
        past_close = past_data.loc[past_data['date'].idxmax(), 'adjClose']
        if not np.isnan(past_close) and past_close != 0:
            rets.append((current_close - past_close) / past_close * 100)
    past_date = cutoff_date - DateOffset(years=3)
    past_data = sym_data[sym_data['date'] <= past_date]
    if not past_data.empty:
        past_close = past_data.loc[past_data['date'].idxmax(), 'adjClose']
        if not np.isnan(past_close) and past_close != 0:
            rets.append((current_close - past_close) / past_close * 100)
    past_date = cutoff_date - DateOffset(years=5)
    past_data = sym_data[sym_data['date'] <= past_date]
    if not past_data.empty:
        past_close = past_data.loc[past_data['date'].idxmax(), 'adjClose']
        if not np.isnan(past_close) and past_close != 0:
            rets.append((current_close - past_close) / past_close * 100)
    return np.mean(rets) if rets else np.nan


def normalize_scores(raw_scores, symbols):
    valid_raw = [raw_scores[s] for s in symbols if not np.isnan(raw_scores[s])]
    if not valid_raw:
        return {s: np.nan for s in symbols}
    min_s = np.min(valid_raw)
    max_s = np.max(valid_raw)
    range_s = max_s - min_s
    if range_s == 0:
        return {s: 50.0 if not np.isnan(raw_scores[s]) else np.nan for s in symbols}
    norm = {}
    for s in symbols:
        if np.isnan(raw_scores[s]):
            norm[s] = np.nan
        else:
            norm[s] = np.clip((raw_scores[s] - min_s) / range_s * 100, 0, 100)
    return norm


def compute_all_norms_for_date(cutoff_date, test_symbols, fg_dict, fr_dict, km_dict, q_dict):
    raw_growth = {s: get_raw_growth(s, fg_dict, fr_dict, cutoff_date) for s in test_symbols}
    norm_growth = normalize_scores(raw_growth, test_symbols)
    raw_val = {s: get_raw_valuation(s, fr_dict, cutoff_date) for s in test_symbols}
    norm_val = normalize_scores(raw_val, test_symbols)
    raw_str = {s: get_raw_strength(s, fr_dict, cutoff_date) for s in test_symbols}
    norm_str = normalize_scores(raw_str, test_symbols)
    raw_prof = {s: get_raw_profitability(s, km_dict, fr_dict, cutoff_date) for s in test_symbols}
    norm_prof = normalize_scores(raw_prof, test_symbols)
    raw_perf = {s: get_raw_performance(s, q_dict, cutoff_date) for s in test_symbols}
    norm_perf = normalize_scores(raw_perf, test_symbols)
    raw_buckler = {}
    for s in test_symbols:
        vals = [norm_growth.get(s, np.nan), norm_val.get(s, np.nan), norm_str.get(s, np.nan), norm_prof.get(s, np.nan),
                norm_perf.get(s, np.nan)]
        valid = [v for v in vals if not np.isnan(v)]
        raw_buckler[s] = np.mean(valid) if valid else np.nan
    norm_buckler = normalize_scores(raw_buckler, test_symbols)
    return {
        'growth': norm_growth,
        'valuation': norm_val,
        'financial_strength': norm_str,
        'profitability': norm_prof,
        'performance': norm_perf,
        'buckler': norm_buckler
    }


if __name__ == "__main__":
    TEST = False  # Set to False for full symbols list
    cutoff_date = datetime(2025, 10, 1)
    if TEST:
        symbols = ['AAPL']
    else:
        symbols = get_symbols()
    peers_dict = get_peers(symbols)
    test_symbols = list(set(symbols + [p for lst in peers_dict.values() for p in lst if lst]))
    if not test_symbols:
        print("No symbols")
        exit()
    conn_str = get_connection_string()

    chunk_size = 1000
    # Fetch financial growth
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    all_fg_rows = []
    for i in range(0, len(test_symbols), chunk_size):
        chunk = test_symbols[i:i + chunk_size]
        symbols_tuple = tuple(chunk)
        cur.execute("SELECT symbol, date, revenueGrowth FROM equity_financial_growth WHERE symbol IN %s",
                    (symbols_tuple,))
        rows = cur.fetchall()
        all_fg_rows.extend(rows)
    cur.close()
    conn.close()
    df_fg = pd.DataFrame(all_fg_rows, columns=['symbol', 'date', 'revenueGrowth'])
    df_fg['date'] = pd.to_datetime(df_fg['date'])
    df_fg.replace({None: np.nan}, inplace=True)
    fg_groups = df_fg.groupby('symbol')
    fg_dict = {}
    for s in test_symbols:
        try:
            fg_dict[s] = fg_groups.get_group(s)
        except KeyError:
            fg_dict[s] = pd.DataFrame(columns=['date', 'revenueGrowth'])

    # Fetch key metrics
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    all_km_rows = []
    for i in range(0, len(test_symbols), chunk_size):
        chunk = test_symbols[i:i + chunk_size]
        symbols_tuple = tuple(chunk)
        cur.execute("SELECT symbol, date, roe FROM equity_key_metrics WHERE symbol IN %s", (symbols_tuple,))
        rows = cur.fetchall()
        all_km_rows.extend(rows)
    cur.close()
    conn.close()
    df_km = pd.DataFrame(all_km_rows, columns=['symbol', 'date', 'roe'])
    df_km['date'] = pd.to_datetime(df_km['date'])
    df_km.replace({None: np.nan}, inplace=True)
    km_groups = df_km.groupby('symbol')
    km_dict = {}
    for s in test_symbols:
        try:
            km_dict[s] = km_groups.get_group(s)
        except KeyError:
            km_dict[s] = pd.DataFrame(columns=['date', 'roe'])

    # Fetch financial ratio
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    all_fr_rows = []
    for i in range(0, len(test_symbols), chunk_size):
        chunk = test_symbols[i:i + chunk_size]
        symbols_tuple = tuple(chunk)
        cur.execute(
            "SELECT symbol, date, debtEquityRatio, currentRatio, interestCoverage, priceEarningsRatio, priceToBookRatio, priceEarningsToGrowthRatio, netProfitMargin, operatingProfitMargin FROM equity_financial_ratio WHERE symbol IN %s",
            (symbols_tuple,))
        rows = cur.fetchall()
        all_fr_rows.extend(rows)
    cur.close()
    conn.close()
    df_fr = pd.DataFrame(all_fr_rows, columns=['symbol', 'date', 'debtEquityRatio', 'currentRatio', 'interestCoverage',
                                               'priceEarningsRatio', 'priceToBookRatio', 'priceEarningsToGrowthRatio',
                                               'netProfitMargin', 'operatingProfitMargin'])
    df_fr['date'] = pd.to_datetime(df_fr['date'])
    df_fr.replace({None: np.nan}, inplace=True)
    fr_groups = df_fr.groupby('symbol')
    fr_dict = {}
    for s in test_symbols:
        try:
            fr_dict[s] = fr_groups.get_group(s)
        except KeyError:
            fr_dict[s] = pd.DataFrame(
                columns=['date', 'debtEquityRatio', 'currentRatio', 'interestCoverage', 'priceEarningsRatio',
                         'priceToBookRatio', 'priceEarningsToGrowthRatio', 'netProfitMargin', 'operatingProfitMargin'])

    # Fetch quotes
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    all_q_rows = []
    for i in range(0, len(test_symbols), chunk_size):
        chunk = test_symbols[i:i + chunk_size]
        symbols_tuple = tuple(chunk)
        cur.execute("SELECT symbol, date, adjClose FROM equity_quotes WHERE symbol IN %s", (symbols_tuple,))
        rows = cur.fetchall()
        all_q_rows.extend(rows)
    cur.close()
    conn.close()
    df_q = pd.DataFrame(all_q_rows, columns=['symbol', 'date', 'adjClose'])
    df_q['date'] = pd.to_datetime(df_q['date'])
    df_q.replace({None: np.nan}, inplace=True)
    q_groups = df_q.groupby('symbol')
    q_dict = {}
    for s in test_symbols:
        try:
            q_dict[s] = q_groups.get_group(s)
        except KeyError:
            q_dict[s] = pd.DataFrame(columns=['date', 'adjClose'])

    dates = generate_dates(cutoff_date)
    scores_data = []
    score_types = ['growth', 'valuation', 'financial_strength', 'profitability', 'performance', 'buckler']
    for date in tqdm(dates, desc="Processing dates"):
        all_norms = compute_all_norms_for_date(date, test_symbols, fg_dict, fr_dict, km_dict, q_dict)
        for score_type in score_types:
            norm_scores = all_norms[score_type]
            cat_scores = {s: norm_scores.get(s, np.nan) for s in test_symbols}
            df_cat = pd.DataFrame({'symbol': test_symbols, 'score': list(cat_scores.values())})
            df_cat = df_cat.sort_values('score', ascending=False, na_position='last').reset_index(drop=True)
            df_cat['rank'] = range(1, len(df_cat) + 1)
            for _, row in df_cat.iterrows():
                if row['symbol'] in symbols:
                    score_val = round(row['score'], 2) if not pd.isna(row['score']) else None
                    scores_data.append({
                        'symbol': row['symbol'],
                        'date': date,
                        'score_type': score_type,
                        'score': score_val,
                        'rank': int(row['rank'])
                    })
    conn = psycopg2.connect(conn_str)
    cur = conn.cursor()
    cur.execute("DROP TABLE IF EXISTS equity_scores")
    cur.execute("""
                CREATE TABLE equity_scores
                (
                    symbol     VARCHAR(20) NOT NULL,
                    date       DATE        NOT NULL,
                    score_type VARCHAR(50) NOT NULL,
                    score      NUMERIC(5, 2),
                    rank       INTEGER     NOT NULL
                )
                """)
    conn.commit()
    insert_query = "INSERT INTO equity_scores (symbol, date, score_type, score, rank) VALUES (%s, %s, %s, %s, %s)"
    data_to_insert = [(row['symbol'], row['date'].date(), row['score_type'], row['score'], row['rank']) for row in
                      scores_data]
    batch_size = 1000
    for i in range(0, len(data_to_insert), batch_size):
        batch = data_to_insert[i:i + batch_size]
        cur.executemany(insert_query, batch)
        conn.commit()
    cur.close()
    conn.close()
    print(
        f"Saved {len(scores_data)} score records for {len(dates)} dates and {len(score_types)} types to equity_scores.")