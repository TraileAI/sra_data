import os
import sys
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import bisect
import calendar
import threading

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config

load_dotenv()

def connect_db():
    return psycopg2.connect(**config.db_config)

def create_scores_table():
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS equity_scores (
            reference_symbol VARCHAR(20) NOT NULL,
            symbol VARCHAR(20) NOT NULL,
            score_date DATE NOT NULL,
            category VARCHAR(50) NOT NULL,
            score NUMERIC NOT NULL,
            rank INTEGER NOT NULL,
            CONSTRAINT equity_scores_pkey PRIMARY KEY (reference_symbol, symbol, score_date, category)
        );
    """)
    conn.commit()
    cur.close()
    conn.close()

def bulk_save_scores(inserts):
    if not inserts:
        return
    conn = connect_db()
    cur = conn.cursor()
    psycopg2.extras.execute_values(
        cur,
        """
        INSERT INTO equity_scores (reference_symbol, symbol, score_date, category, score, rank)
        VALUES %s
        ON CONFLICT (reference_symbol, symbol, score_date, category) DO UPDATE SET
            score = EXCLUDED.score,
            rank = EXCLUDED.rank
        """,
        inserts
    )
    conn.commit()
    cur.close()
    conn.close()

def fetch_symbol_data(symbol):
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT date::date, returnonequity, netprofitmargin, operatingprofitmargin, debtequityratio,
                   currentratio, interestcoverage, priceearningsratio, pricetobookratio, priceearningstogrowthratio
            FROM equity_financial_ratio WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        rows = cur.fetchall()
        ratios = [(row[0], {
            'returnOnEquity': row[1], 'netProfitMargin': row[2], 'operatingProfitMargin': row[3],
            'debtEquityRatio': row[4], 'currentRatio': row[5], 'interestCoverage': row[6],
            'priceEarningsRatio': row[7], 'priceToBookRatio': row[8], 'priceEarningsToGrowthRatio': row[9]
        }) for row in rows]

        cur.execute("""
            SELECT date::date, threeyrevenuegrowthpershare, threeynetincomegrowthpershare
            FROM equity_financial_growth WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        rows = cur.fetchall()
        growth = [(row[0], {
            'threeYRevenueGrowthPerShare': row[1], 'threeYNetIncomeGrowthPerShare': row[2]
        }) for row in rows]

        cur.execute("""
            SELECT date::date, adjclose FROM equity_quotes WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        quotes = cur.fetchall()

        cur.close()
        conn.close()
        return {'ratios': ratios, 'growth': growth, 'quotes': quotes}
    except Exception:
        cur.close()
        conn.close()
        raise

def calculate_returns(quotes, as_of_date):
    if not quotes:
        return {}
    dates = [q[0] for q in quotes]
    prices = [q[1] for q in quotes]
    idx = bisect.bisect_right(dates, as_of_date) - 1
    if idx < 0:
        return {}
    current_price = prices[idx]
    if not current_price:
        return {}
    periods = {
        '1M_return': as_of_date - timedelta(days=30),
        'YTD_return': date(as_of_date.year, 1, 1),
        '1Y_return': as_of_date - timedelta(days=365),
        '3Y_return': as_of_date - timedelta(days=1095),
        '5Y_return': as_of_date - timedelta(days=1825),
    }
    returns = {}
    for key, past_date in periods.items():
        p_idx = bisect.bisect_right(dates, past_date) - 1
        if p_idx >= 0:
            past_price = prices[p_idx]
            if past_price and past_price != 0:
                ret = (current_price - past_price) / past_price * 100
                returns[key] = ret
    return returns

def compute_scores_for_date(hist_date, data, symbols, higher_better, CATEGORIES, cat_max, total_max, reference_symbol, lock):
    raw_values = {}
    for s in symbols:
        d = data[s]
        ratios_list = [item for item in d['ratios'] if item[0] <= hist_date]
        latest_ratios = max(ratios_list, key=lambda x: x[0])[1] if ratios_list else {}
        growth_list = [item for item in d['growth'] if item[0] <= hist_date]
        latest_growth = max(growth_list, key=lambda x: x[0])[1] if growth_list else {}
        quotes_list = [q for q in d['quotes'] if q[0] <= hist_date]
        returns = calculate_returns(quotes_list, hist_date)
        raw_values[s] = {**latest_ratios, **latest_growth, **returns}

    all_cat_scores = {}
    for cat, config in CATEGORIES.items():
        metrics = config['metrics']
        weights = config['weights']
        total_w = sum(weights.values())
        metric_sub_scores = {}
        for metric in metrics:
            values = [raw_values[s].get(metric) for s in symbols if raw_values[s].get(metric) is not None]
            valid_symbols = [s for s in symbols if raw_values[s].get(metric) is not None]
            if len(valid_symbols) < 2:
                sub_s = {s: 50.0 for s in symbols}
            else:
                min_v = min(values)
                max_v = max(values)
                sub_s = {}
                if min_v == max_v:
                    for ss in valid_symbols:
                        sub_s[ss] = 50.0
                else:
                    for i, ss in enumerate(valid_symbols):
                        v = values[i]
                        if higher_better[metric]:
                            norm = 100 * (v - min_v) / (max_v - min_v)
                        else:
                            norm = 100 * (max_v - v) / (max_v - min_v)
                        sub_s[ss] = norm
                for ss in symbols:
                    if ss not in sub_s:
                        sub_s[ss] = 0.0
            metric_sub_scores[metric] = sub_s
        cat_scores = {}
        for s in symbols:
            weighted_sum = sum(metric_sub_scores[m][s] * weights[m] for m in metrics)
            cat_scores[s] = weighted_sum / total_w if total_w else 0
        all_cat_scores[cat] = cat_scores

    overall_scores = {}
    for s in symbols:
        cat_contrib = sum(all_cat_scores[cat].get(s, 0) / 100 * cat_max[cat] for cat in CATEGORIES)
        overall_scores[s] = cat_contrib / total_max * 100 if total_max else 0

    inserts = []
    for cat in CATEGORIES:
        cat_scores = all_cat_scores[cat]
        normalized_cat = normalize_scores(cat_scores)
        sorted_symbols = sorted(normalized_cat, key=normalized_cat.get, reverse=True)
        with lock:
            print_ranked_scores(normalized_cat, f"{cat} as of {hist_date}")
        for rank, sym in enumerate(sorted_symbols, 1):
            inserts.append((reference_symbol, sym, hist_date, cat, normalized_cat[sym], rank))

    normalized_overall = normalize_scores(overall_scores)
    sorted_symbols = sorted(normalized_overall, key=normalized_overall.get, reverse=True)
    with lock:
        print_ranked_scores(normalized_overall, f"Overall as of {hist_date}")
    for rank, sym in enumerate(sorted_symbols, 1):
        inserts.append((reference_symbol, sym, hist_date, 'Overall', normalized_overall[sym], rank))

    return inserts

def print_ranked_scores(scores, title):
    sorted_symbols = sorted(scores, key=scores.get, reverse=True)
    print(f"\n{title}:")
    print(f"{'Symbol':<10} {'Score':<8} {'Rank':<4}")
    print("-" * 24)
    for rank, sym in enumerate(sorted_symbols, 1):
        print(f"{sym:<10} {scores[sym]:<8.2f} {rank:<4}")

def normalize_scores(scores):
    values = list(scores.values())
    if not values:
        return {s: 0.0 for s in scores}
    min_v = min(values)
    max_v = max(values)
    if min_v == max_v:
        return {s: 50.0 for s in scores}
    return {s: 100 * (scores[s] - min_v) / (max_v - min_v) for s in scores}

def process_ref(ref, peers_dict, data, historical_dates, higher_better, CATEGORIES, cat_max, total_max, lock):
    peers = peers_dict[ref]
    symbols = list(set([ref] + peers))
    if len(symbols) < 2:
        print(f"Insufficient symbols for {ref}")
        return []
    ref_data = {s: data.get(s, {'ratios': [], 'growth': [], 'quotes': []}) for s in symbols}
    ref_inserts = []
    with tqdm(total=len(historical_dates), desc=f"Computing for {ref}", leave=False) as date_pbar:
        with ThreadPoolExecutor(max_workers=min(12, len(historical_dates))) as executor:
            future_to_date = {
                executor.submit(
                    compute_scores_for_date, hist_date, ref_data, symbols, higher_better, CATEGORIES, cat_max, total_max, ref, lock
                ): hist_date for hist_date in historical_dates
            }
            for future in as_completed(future_to_date):
                try:
                    ins = future.result()
                    ref_inserts.extend(ins)
                except Exception as e:
                    print(f"Error computing for {ref} on {future_to_date[future]}: {e}")
                date_pbar.update(1)
    return ref_inserts

def fetch_peers(ref):
    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT peer_symbol FROM equity_peers WHERE symbol = %s", (ref,))
    peers_rows = cur.fetchall()
    peers = [r[0].strip().upper() for r in peers_rows if r[0]]
    cur.close()
    conn.close()
    return peers

if __name__ == "__main__":
    create_scores_table()

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT symbol FROM equity_profile ORDER BY symbol")
    reference_symbols = [row[0].strip().upper() for row in cur.fetchall()]
    cur.close()
    conn.close()

    all_symbols = set()
    peers_dict = {}
    with tqdm(total=len(reference_symbols), desc="Fetching peers") as pbar:
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_ref = {executor.submit(fetch_peers, ref): ref for ref in reference_symbols}
            for future in as_completed(future_to_ref):
                ref = future_to_ref[future]
                try:
                    peers = future.result()
                    peers_dict[ref] = peers
                    all_symbols.update([ref] + peers)
                except Exception as e:
                    print(f"Error fetching peers for {ref}: {e}")
                pbar.update(1)

    all_symbols = list(all_symbols)

    data = {}
    with tqdm(total=len(all_symbols), desc="Fetching data") as pbar:
        with ThreadPoolExecutor(max_workers=min(10, len(all_symbols))) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data, s): s for s in all_symbols}
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                try:
                    data[s] = future.result()
                except Exception as e:
                    print(f"Error fetching data for {s}: {e}")
                pbar.update(1)

    today = date.today()
    historical_dates = [today]
    current_year, current_month = today.year, today.month
    for _ in range(11):
        current_month -= 1
        if current_month == 0:
            current_month = 12
            current_year -= 1
        month_end = date(current_year, current_month, calendar.monthrange(current_year, current_month)[1])
        historical_dates.append(month_end)
    historical_dates.sort()

    higher_better = {
        'returnOnEquity': True, 'netProfitMargin': True, 'operatingProfitMargin': True,
        'debtEquityRatio': False, 'currentRatio': True, 'interestCoverage': True,
        'priceEarningsRatio': False, 'priceToBookRatio': False, 'priceEarningsToGrowthRatio': False,
        'threeYRevenueGrowthPerShare': True, 'threeYNetIncomeGrowthPerShare': True,
        '1M_return': True, 'YTD_return': True, '1Y_return': True, '3Y_return': True, '5Y_return': True,
    }

    CATEGORIES = {
        'Profitability': {
            'metrics': ['returnOnEquity', 'netProfitMargin', 'operatingProfitMargin'],
            'weights': {'returnOnEquity': 6, 'netProfitMargin': 5, 'operatingProfitMargin': 4}
        },
        'Price Performance': {
            'metrics': ['1M_return', 'YTD_return', '1Y_return', '3Y_return', '5Y_return'],
            'weights': {'1M_return': 1, 'YTD_return': 2, '1Y_return': 5, '3Y_return': 10, '5Y_return': 7}
        },
        'Growth': {
            'metrics': ['threeYRevenueGrowthPerShare', 'threeYNetIncomeGrowthPerShare'],
            'weights': {'threeYRevenueGrowthPerShare': 10, 'threeYNetIncomeGrowthPerShare': 10}
        },
        'Financial Strength': {
            'metrics': ['debtEquityRatio', 'currentRatio', 'interestCoverage'],
            'weights': {'debtEquityRatio': 10, 'currentRatio': 5, 'interestCoverage': 5}
        },
        'Valuation': {
            'metrics': ['priceEarningsRatio', 'priceToBookRatio', 'priceEarningsToGrowthRatio'],
            'weights': {'priceEarningsRatio': 8, 'priceToBookRatio': 6, 'priceEarningsToGrowthRatio': 6}
        }
    }

    cat_max = {'Profitability': 15, 'Price Performance': 25, 'Growth': 20, 'Financial Strength': 20, 'Valuation': 20}
    total_max = 100

    lock = threading.Lock()
    with tqdm(total=len(reference_symbols), desc="Processing references") as ref_pbar:
        with ThreadPoolExecutor(max_workers=5) as ref_executor:
            future_to_ref = {ref_executor.submit(process_ref, ref, peers_dict, data, historical_dates, higher_better, CATEGORIES, cat_max, total_max, lock): ref for ref in reference_symbols}
            for future in as_completed(future_to_ref):
                ref = future_to_ref[future]
                try:
                    ref_inserts = future.result()
                    try:
                        bulk_save_scores(ref_inserts)
                    except Exception as e:
                        print(f"Error saving for {ref}: {e}")
                except Exception as e:
                    print(f"Error processing {ref}: {e}")
                ref_pbar.update(1)