import os
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import bisect
import calendar

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'Naura'),
    'user': os.getenv('DB_USER', 'nauraai'),
    'password': os.getenv('DB_PASSWORD', '')
}

def connect_db():
    return psycopg2.connect(**DB_CONFIG)

def fetch_symbol_data(symbol):
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT date::date, "returnOnEquity", "netProfitMargin", "operatingProfitMargin", "debtEquityRatio",
                   "currentRatio", "interestCoverage", "priceEarningsRatio", "priceToBookRatio", "priceEarningsToGrowthRatio"
            FROM equity_ratios WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        rows = cur.fetchall()
        ratios = [(row[0], {
            'returnOnEquity': row[1], 'netProfitMargin': row[2], 'operatingProfitMargin': row[3],
            'debtEquityRatio': row[4], 'currentRatio': row[5], 'interestCoverage': row[6],
            'priceEarningsRatio': row[7], 'priceToBookRatio': row[8], 'priceEarningsToGrowthRatio': row[9]
        }) for row in rows]

        cur.execute("""
            SELECT date::date, "threeYRevenueGrowthPerShare", "threeYNetIncomeGrowthPerShare"
            FROM equity_financial_growth WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        rows = cur.fetchall()
        growth = [(row[0], {
            'threeYRevenueGrowthPerShare': row[1], 'threeYNetIncomeGrowthPerShare': row[2]
        }) for row in rows]

        cur.execute("""
            SELECT date::date, "adjClose" FROM equity_quotes WHERE symbol = %s ORDER BY date ASC
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

if __name__ == "__main__":
    symbol = input("Enter ticker symbol: ").strip().upper()
    if not symbol:
        exit()

    conn = connect_db()
    cur = conn.cursor()
    cur.execute("SELECT peer_symbol FROM equity_peers WHERE symbol = %s", (symbol,))
    peers_rows = cur.fetchall()
    cur.close()
    conn.close()

    peers = [r[0].strip().upper() for r in peers_rows if r[0]]
    print(f"Peers for {symbol}: {', '.join(peers) if peers else 'None'}")

    symbols = list(set([symbol] + peers))
    n_symbols = len(symbols)
    if n_symbols < 2:
        print("Insufficient symbols for ranking.")
        exit()

    data = {}
    with tqdm(total=n_symbols, desc="Fetching data") as pbar:
        with ThreadPoolExecutor(max_workers=min(10, n_symbols)) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data, s): s for s in symbols}
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                data[s] = future.result()
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

    with tqdm(total=len(historical_dates), desc="Computing scores") as pbar:
        for hist_date in historical_dates:
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
                    values = []
                    valid_symbols = []
                    for s in symbols:
                        v = raw_values.get(s, {}).get(metric)
                        if v is not None:
                            values.append(v)
                            valid_symbols.append(s)
                    if len(valid_symbols) < 2:
                        sub_s = {s: 50.0 for s in symbols}
                    else:
                        min_v = min(values)
                        max_v = max(values)
                        sub_s = {}
                        if min_v == max_v:
                            for s in valid_symbols:
                                sub_s[s] = 50.0
                        else:
                            for i, s in enumerate(valid_symbols):
                                v = values[i]
                                if higher_better[metric]:
                                    norm = 100 * (v - min_v) / (max_v - min_v)
                                else:
                                    norm = 100 * (max_v - v) / (max_v - min_v)
                                sub_s[s] = norm
                        for s in symbols:
                            sub_s.setdefault(s, 0.0)
                    metric_sub_scores[metric] = sub_s

                cat_scores = {}
                for s in symbols:
                    score = sum(metric_sub_scores[m][s] * weights[m] for m in metrics) / total_w if total_w else 0
                    cat_scores[s] = score
                all_cat_scores[cat] = cat_scores

                normalized_cat = normalize_scores(cat_scores)
                print_ranked_scores(normalized_cat, f"{cat} as of {hist_date}")

            overall_scores = {}
            for s in symbols:
                score = sum(all_cat_scores[cat][s] / 100 * cat_max[cat] for cat in all_cat_scores) / total_max * 100
                overall_scores[s] = score

            normalized_overall = normalize_scores(overall_scores)
            print_ranked_scores(normalized_overall, f"Overall as of {hist_date}")

            pbar.update(1)