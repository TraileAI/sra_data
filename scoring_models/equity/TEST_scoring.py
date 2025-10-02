import os
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import bisect

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
            SELECT returnonequity, netprofitmargin, operatingprofitmargin, debtequityratio,
                   currentratio, interestcoverage, priceearningsratio, pricetobookratio, priceearningstogrowthratio
            FROM equity_financial_ratio WHERE symbol = %s ORDER BY date DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        ratios = {}
        if row:
            ratios = {
                'returnOnEquity': row[0],
                'netProfitMargin': row[1],
                'operatingProfitMargin': row[2],
                'debtEquityRatio': row[3],
                'currentRatio': row[4],
                'interestCoverage': row[5],
                'priceEarningsRatio': row[6],
                'priceToBookRatio': row[7],
                'priceEarningsToGrowthRatio': row[8],
            }

        cur.execute("""
            SELECT threeyrevenuegrowthpershare, threeynetincomegrowthpershare
            FROM equity_financial_growth WHERE symbol = %s ORDER BY date DESC LIMIT 1
        """, (symbol,))
        row = cur.fetchone()
        growth = {}
        if row:
            growth = {
                'threeYRevenueGrowthPerShare': row[0],
                'threeYNetIncomeGrowthPerShare': row[1],
            }

        cur.execute("""
            SELECT date::date, adjclose FROM equity_quotes WHERE symbol = %s ORDER BY date ASC
        """, (symbol,))
        quotes = cur.fetchall()

        returns = calculate_returns(quotes)

        cur.close()
        conn.close()
        return {'ratios': ratios, 'growth': growth, 'returns': returns}
    except Exception:
        cur.close()
        conn.close()
        raise

def calculate_returns(quotes):
    if not quotes:
        return {}
    dates = [q[0] for q in quotes]
    prices = [q[1] for q in quotes]
    current_price = prices[-1] if prices else None
    if not current_price:
        return {}
    today = date.today()
    periods = {
        '1M_return': today - timedelta(days=30),
        'YTD_return': date(today.year, 1, 1),
        '1Y_return': today - timedelta(days=365),
        '3Y_return': today - timedelta(days=1095),
        '5Y_return': today - timedelta(days=1825),
    }
    returns = {}
    for key, past_date in periods.items():
        idx = bisect.bisect_left(dates, past_date) - 1
        if idx >= 0:
            past_price = prices[idx]
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

    raw_values = {s: {**data[s]['ratios'], **data[s]['growth'], **data[s]['returns']} for s in symbols}

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
            score = sum(metric_sub_scores[m][s] * weights[m] for m in metrics) / total_w
            cat_scores[s] = score
        all_cat_scores[cat] = cat_scores

        normalized_cat_scores = normalize_scores(cat_scores)
        print_ranked_scores(normalized_cat_scores, cat)

    overall_scores = {}
    for s in symbols:
        score = sum(all_cat_scores[cat][s] / 100 * cat_max[cat] for cat in all_cat_scores) / total_max * 100
        overall_scores[s] = score

    normalized_overall = normalize_scores(overall_scores)
    print_ranked_scores(normalized_overall, "Overall")