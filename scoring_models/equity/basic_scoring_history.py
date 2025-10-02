import os
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
import bisect
import pandas as pd
from dateutil.relativedelta import relativedelta

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

def get_month_ends(num_months=12):
    today = date.today()
    month_ends = []
    for i in range(num_months):
        end_date = today - relativedelta(months=i)
        month_end = date(end_date.year, end_date.month, 1) + relativedelta(months=1) - timedelta(days=1)
        month_ends.append(month_end)
    return sorted(month_ends)

def fetch_symbol_data_at_date(symbol, cutoff_date):
    cutoff_date = min(cutoff_date, date.today())
    conn = connect_db()
    cur = conn.cursor()
    try:
        cur.execute("""
            SELECT returnonequity, netprofitmargin, operatingprofitmargin, debtequityratio,
                   currentratio, interestcoverage, priceearningsratio, pricetobookratio, priceearningstogrowthratio
            FROM equity_financial_ratio WHERE symbol = %s AND date::date <= %s ORDER BY date::date DESC LIMIT 1
        """, (symbol, cutoff_date))
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
            FROM equity_financial_growth WHERE symbol = %s AND date::date <= %s ORDER BY date::date DESC LIMIT 1
        """, (symbol, cutoff_date))
        row = cur.fetchone()
        growth = {}
        if row:
            growth = {
                'threeYRevenueGrowthPerShare': row[0],
                'threeYNetIncomeGrowthPerShare': row[1],
            }

        cur.execute("""
            SELECT date::date AS quote_date, adjclose FROM equity_quotes
            WHERE symbol = %s AND date::date <= %s ORDER BY date::date ASC
        """, (symbol, cutoff_date))
        quotes = cur.fetchall()

        returns = calculate_returns(quotes, cutoff_date)

        cur.close()
        conn.close()
        return {'ratios': ratios, 'growth': growth, 'returns': returns}
    except Exception:
        cur.close()
        conn.close()
        raise

def calculate_returns(quotes, cutoff_date):
    if not quotes:
        return {}
    dates = [q[0] for q in quotes]
    prices = [q[1] for q in quotes]
    current_price = prices[-1] if prices else None
    if not current_price:
        return {}
    periods = {
        '1M_return': cutoff_date - timedelta(days=30),
        'YTD_return': date(cutoff_date.year, 1, 1),
        '1Y_return': cutoff_date - timedelta(days=365),
        '3Y_return': cutoff_date - timedelta(days=1095),
        '5Y_return': cutoff_date - timedelta(days=1825),
    }
    returns = {}
    for key, past_date in periods.items():
        idx = bisect.bisect_right(dates, past_date) - 1
        if idx >= 0:
            past_price = prices[idx]
            if past_price and past_price != 0:
                ret = (current_price - past_price) / past_price * 100
                returns[key] = ret
    return returns

def compute_scores(raw_values, symbols, higher_better, CATEGORIES, cat_max):
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
            if len(valid_symbols) == 0:
                sub_s = {s: 0.0 for s in symbols}
            elif len(valid_symbols) == 1:
                sub_s = {valid_symbols[0]: 100.0}
                for s in symbols:
                    if s not in sub_s:
                        sub_s[s] = 0.0
            else:
                sub_s = {}
                for i, s in enumerate(valid_symbols):
                    v = values[i]
                    if higher_better[metric]:
                        count = sum(1 for vv in values if vv <= v)
                    else:
                        count = sum(1 for vv in values if vv >= v)
                    perc = (count / len(values)) * 100
                    sub_s[s] = perc
                for s in symbols:
                    if s not in sub_s:
                        sub_s[s] = 0.0
            metric_sub_scores[metric] = sub_s

        cat_scores = {}
        for s in symbols:
            score = sum(metric_sub_scores[m][s] * (weights[m] / total_w) for m in metrics)
            cat_scores[s] = score
        all_cat_scores[cat] = cat_scores

    total_max = sum(cat_max.values())
    overall_scores = {}
    for s in symbols:
        score = sum(all_cat_scores[cat][s] / 100 * cat_max[cat] for cat in all_cat_scores) / total_max * 100
        overall_scores[s] = score

    return all_cat_scores, overall_scores

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

    month_ends = get_month_ends(12)
    print(f"Computing scores for {len(month_ends)} months: {month_ends[-1].strftime('%Y-%m-%d')} to {month_ends[0].strftime('%Y-%m-%d')}")

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
            'weights': {'returnOnEquity': 12, 'netProfitMargin': 10, 'operatingProfitMargin': 8}
        },
        'Price Performance': {
            'metrics': ['1M_return', 'YTD_return', '1Y_return', '3Y_return', '5Y_return'],
            'weights': {'1M_return': 1, 'YTD_return': 2, '1Y_return': 5, '3Y_return': 10, '5Y_return': 7}
        },
        'Growth': {
            'metrics': ['threeYRevenueGrowthPerShare', 'threeYNetIncomeGrowthPerShare'],
            'weights': {'threeYRevenueGrowthPerShare': 15, 'threeYNetIncomeGrowthPerShare': 15}
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

    cat_max = {'Profitability': 30, 'Price Performance': 25, 'Growth': 30, 'Financial Strength': 20, 'Valuation': 20}

    historical_cat_scores = {cat: {md: {} for md in month_ends} for cat in CATEGORIES}
    historical_cat_ranks = {cat: {md: {} for md in month_ends} for cat in CATEGORIES}
    historical_overall_scores = {md: {} for md in month_ends}
    historical_overall_ranks = {md: {} for md in month_ends}
    historical_raw_values = {md: {} for md in month_ends}

    for md in tqdm(month_ends, desc="Processing months"):
        with ThreadPoolExecutor(max_workers=min(10, n_symbols)) as executor:
            future_to_symbol = {executor.submit(fetch_symbol_data_at_date, s, md): s for s in symbols}
            for future in as_completed(future_to_symbol):
                s = future_to_symbol[future]
                historical_raw_values[md][s] = future.result()

        raw_values = {s: {**historical_raw_values[md][s]['ratios'], **historical_raw_values[md][s]['growth'], **historical_raw_values[md][s]['returns']} for s in symbols}
        all_cat, overall = compute_scores(raw_values, symbols, higher_better, CATEGORIES, cat_max)
        for cat in CATEGORIES:
            historical_cat_scores[cat][md] = all_cat[cat]
            sorted_symbols = sorted(symbols, key=lambda x: all_cat[cat][x], reverse=True)
            for rank, s in enumerate(sorted_symbols, 1):
                historical_cat_ranks[cat][md][s] = rank
        historical_overall_scores[md] = overall
        sorted_symbols = sorted(symbols, key=lambda x: overall[x], reverse=True)
        for rank, s in enumerate(sorted_symbols, 1):
            historical_overall_ranks[md][s] = rank

    month_labels = [md.strftime('%Y-%m') for md in month_ends]
    for cat in CATEGORIES:
        df_data = []
        for s in symbols:
            scores = [historical_cat_scores[cat][md][s] for md in month_ends]
            df_data.append([s] + scores)
        df = pd.DataFrame(df_data, columns=['Symbol'] + month_labels)
        print(f"\n{cat} Normalized Scores (/100):")
        print(df.to_string(index=False))

        df_data = []
        for s in symbols:
            ranks = [historical_cat_ranks[cat][md][s] for md in month_ends]
            df_data.append([s] + ranks)
        df = pd.DataFrame(df_data, columns=['Symbol'] + month_labels)
        print(f"\n{cat} Ranks (1 = best):")
        print(df.to_string(index=False))

    df_data = []
    for s in symbols:
        scores = [historical_overall_scores[md][s] for md in month_ends]
        df_data.append([s] + scores)
    df = pd.DataFrame(df_data, columns=['Symbol'] + month_labels)
    print(f"\nOverall Normalized Scores (/100):")
    print(df.to_string(index=False))

    df_data = []
    for s in symbols:
        ranks = [historical_overall_ranks[md][s] for md in month_ends]
        df_data.append([s] + ranks)
    df = pd.DataFrame(df_data, columns=['Symbol'] + month_labels)
    print(f"\nOverall Ranks (1 = best):")
    print(df.to_string(index=False))

    latest_md = month_ends[0]
    raw_latest = {s: {**historical_raw_values[latest_md][s]['ratios'], **historical_raw_values[latest_md][s]['growth'], **historical_raw_values[latest_md][s]['returns']} for s in symbols}
    df_raw = pd.DataFrame.from_dict(raw_latest, orient='index')
    print(f"\nRaw Metric Values for Latest Month ({month_labels[0]}):")
    print(df_raw.to_string())