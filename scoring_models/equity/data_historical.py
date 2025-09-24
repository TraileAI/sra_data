import os
import psycopg2
from dotenv import load_dotenv
from tqdm import tqdm

load_dotenv()

DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'Naura'),
    'user': os.getenv('DB_USER', 'nauraai'),
    'password': os.getenv('DB_PASSWORD', '')
}

METRICS = {
    'Profitability': {
        'ROE': ('equity_ratios', 'returnOnEquity'),
        'Net Profit Margin': ('equity_ratios', 'netProfitMargin'),
        'Operating Margin': ('equity_ratios', 'operatingProfitMargin')
    },
    'Growth': {
        '3Y Revenue Growth': ('equity_financial_growth', 'threeYRevenueGrowthPerShare'),
        '3Y Earnings Growth': ('equity_financial_growth', 'threeYNetIncomeGrowthPerShare')
    },
    'Financial Strength': {
        'Debt-to-Equity': ('equity_ratios', 'debtEquityRatio'),
        'Current Ratio': ('equity_ratios', 'currentRatio'),
        'Interest Coverage': ('equity_ratios', 'interestCoverage')
    },
    'Valuation': {
        'P/E Ratio': ('equity_ratios', 'priceEarningsRatio'),
        'P/B Ratio': ('equity_ratios', 'priceToBookRatio'),
        'PEG Ratio': ('equity_ratios', 'priceEarningsToGrowthRatio')
    }
}


def connect_db():
    return psycopg2.connect(**DB_CONFIG)


def check_historical_data(cursor, table, column, symbol):
    col_quoted = f'"{column}"'
    cursor.execute(f"""
        SELECT COUNT(*) FROM {table}
        WHERE symbol = %s AND date::date >= CURRENT_DATE - INTERVAL '12 months' AND {col_quoted} IS NOT NULL
    """, (symbol,))
    count = cursor.fetchone()[0]

    cursor.execute(f"""
        SELECT MIN(date::date), MAX(date::date) FROM {table}
        WHERE symbol = %s AND date::date >= CURRENT_DATE - INTERVAL '12 months' AND {col_quoted} IS NOT NULL
    """, (symbol,))
    result = cursor.fetchone()
    min_date, max_date = (None, None) if result is None else result

    return count, min_date, max_date


def verify_ticker_history(symbol):
    conn = connect_db()
    cursor = conn.cursor()

    results = {}
    total_checks = sum(len(metrics) for metrics in METRICS.values())

    with tqdm(total=total_checks, desc="Checking historical data") as pbar:
        for category, metrics in METRICS.items():
            results[category] = {}
            for metric, (table, column) in metrics.items():
                count, min_date, max_date = check_historical_data(cursor, table, column, symbol)
                results[category][metric] = {'count': count, 'min_date': min_date, 'max_date': max_date}
                pbar.update(1)

    cursor.close()
    conn.close()

    print(f"\nHistorical data for {symbol} (last 12 months):")
    for category, metrics in results.items():
        print(f"\n{category}:")
        for metric, data in metrics.items():
            count = data['count']
            min_date = data['min_date']
            max_date = data['max_date']
            print(f"  {metric}: {count} records")
            if min_date and max_date:
                print(f"    Date range: {min_date} to {max_date}")
            threshold = 200 if category == 'Price Performance' else 1
            status = "✓ Sufficient" if count >= threshold else "✗ Insufficient"
            print(f"    Status: {status}")


if __name__ == "__main__":
    symbol = input("Enter ticker symbol: ").strip().upper()
    if symbol:
        verify_ticker_history(symbol)
    else:
        print("No ticker provided.")