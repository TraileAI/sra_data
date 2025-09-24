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
    'Price Performance': {
        'Adj Close Prices': ('equity_quotes', 'adjClose')
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


def has_data(cursor, table, column, symbol):
    col_quoted = f'"{column}"'
    if table == 'equity_quotes' and column == 'adjClose':
        cursor.execute(f"""
            SELECT 1 FROM {table} 
            WHERE symbol = %s AND {col_quoted} IS NOT NULL 
            LIMIT 1
        """, (symbol,))
    else:
        cursor.execute(f"""
            SELECT 1 FROM {table} 
            WHERE symbol = %s AND {col_quoted} IS NOT NULL 
            ORDER BY date DESC 
            LIMIT 1
        """, (symbol,))
    return cursor.fetchone() is not None


def check_ticker_data(symbol):
    conn = connect_db()
    cursor = conn.cursor()

    results = {}
    total_checks = sum(len(metrics) for metrics in METRICS.values())

    with tqdm(total=total_checks, desc="Checking data") as pbar:
        for category, metrics in METRICS.items():
            results[category] = {}
            for metric, (table, column) in metrics.items():
                results[category][metric] = has_data(cursor, table, column, symbol)
                pbar.update(1)

    cursor.close()
    conn.close()

    print(f"\nData availability for {symbol}:")
    for category, metrics in results.items():
        print(f"\n{category}:")
        for metric, available in metrics.items():
            status = "✓ Available" if available else "✗ Missing"
            print(f"  {metric}: {status}")


if __name__ == "__main__":
    symbol = input("Enter ticker symbol: ").strip().upper()
    if symbol:
        check_ticker_data(symbol)
    else:
        print("No ticker provided.")