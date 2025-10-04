"""
Calculate market/sector performance with top gainers and losers.
This script analyzes constituents of major indices and sectors to identify top performers.
"""
import requests
import psycopg2
import logging
import os
import sys
from typing import List, Dict

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

# Market and sector symbols to track
SYMBOLS = ['XIU.TO', '^GSPC', 'QQQ', 'DIA', 'SPY', 'XLK', 'XLV', 'XLF',
           'XLY', 'XLI', 'XLB', 'XLE', 'XLU', 'XLRE', 'XLC', 'XLP']

SECTOR_MAP = {
    'XLK': 'Technology',
    'XLV': 'Healthcare',
    'XLF': 'Financial Services',
    'XLY': 'Consumer Cyclical',
    'XLI': 'Industrials',
    'XLB': 'Basic Materials',
    'XLE': 'Energy',
    'XLU': 'Utilities',
    'XLRE': 'Real Estate',
    'XLC': 'Communication Services',
    'XLP': 'Consumer Defensive'
}

TSX60_SYMBOLS = ['AEM', 'AQN', 'ATD', 'BMO', 'BNS', 'ABX', 'BCE', 'BAM', 'BN', 'BIP-UN',
                 'CAE', 'CCO', 'CAR-UN', 'CM', 'CNR', 'CNQ', 'CP', 'CTC-A', 'CCL-B', 'CVE',
                 'GIB-A', 'CSU', 'DOL', 'EMA', 'ENB', 'FM', 'FSV', 'FTS', 'FNV', 'WPM',
                 'GIL', 'H', 'IMO', 'IFC', 'K', 'L', 'MG', 'MFC', 'MRU', 'NA',
                 'NTR', 'OTEX', 'PPL', 'POW', 'QSR', 'RCI-B', 'RY', 'SAP', 'SHOP', 'SLF',
                 'SU', 'TRP', 'TECK-B', 'T', 'TRI', 'TD', 'TOU', 'WCN', 'WSP']

BROAD_ENDPOINTS = {
    'SPY': 'https://financialmodelingprep.com/stable/sp500-constituent',
    '^GSPC': 'https://financialmodelingprep.com/stable/sp500-constituent',
    'QQQ': 'https://financialmodelingprep.com/stable/nasdaq-constituent',
    'DIA': 'https://financialmodelingprep.com/stable/dowjones-constituent'
}

def create_table(conn):
    """Create market_sector_pages_performance table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS market_sector_pages_performance (
                parent_symbol VARCHAR(15),
                symbol VARCHAR(15),
                type VARCHAR(50),
                category VARCHAR(50),
                rank INTEGER,
                sector VARCHAR(100),
                industry VARCHAR(100),
                name VARCHAR(255),
                price DOUBLE PRECISION,
                day_change DOUBLE PRECISION,
                week_change DOUBLE PRECISION,
                quarter_change DOUBLE PRECISION,
                ytd_change DOUBLE PRECISION,
                trend VARCHAR(50),
                PRIMARY KEY (parent_symbol, symbol, category)
            )
        """)
    conn.commit()

def get_constituents(parent_symbol: str, conn) -> List[str]:
    """Get constituent symbols for a given parent symbol."""
    if parent_symbol in SECTOR_MAP:
        # Get stocks from this sector
        sector = SECTOR_MAP[parent_symbol]
        with conn.cursor() as cur:
            cur.execute("SELECT symbol FROM equity_profile WHERE sector = %s", (sector,))
            return [row[0] for row in cur.fetchall()]

    elif parent_symbol == 'XIU.TO':
        # TSX 60 constituents
        return [s + '.TO' for s in TSX60_SYMBOLS]

    elif parent_symbol in BROAD_ENDPOINTS:
        # Fetch from API
        url = f"{BROAD_ENDPOINTS[parent_symbol]}?apikey={FMP_API_KEY}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            return [item['symbol'] for item in data]
        except Exception as e:
            logger.error(f"Error fetching constituents for {parent_symbol}: {e}")
            return []

    return []

def get_profile_data(symbol: str, conn) -> Dict:
    """Get profile data for a symbol."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT sector, industry, company_name
            FROM equity_profile
            WHERE symbol = %s
        """, (symbol,))
        row = cur.fetchone()
        if row:
            return {'sector': row[0], 'industry': row[1], 'name': row[2]}
    return {'sector': None, 'industry': None, 'name': None}

def calculate_metrics(symbol: str, conn) -> Dict:
    """Calculate performance metrics for a symbol from equity_quotes."""
    with conn.cursor() as cur:
        cur.execute("""
            WITH latest AS (
                SELECT adjClose as price, date
                FROM equity_quotes
                WHERE symbol = %s
                ORDER BY date DESC
                LIMIT 1
            ),
            day_ago AS (
                SELECT adjClose
                FROM equity_quotes
                WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '1 day'
                ORDER BY date DESC
                LIMIT 1
            ),
            week_ago AS (
                SELECT adjClose
                FROM equity_quotes
                WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '7 days'
                ORDER BY date DESC
                LIMIT 1
            ),
            quarter_ago AS (
                SELECT adjClose
                FROM equity_quotes
                WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '90 days'
                ORDER BY date DESC
                LIMIT 1
            ),
            ytd_start AS (
                SELECT adjClose
                FROM equity_quotes
                WHERE symbol = %s
                  AND EXTRACT(YEAR FROM date) = EXTRACT(YEAR FROM (SELECT date FROM latest))
                ORDER BY date ASC
                LIMIT 1
            ),
            weekly_prices AS (
                SELECT date, adjClose,
                       AVG(adjClose) OVER (ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ema10,
                       AVG(adjClose) OVER (ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as ema30
                FROM (
                    SELECT date, adjClose,
                           ROW_NUMBER() OVER (PARTITION BY DATE_TRUNC('week', date) ORDER BY date DESC) as rn
                    FROM equity_quotes
                    WHERE symbol = %s
                ) weekly
                WHERE rn = 1
                ORDER BY date DESC
                LIMIT 1
            )
            SELECT
                l.price,
                CASE WHEN d.adjClose IS NOT NULL THEN ((l.price - d.adjClose) / d.adjClose * 100) ELSE 0 END as day_change,
                CASE WHEN w.adjClose IS NOT NULL THEN ((l.price - w.adjClose) / w.adjClose * 100) ELSE 0 END as week_change,
                CASE WHEN q.adjClose IS NOT NULL THEN ((l.price - q.adjClose) / q.adjClose * 100) ELSE 0 END as quarter_change,
                CASE WHEN y.adjClose IS NOT NULL THEN ((l.price - y.adjClose) / y.adjClose * 100) ELSE 0 END as ytd_change,
                CASE
                    WHEN wp.ema10 IS NULL OR wp.ema30 IS NULL THEN 'Insufficient data'
                    WHEN l.price > wp.ema30 AND wp.ema10 > wp.ema30 THEN 'Uptrend'
                    WHEN (l.price > wp.ema30 AND wp.ema10 < wp.ema30) OR (l.price < wp.ema30 AND wp.ema10 > wp.ema30) THEN 'HOLD'
                    ELSE 'Downtrend'
                END as trend
            FROM latest l
            LEFT JOIN day_ago d ON true
            LEFT JOIN week_ago w ON true
            LEFT JOIN quarter_ago q ON true
            LEFT JOIN ytd_start y ON true
            LEFT JOIN weekly_prices wp ON true
        """, (symbol, symbol, symbol, symbol, symbol, symbol))

        row = cur.fetchone()
        if row:
            return {
                'price': row[0],
                'day_change': row[1],
                'week_change': row[2],
                'quarter_change': row[3],
                'ytd_change': row[4],
                'trend': row[5]
            }
    return None

def process_parent_symbol(parent_symbol: str, conn):
    """Process a parent symbol and its constituents."""
    logger.info(f"Processing {parent_symbol}...")

    # Get profile for parent
    profile = get_profile_data(parent_symbol, conn)

    # Calculate parent metrics
    parent_metrics = calculate_metrics(parent_symbol, conn)
    if not parent_metrics:
        # Try market_sector_quotes instead
        with conn.cursor() as cur:
            cur.execute("""
                SELECT adjClose FROM market_sector_quotes
                WHERE symbol = %s
                ORDER BY date DESC LIMIT 1
            """, (parent_symbol,))
            row = cur.fetchone()
            if row:
                parent_metrics = calculate_metrics(parent_symbol, conn)

    results = []

    # Add parent as 'Main' entry
    if parent_metrics:
        is_sector = parent_symbol in SECTOR_MAP
        results.append({
            'parent_symbol': parent_symbol,
            'symbol': parent_symbol,
            'type': 'Sector' if is_sector else 'Index',
            'category': 'Main',
            'rank': None,
            'sector': profile['sector'],
            'industry': profile['industry'],
            'name': profile['name'],
            **parent_metrics
        })

    # Get constituents
    constituents = get_constituents(parent_symbol, conn)
    logger.info(f"Found {len(constituents)} constituents for {parent_symbol}")

    if not constituents:
        return results

    # Calculate metrics for all constituents
    constituent_metrics = []
    for symbol in constituents:
        metrics = calculate_metrics(symbol, conn)
        if metrics and metrics['day_change'] is not None:
            profile = get_profile_data(symbol, conn)
            constituent_metrics.append({
                'symbol': symbol,
                'metrics': metrics,
                'profile': profile
            })

    # Sort by day_change and get top 10 gainers/losers
    constituent_metrics.sort(key=lambda x: x['metrics']['day_change'], reverse=True)

    # Top 10 gainers
    for rank, item in enumerate(constituent_metrics[:10], 1):
        results.append({
            'parent_symbol': parent_symbol,
            'symbol': item['symbol'],
            'type': 'Stock',
            'category': 'Gainer',
            'rank': rank,
            'sector': item['profile']['sector'],
            'industry': item['profile']['industry'],
            'name': item['profile']['name'],
            **item['metrics']
        })

    # Top 10 losers
    for rank, item in enumerate(constituent_metrics[-10:][::-1], 1):
        results.append({
            'parent_symbol': parent_symbol,
            'symbol': item['symbol'],
            'type': 'Stock',
            'category': 'Loser',
            'rank': rank,
            'sector': item['profile']['sector'],
            'industry': item['profile']['industry'],
            'name': item['profile']['name'],
            **item['metrics']
        })

    return results

def save_to_db(conn, all_results):
    """Save results to database."""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("TRUNCATE TABLE market_sector_pages_performance")

        # Insert new data
        for result in all_results:
            cur.execute("""
                INSERT INTO market_sector_pages_performance (
                    parent_symbol, symbol, type, category, rank,
                    sector, industry, name, price, day_change,
                    week_change, quarter_change, ytd_change, trend
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                result['parent_symbol'], result['symbol'], result['type'],
                result['category'], result['rank'], result['sector'],
                result['industry'], result['name'], result['price'],
                result['day_change'], result['week_change'], result['quarter_change'],
                result['ytd_change'], result['trend']
            ))

    conn.commit()
    logger.info(f"✅ Saved {len(all_results)} performance records")

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)

    all_results = []
    for symbol in SYMBOLS:
        results = process_parent_symbol(symbol, conn)
        all_results.extend(results)

    save_to_db(conn, all_results)
    conn.close()
