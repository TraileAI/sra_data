"""
Calculate index performance metrics from market_sector_quotes table.
This script calculates day, week, quarter, YTD changes and trends for market indices.
"""
import psycopg2
import logging
import os
import sys

# Add parent directory to path for config import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Market indices to track
SYMBOLS = ['XIU.TO', '^GSPC', 'QQQ', 'DIA', 'SPY', 'XLK', 'XLV', 'XLF',
           'XLY', 'XLI', 'XLB', 'XLE', 'XLU', 'XLRE', 'XLC', 'XLP']

def create_table(conn):
    """Create index_performance table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS index_performance (
                symbol VARCHAR(15) PRIMARY KEY,
                price DOUBLE PRECISION,
                day_change DOUBLE PRECISION,
                week_change DOUBLE PRECISION,
                quarter_change DOUBLE PRECISION,
                ytd_change DOUBLE PRECISION,
                trend VARCHAR(50)
            )
        """)
    conn.commit()

def calculate_performance(conn):
    """Calculate performance metrics from market_sector_quotes."""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("TRUNCATE TABLE index_performance")

        # Calculate metrics for each symbol
        for symbol in SYMBOLS:
            cur.execute("""
                WITH latest AS (
                    SELECT adjClose as price, date
                    FROM market_sector_quotes
                    WHERE symbol = %s
                    ORDER BY date DESC
                    LIMIT 1
                ),
                day_ago AS (
                    SELECT adjClose
                    FROM market_sector_quotes
                    WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '1 day'
                    ORDER BY date DESC
                    LIMIT 1
                ),
                week_ago AS (
                    SELECT adjClose
                    FROM market_sector_quotes
                    WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '7 days'
                    ORDER BY date DESC
                    LIMIT 1
                ),
                quarter_ago AS (
                    SELECT adjClose
                    FROM market_sector_quotes
                    WHERE symbol = %s AND date <= (SELECT date FROM latest) - INTERVAL '90 days'
                    ORDER BY date DESC
                    LIMIT 1
                ),
                ytd_start AS (
                    SELECT adjClose
                    FROM market_sector_quotes
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
                        FROM market_sector_quotes
                        WHERE symbol = %s
                    ) weekly
                    WHERE rn = 1
                    ORDER BY date DESC
                    LIMIT 1
                )
                INSERT INTO index_performance (symbol, price, day_change, week_change, quarter_change, ytd_change, trend)
                SELECT
                    %s as symbol,
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
            """, (symbol, symbol, symbol, symbol, symbol, symbol, symbol))

    conn.commit()
    logger.info(f"✅ Calculated performance metrics for {len(SYMBOLS)} indices")

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)
    calculate_performance(conn)
    conn.close()
