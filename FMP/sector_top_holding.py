"""
Calculate top 10 holdings by market cap for each sector.
This script identifies the largest companies in each sector with performance metrics.
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

SECTORS = [
    'Technology', 'Healthcare', 'Financial Services', 'Consumer Cyclical',
    'Industrials', 'Basic Materials', 'Energy', 'Utilities',
    'Real Estate', 'Communication Services', 'Consumer Defensive'
]

def create_table(conn):
    """Create sector_top_holdings table if it doesn't exist."""
    with conn.cursor() as cur:
        cur.execute("""
            CREATE TABLE IF NOT EXISTS sector_top_holdings (
                sector VARCHAR(100),
                rank INTEGER,
                symbol VARCHAR(15),
                company VARCHAR(255),
                industry VARCHAR(100),
                mkt_cap BIGINT,
                price DOUBLE PRECISION,
                day_change DOUBLE PRECISION,
                week_change DOUBLE PRECISION,
                quarter_change DOUBLE PRECISION,
                ytd_change DOUBLE PRECISION,
                trend VARCHAR(50),
                PRIMARY KEY (sector, rank)
            )
        """)
    conn.commit()

def calculate_sector_top_holdings(conn):
    """Calculate top 10 holdings for each sector."""
    with conn.cursor() as cur:
        # Clear existing data
        cur.execute("TRUNCATE TABLE sector_top_holdings")

        for sector in SECTORS:
            logger.info(f"Processing sector: {sector}")

            # Get top 10 companies by market cap in this sector with performance metrics
            cur.execute("""
                WITH ranked_companies AS (
                    SELECT
                        p.symbol,
                        p.company_name,
                        p.industry,
                        CAST(p.mkt_cap AS BIGINT) as mkt_cap,
                        ROW_NUMBER() OVER (ORDER BY CAST(p.mkt_cap AS BIGINT) DESC) as rank
                    FROM equity_profile p
                    WHERE p.sector = %s
                      AND p.mkt_cap IS NOT NULL
                      AND CAST(p.mkt_cap AS BIGINT) > 0
                ),
                latest_quotes AS (
                    SELECT DISTINCT ON (symbol)
                        symbol,
                        adjclose as price,
                        date
                    FROM equity_quotes
                    ORDER BY symbol, date DESC
                ),
                day_ago_quotes AS (
                    SELECT DISTINCT ON (q.symbol)
                        q.symbol,
                        q.adjclose
                    FROM equity_quotes q
                    INNER JOIN latest_quotes l ON q.symbol = l.symbol
                    WHERE q.date <= l.date - INTERVAL '1 day'
                    ORDER BY q.symbol, q.date DESC
                ),
                week_ago_quotes AS (
                    SELECT DISTINCT ON (q.symbol)
                        q.symbol,
                        q.adjclose
                    FROM equity_quotes q
                    INNER JOIN latest_quotes l ON q.symbol = l.symbol
                    WHERE q.date <= l.date - INTERVAL '7 days'
                    ORDER BY q.symbol, q.date DESC
                ),
                quarter_ago_quotes AS (
                    SELECT DISTINCT ON (q.symbol)
                        q.symbol,
                        q.adjclose
                    FROM equity_quotes q
                    INNER JOIN latest_quotes l ON q.symbol = l.symbol
                    WHERE q.date <= l.date - INTERVAL '90 days'
                    ORDER BY q.symbol, q.date DESC
                ),
                ytd_start_quotes AS (
                    SELECT DISTINCT ON (q.symbol)
                        q.symbol,
                        q.adjclose
                    FROM equity_quotes q
                    INNER JOIN latest_quotes l ON q.symbol = l.symbol
                    WHERE EXTRACT(YEAR FROM q.date) = EXTRACT(YEAR FROM l.date)
                    ORDER BY q.symbol, q.date ASC
                ),
                weekly_trends AS (
                    SELECT
                        symbol,
                        adjclose,
                        AVG(adjclose) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 9 PRECEDING AND CURRENT ROW) as ema10,
                        AVG(adjclose) OVER (PARTITION BY symbol ORDER BY date ROWS BETWEEN 29 PRECEDING AND CURRENT ROW) as ema30,
                        ROW_NUMBER() OVER (PARTITION BY symbol, DATE_TRUNC('week', date) ORDER BY date DESC) as week_rank
                    FROM equity_quotes
                )
                INSERT INTO sector_top_holdings (
                    sector, rank, symbol, company, industry, mkt_cap,
                    price, day_change, week_change, quarter_change, ytd_change, trend
                )
                SELECT
                    %s as sector,
                    rc.rank,
                    rc.symbol,
                    rc.company_name,
                    rc.industry,
                    rc.mkt_cap,
                    lq.price,
                    CASE WHEN dq.adjclose IS NOT NULL
                        THEN ((lq.price - dq.adjclose) / dq.adjclose * 100)
                        ELSE 0
                    END as day_change,
                    CASE WHEN wq.adjclose IS NOT NULL
                        THEN ((lq.price - wq.adjclose) / wq.adjclose * 100)
                        ELSE 0
                    END as week_change,
                    CASE WHEN qq.adjclose IS NOT NULL
                        THEN ((lq.price - qq.adjclose) / qq.adjclose * 100)
                        ELSE 0
                    END as quarter_change,
                    CASE WHEN yq.adjclose IS NOT NULL
                        THEN ((lq.price - yq.adjclose) / yq.adjclose * 100)
                        ELSE 0
                    END as ytd_change,
                    CASE
                        WHEN wt.ema10 IS NULL OR wt.ema30 IS NULL THEN 'Insufficient data'
                        WHEN lq.price > wt.ema30 AND wt.ema10 > wt.ema30 THEN 'Uptrend'
                        WHEN (lq.price > wt.ema30 AND wt.ema10 < wt.ema30)
                          OR (lq.price < wt.ema30 AND wt.ema10 > wt.ema30) THEN 'HOLD'
                        ELSE 'Downtrend'
                    END as trend
                FROM ranked_companies rc
                LEFT JOIN latest_quotes lq ON rc.symbol = lq.symbol
                LEFT JOIN day_ago_quotes dq ON rc.symbol = dq.symbol
                LEFT JOIN week_ago_quotes wq ON rc.symbol = wq.symbol
                LEFT JOIN quarter_ago_quotes qq ON rc.symbol = qq.symbol
                LEFT JOIN ytd_start_quotes yq ON rc.symbol = yq.symbol
                LEFT JOIN (
                    SELECT symbol, adjclose, ema10, ema30
                    FROM weekly_trends
                    WHERE week_rank = 1
                ) wt ON rc.symbol = wt.symbol
                WHERE rc.rank <= 10
                ORDER BY rc.rank
            """, (sector, sector))

    conn.commit()
    logger.info(f"✅ Calculated top holdings for {len(SECTORS)} sectors")

if __name__ == "__main__":
    conn = psycopg2.connect(**config.db_config)
    create_table(conn)
    calculate_sector_top_holdings(conn)
    conn.close()
