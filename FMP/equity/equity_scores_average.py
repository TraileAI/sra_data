import psycopg2
import pandas as pd
from datetime import datetime
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

conn_str = get_connection_string()
conn = psycopg2.connect(conn_str)
cur = conn.cursor()

# Get score_types
cur.execute("SELECT DISTINCT score_type FROM equity_scores")
score_types = [row[0] for row in cur.fetchall()]

# Create table if not exists
cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_scores_average
            (
                group_type
                VARCHAR
            (
                20
            ) NOT NULL,
                group_name VARCHAR
            (
                100
            ) NOT NULL,
                score_type VARCHAR
            (
                50
            ) NOT NULL,
                avg_score NUMERIC
            (
                5,
                2
            ),
                calculation_date DATE NOT NULL
                )
            """)
conn.commit()

# Clear existing data
cur.execute("DELETE FROM equity_scores_average")
conn.commit()

averages_data = []
calculation_date = datetime.now().date()

for st in score_types:
    # Get latest date for this score_type
    cur.execute("SELECT MAX(date) FROM equity_scores WHERE score_type = %s", (st,))
    latest_date = cur.fetchone()[0]

    if latest_date is None:
        continue

    # Averages by sector
    cur.execute("""
                SELECT ep.sector AS group_name, AVG(es.score) AS avg_score
                FROM equity_scores es
                         JOIN equity_profile ep ON es.symbol = ep.symbol
                WHERE es.score_type = %s
                  AND es.date = %s
                  AND es.score IS NOT NULL
                  AND ep.sector IS NOT NULL
                GROUP BY ep.sector
                HAVING COUNT(*) > 0
                """, (st, latest_date))
    sector_rows = cur.fetchall()
    for row in sector_rows:
        averages_data.append({
            'group_type': 'sector',
            'group_name': row[0],
            'score_type': st,
            'avg_score': round(float(row[1]), 2) if row[1] is not None else None,
            'calculation_date': calculation_date
        })

    # Averages by industry
    cur.execute("""
                SELECT ep.industry AS group_name, AVG(es.score) AS avg_score
                FROM equity_scores es
                         JOIN equity_profile ep ON es.symbol = ep.symbol
                WHERE es.score_type = %s
                  AND es.date = %s
                  AND es.score IS NOT NULL
                  AND ep.industry IS NOT NULL
                GROUP BY ep.industry
                HAVING COUNT(*) > 0
                """, (st, latest_date))
    industry_rows = cur.fetchall()
    for row in industry_rows:
        averages_data.append({
            'group_type': 'industry',
            'group_name': row[0],
            'score_type': st,
            'avg_score': round(float(row[1]), 2) if row[1] is not None else None,
            'calculation_date': calculation_date
        })

# Insert batches
insert_query = """
               INSERT INTO equity_scores_average (group_type, group_name, score_type, avg_score, calculation_date)
               VALUES (%s, %s, %s, %s, %s) \
               """
batch_size = 1000
for i in range(0, len(averages_data), batch_size):
    batch = [(d['group_type'], d['group_name'], d['score_type'], d['avg_score'], d['calculation_date']) for d in
             averages_data[i:i + batch_size]]
    cur.executemany(insert_query, batch)
    conn.commit()

cur.close()
conn.close()
print(f"Saved {len(averages_data)} average records to equity_scores_average.")