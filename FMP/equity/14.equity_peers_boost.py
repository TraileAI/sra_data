import os
from dotenv import load_dotenv
import pandas as pd
from sqlalchemy import create_engine, text
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DB_URL)

low_peer_query = """
SELECT symbol
FROM (
    SELECT symbol, COUNT(peer_symbol) as peer_count
    FROM equity_peers
    GROUP BY symbol
    HAVING COUNT(peer_symbol) <= 3
) t
"""
low_symbols_df = pd.read_sql(low_peer_query, engine)
low_symbols = low_symbols_df['symbol'].tolist()

if not low_symbols:
    print("No symbols with <=3 peers found.")
    exit()

print(f"Found {len(low_symbols)} symbols with <=3 peers")

profiles_query = text("""
SELECT symbol, sector, industry, currency, mkt_cap
FROM equity_profile
WHERE symbol = ANY(:symbols)
""")

with engine.connect() as conn:
    result = conn.execute(profiles_query, {'symbols': low_symbols})
    profiles_df = pd.DataFrame(result.fetchall(), columns=result.keys())

profiles_dict = profiles_df.set_index('symbol').to_dict('index')

def find_similar(symbol):
    profile = profiles_dict[symbol]
    sector = profile['sector']
    industry = profile['industry']
    currency = profile['currency']
    mkt_cap = profile['mkt_cap']

    base_query_str = """
    SELECT symbol as peer_symbol
    FROM equity_profile
    WHERE sector = :sector AND industry = :industry AND currency = :currency
    AND symbol != :symbol
    """
    params = {'sector': sector, 'industry': industry, 'currency': currency, 'symbol': symbol}

    if pd.notna(mkt_cap):
        low_cap = mkt_cap * 0.5
        high_cap = mkt_cap * 2.0
        base_query_str += " AND mkt_cap BETWEEN :low_cap AND :high_cap"
        params.update({'low_cap': low_cap, 'high_cap': high_cap})

    base_query = text(base_query_str)

    with engine.connect() as conn:
        similar_result = conn.execute(base_query, params)
        similar_df = pd.DataFrame(similar_result.fetchall(), columns=similar_result.keys())

        existing_query = text("SELECT peer_symbol FROM equity_peers WHERE symbol = :symbol")
        existing_result = conn.execute(existing_query, {'symbol': symbol})
        existing_df = pd.DataFrame(existing_result.fetchall(), columns=existing_result.keys())

    existing_peers = set(existing_df['peer_symbol'].tolist())
    new_peers = [s for s in similar_df['peer_symbol'].tolist() if s not in existing_peers]
    return [{'symbol': symbol, 'peer_symbol': p} for p in new_peers[:10]]

new_rows = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = [executor.submit(find_similar, sym) for sym in low_symbols]
    for future in tqdm(as_completed(futures), total=len(futures), desc="Finding similar peers"):
        new_rows.extend(future.result())

if new_rows:
    insert_df = pd.DataFrame(new_rows)
    insert_df.to_sql('equity_peers', engine, if_exists='append', index=False, method='multi', chunksize=1000)
    print(f"Added {len(new_rows)} new peer relationships")
else:
    print("No new peers to add.")