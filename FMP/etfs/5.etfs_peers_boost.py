import os
import sys
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
    FROM etfs_peers
    GROUP BY symbol
    HAVING COUNT(peer_symbol) <= 3
) t
"""
low_symbols_df = pd.read_sql(low_peer_query, engine)
low_symbols = low_symbols_df['symbol'].tolist()

if not low_symbols:
    print("No symbols with <=3 peers found.")
    sys.exit(0)

print(f"Found {len(low_symbols)} symbols with <=3 peers")

profiles_query = text("""
SELECT symbol, currency, domicile, "assetClass", "expenseRatio", aum
FROM etfs_profile
WHERE symbol = ANY(:symbols)
""")

with engine.connect() as conn:
    result = conn.execute(profiles_query, {'symbols': low_symbols})
    profiles_df = pd.DataFrame(result.fetchall(), columns=result.keys())

profiles_dict = profiles_df.set_index('symbol').to_dict('index')

def find_similar(symbol):
    if symbol not in profiles_dict:
        return []
    profile = profiles_dict[symbol]
    currency = profile['currency']
    domicile = profile['domicile']
    assetClass = profile['assetClass']
    expenseRatio = profile['expenseRatio']
    aum = profile['aum']

    base_query_str = """
    SELECT symbol as peer_symbol
    FROM etfs_profile
    WHERE currency = :currency AND domicile = :domicile AND "assetClass" = :assetClass
    AND symbol != :symbol
    """
    params = {'currency': currency, 'domicile': domicile, 'assetClass': assetClass, 'symbol': symbol}

    if pd.notna(expenseRatio):
        low_er = expenseRatio * 0.8
        high_er = expenseRatio * 1.2
        base_query_str += ' AND "expenseRatio" BETWEEN :low_er AND :high_er'
        params.update({'low_er': low_er, 'high_er': high_er})

    if pd.notna(aum):
        low_aum = aum * 0.5
        high_aum = aum * 2.0
        base_query_str += ' AND aum BETWEEN :low_aum AND :high_aum'
        params.update({'low_aum': low_aum, 'high_aum': high_aum})

    base_query = text(base_query_str)

    with engine.connect() as conn:
        similar_result = conn.execute(base_query, params)
        similar_df = pd.DataFrame(similar_result.fetchall(), columns=similar_result.keys())

        existing_query = text('SELECT peer_symbol FROM etfs_peers WHERE symbol = :symbol')
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
    insert_df.to_sql('etfs_peers', engine, if_exists='append', index=False, method='multi', chunksize=1000)
    print(f"Added {len(new_rows)} new peer relationships")
else:
    print("No new peers to add.")