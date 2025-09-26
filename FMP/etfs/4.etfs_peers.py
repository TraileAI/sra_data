import os
from dotenv import load_dotenv
import requests
import io
import pandas as pd
from sqlalchemy import create_engine, text, MetaData, Table
from sqlalchemy.dialects.postgresql import insert
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

with engine.connect() as conn:
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS etfs_peers (
            symbol VARCHAR(15) NOT NULL,
            peer_symbol VARCHAR(15) NOT NULL,
            PRIMARY KEY (symbol, peer_symbol)
        );
    """))
    conn.commit()

symbols_df = pd.read_sql("SELECT DISTINCT symbol FROM etfs_profile", engine)
symbols = symbols_df['symbol'].tolist()

api_key = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
url = f"https://financialmodelingprep.com/api/v4/stock_peers_bulk?apikey={api_key}"
response = requests.get(url)
peers_df = pd.read_csv(io.StringIO(response.text))
peers_df = peers_df[peers_df['symbol'].isin(symbols)]

def process_row(row):
    if pd.isna(row['peers']):
        return []
    peers = [p.strip() for p in str(row['peers']).split(',') if p.strip()]
    return [{'symbol': row['symbol'], 'peer_symbol': peer} for peer in peers]

rows = []
with ThreadPoolExecutor(max_workers=10) as executor:
    futures = {executor.submit(process_row, row): row for _, row in peers_df.iterrows()}
    for future in tqdm(as_completed(futures), total=len(futures), desc="Processing rows"):
        rows.extend(future.result())

if rows:
    insert_df = pd.DataFrame(rows).drop_duplicates()
    metadata = MetaData()
    table = Table('etfs_peers', metadata, autoload_with=engine)
    chunksize = 1000
    with engine.connect() as conn:
        for i in tqdm(range(0, len(insert_df), chunksize), desc="Inserting chunks"):
            chunk = insert_df.iloc[i:i+chunksize]
            stmt = insert(table).values(chunk.to_dict('records'))
            stmt = stmt.on_conflict_do_nothing(index_elements=['symbol', 'peer_symbol'])
            conn.execute(stmt)
        conn.commit()