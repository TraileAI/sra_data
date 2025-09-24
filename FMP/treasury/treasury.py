import os
import requests
import pandas as pd
from sqlalchemy import create_engine
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

API_KEY = 'Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT'

DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

engine = create_engine(
    f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
)

url = f"https://financialmodelingprep.com/api/v4/treasury?from=1990-01-01&to=2025-09-20&apikey={API_KEY}"
response = requests.get(url)
data = response.json()

df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])
for col in df.columns[1:]:
    df[col] = pd.to_numeric(df[col])

chunksize = 1000
total_chunks = (len(df) // chunksize) + (1 if len(df) % chunksize else 0)

for i in tqdm(range(total_chunks), desc="Inserting treasury data"):
    start = i * chunksize
    end = start + chunksize
    chunk = df.iloc[start:end]
    chunk.to_sql('treasury', engine, if_exists='append', index=False, method='multi')

print(f"Inserted {len(df)} records into treasury table.")