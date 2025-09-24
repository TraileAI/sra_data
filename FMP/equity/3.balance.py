import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def fetch_balance_sheet(symbol):
    url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period=quarter&limit=1000&apikey={FMP_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return symbol, response.json()

if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    symbols = pd.read_sql("SELECT DISTINCT symbol FROM equity_profile;", engine)['symbol'].tolist()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_balance_sheet, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df.to_sql('equity_balance', engine, if_exists='append', index=False)
            else:
                print(f"No data for {symbol}")