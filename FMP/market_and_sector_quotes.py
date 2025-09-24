import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def fetch_price_history(symbol):
    start_date = datetime(1950, 1, 1).date()
    today = datetime.now().date()
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={start_date}&to={today}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    json_data = response.json()
    data = json_data.get('historical', [])
    return symbol, data

if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    symbols = ['XIU.TO', '^GSPC', 'QQQ', 'DIA', 'EURUSD', 'SPY', 'XLK', 'XLV', 'XLF', 'XLY', 'XLI', 'XLB', 'XLE', 'XLU', 'XLRE', 'XLC']
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_price_history, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df.to_sql('market_and_sector_quotes', engine, if_exists='append', index=False)
            else:
                print(f"No data for {symbol}")