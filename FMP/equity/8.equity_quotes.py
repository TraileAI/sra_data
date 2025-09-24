import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def fetch_price_history(symbol, ipo_date):
    start_date = ipo_date if ipo_date else datetime(1900, 1, 1).date()
    today = datetime.now().date()
    data = []
    current_start = start_date
    while current_start < today:
        current_end = current_start + timedelta(days=5 * 365 + 1)  # Approx 5 years, +1 for leap
        current_end = min(current_end, today)
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={current_start}&to={current_end}&apikey={FMP_API_KEY}"
        response = requests.get(url)
        response.raise_for_status()
        json_data = response.json()
        if 'historical' in json_data:
            data.extend(json_data['historical'])
        current_start = current_end + timedelta(days=1)
    return symbol, data

if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    profile_df = pd.read_sql("SELECT symbol, ipo_date FROM equity_profile;", engine)
    profile_dict = dict(zip(profile_df['symbol'], profile_df['ipo_date']))
    symbols = list(profile_dict.keys())
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_price_history, symbol, profile_dict[symbol]) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df.to_sql('equity_quotes', engine, if_exists='append', index=False)
            else:
                print(f"No data for {symbol}")