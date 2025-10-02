import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
from datetime import date, datetime

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')

def fetch_price_history(symbol, ipo_date_value):
    if ipo_date_value:
        # Handle both date objects and strings
        if isinstance(ipo_date_value, date):
            ipo_date = ipo_date_value
        elif isinstance(ipo_date_value, datetime):
            ipo_date = ipo_date_value.date()
        else:
            ipo_date = datetime.strptime(ipo_date_value, '%Y-%m-%d').date()
    else:
        ipo_date = date(1900, 1, 1)
    end_date = date.today()
    if ipo_date >= end_date:
        return symbol, []
    url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{symbol}?from={ipo_date}&to={end_date}&apikey={FMP_API_KEY}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    json_data = response.json()
    historical = json_data.get('historical', [])
    return symbol, historical

if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    profile_df = pd.read_sql('SELECT symbol, ipodate FROM etfs_profile WHERE ipodate IS NOT NULL;', engine)
    profile_dict = dict(zip(profile_df['symbol'], profile_df['ipodate']))
    symbols = list(profile_dict.keys())
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_price_history, symbol, profile_dict[symbol]): symbol for symbol in symbols}
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Fetching ETF price histories"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df.to_sql('etfs_quotes', engine, if_exists='append', index=False, method='multi')
            else:
                print(f"No data for {symbol}")