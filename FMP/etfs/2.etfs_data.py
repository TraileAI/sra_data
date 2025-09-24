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

def fetch_holdings(symbol):
    url = f"https://financialmodelingprep.com/stable/etf/holdings?symbol={symbol}&apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            return symbol, response.json()
        else:
            print(f"No holdings data for {symbol}: {response.status_code}")
            return symbol, []
    except Exception as e:
        print(f"Error fetching holdings for {symbol}: {e}")
        return symbol, []

def fetch_country_weightings(symbol):
    url = f"https://financialmodelingprep.com/api/v3/etf-country-weightings/{symbol}?apikey={FMP_API_KEY}"
    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                item['weight_percentage'] = float(item['weightPercentage'].rstrip('%')) if 'weightPercentage' in item else None
                item.pop('weightPercentage', None)
            return symbol, data
        else:
            print(f"No country data for {symbol}: {response.status_code}")
            return symbol, []
    except Exception as e:
        print(f"Error fetching country for {symbol}: {e}")
        return symbol, []

if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')
    symbols = pd.read_sql("SELECT DISTINCT symbol FROM etfs_profile;", engine)['symbol'].tolist()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures_holdings = [executor.submit(fetch_holdings, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures_holdings), total=len(symbols), desc="Processing holdings"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df.to_sql('etfs_holdings', engine, if_exists='append', index=False)
        futures_country = [executor.submit(fetch_country_weightings, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures_country), total=len(symbols), desc="Processing country weightings"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                df['symbol'] = symbol
                df.to_sql('etfs_country_weightings', engine, if_exists='append', index=False)