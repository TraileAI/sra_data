import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import json
import time

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"
DB_HOST = os.getenv('DB_HOST')
DB_PORT = os.getenv('DB_PORT')
DB_NAME = os.getenv('DB_NAME')
DB_USER = os.getenv('DB_USER')
DB_PASSWORD = os.getenv('DB_PASSWORD')


def fetch_etf_list(exchange: str, country: str, volume_threshold: int) -> list:
    url = "https://financialmodelingprep.com/api/v3/stock-screener"
    params = {
        'exchange': exchange,
        'isEtf': 'true',
        'isFund': 'false',
        'priceMoreThan': '1',
        'volumeMoreThan': str(volume_threshold),
        'isActivelyTrading': 'true',
        'country': country,
        'limit': '100000',
        'apikey': FMP_API_KEY
    }
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, list):
            raise ValueError(f"Unexpected response format for {country} ETFs")
        return sorted(set(item['symbol'] for item in data))
    except (requests.RequestException, ValueError) as e:
        print(f"Error fetching {country} ETFs: {e}")
        return []


def fetch_combined_profile(symbol: str) -> dict:
    profile_url = f"https://financialmodelingprep.com/api/v3/profile/{symbol}?apikey={FMP_API_KEY}"
    etf_info_url = f"https://financialmodelingprep.com/api/v4/etf-info?symbol={symbol}&apikey={FMP_API_KEY}"
    try:
        profile_resp = requests.get(profile_url, timeout=30)
        profile_resp.raise_for_status()
        profile = profile_resp.json()
        etf_info_resp = requests.get(etf_info_url, timeout=30)
        etf_info_resp.raise_for_status()
        etf_info = etf_info_resp.json()
        if profile and etf_info:
            combined = {**profile[0], **etf_info[0]}
            return combined
        return {}
    except Exception as e:
        print(f"Error fetching profile for {symbol}: {e}")
        return {}


if __name__ == "__main__":
    engine = create_engine(f'postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}')

    us_etfs = fetch_etf_list('nyse,nasdaq,amex', 'US', 10000)
    print(f"Number of US ETFs fetched: {len(us_etfs)}")

    ca_etfs = fetch_etf_list('tsx,tsxv, tmx', 'CA', 5000)
    print(f"Number of CA ETFs fetched: {len(ca_etfs)}")

    filtered_symbols = sorted(set(us_etfs + ca_etfs))
    print(f"Total unique ETF symbols: {len(filtered_symbols)}")

    profiles = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(fetch_combined_profile, symbol) for symbol in filtered_symbols]
        for future in tqdm(as_completed(futures), total=len(filtered_symbols), desc="Fetching profiles"):
            profile = future.result()
            if profile:
                profiles.append(profile)

    if profiles:
        df = pd.DataFrame(profiles)
        df = df[df['price'] > 1]
        df = df[df['mktCap'] > 1000000]
        numeric_fields = ['aum', 'avgVolume', 'expenseRatio', 'nav', 'mktCap']
        for field in numeric_fields:
            if field in df.columns:
                df[field] = pd.to_numeric(df[field], errors='coerce')
        if 'inceptionDate' in df.columns:
            df['inceptionDate'] = pd.to_datetime(df['inceptionDate'], errors='coerce')
        if 'sectorsList' in df.columns:
            df['sectorsList'] = df['sectorsList'].apply(json.dumps)
        print(f"Number of ETFs that match the criteria: {len(df)}")
        df.to_sql('etfs_profile', engine, if_exists='replace', index=False)
        print("Saved ETF profiles to database table 'etfs_profile'")
    else:
        print("No ETF profiles fetched.")