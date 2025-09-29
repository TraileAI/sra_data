import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

def fetch_earnings(symbol):
    url = f"https://financialmodelingprep.com/stable/earnings?symbol={symbol}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    data = response.json()
    if data:
        df = pd.DataFrame(data)
        df['symbol'] = symbol
        return df
    return None

if __name__ == "__main__":
    symbols = pd.read_csv('equity_profile.csv')['symbol'].unique().tolist()
    list_of_dfs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_earnings, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            df = future.result()
            if df is not None:
                list_of_dfs.append(df)
    if list_of_dfs:
        all_data = pd.concat(list_of_dfs, ignore_index=True)
        all_data.to_csv('equity_earnings.csv', index=False)