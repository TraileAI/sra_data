import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

FMP_API_KEY = "Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT"

def fetch_financial_scores(symbol):
    url = f"https://financialmodelingprep.com/api/v4/score?symbol={symbol}&apikey={FMP_API_KEY}"
    response = requests.get(url)
    response.raise_for_status()
    return symbol, response.json()

if __name__ == "__main__":
    symbols = pd.read_csv('equity_profile.csv')['symbol'].unique().tolist()
    list_of_dfs = []
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(fetch_financial_scores, symbol) for symbol in symbols]
        for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
            symbol, data = future.result()
            if data:
                df = pd.DataFrame(data)
                list_of_dfs.append(df)
            else:
                print(f"No data for {symbol}")
    if list_of_dfs:
        all_data = pd.concat(list_of_dfs, ignore_index=True)
        all_data.to_csv('equity_financial_scores.csv', index=False)