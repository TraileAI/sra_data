import requests
import json
from tqdm import tqdm
import os
from dotenv import load_dotenv
import psycopg2
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

API_KEY = 'Wgpe8YcRGhAYrgJcwtFum4mfqP57DOlT'

db_host = os.getenv('DB_HOST')
db_port = os.getenv('DB_PORT')
db_name = os.getenv('DB_NAME')
db_user = os.getenv('DB_USER')
db_password = os.getenv('DB_PASSWORD')

def process_symbol(symbol):
    try:
        local_conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
        local_cur = local_conn.cursor()
        url = f"https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period=quarter&limit=500&apikey={API_KEY}"
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            for item in data:
                columns = ', '.join(f'"{k}"' for k in item)
                placeholders = ', '.join('%s' for _ in item)
                insert_query = f"""
                INSERT INTO equity_income ({columns}) VALUES ({placeholders})
                ON CONFLICT ("symbol", "date") DO NOTHING
                """
                local_cur.execute(insert_query, tuple(item.values()))
            local_conn.commit()
        else:
            print(f"Error for {symbol}: {response.status_code} - {response.text}")
        local_cur.close()
        local_conn.close()
    except Exception as e:
        print(f"Exception for {symbol}: {e}")

conn = psycopg2.connect(host=db_host, port=db_port, database=db_name, user=db_user, password=db_password)
cur = conn.cursor()

cur.execute("""
CREATE TABLE IF NOT EXISTS equity_income (
    "date" DATE,
    "symbol" VARCHAR(10),
    "reportedCurrency" VARCHAR(3),
    "cik" VARCHAR(10),
    "fillingDate" DATE,
    "acceptedDate" TIMESTAMP,
    "calendarYear" SMALLINT,
    "period" VARCHAR(3),
    "revenue" BIGINT,
    "costOfRevenue" BIGINT,
    "grossProfit" BIGINT,
    "grossProfitRatio" DOUBLE PRECISION,
    "researchAndDevelopmentExpenses" BIGINT,
    "generalAndAdministrativeExpenses" BIGINT,
    "sellingAndMarketingExpenses" BIGINT,
    "sellingGeneralAndAdministrativeExpenses" BIGINT,
    "otherExpenses" BIGINT,
    "operatingExpenses" BIGINT,
    "costAndExpenses" BIGINT,
    "interestIncome" BIGINT,
    "interestExpense" BIGINT,
    "depreciationAndAmortization" BIGINT,
    "ebitda" BIGINT,
    "ebitdaratio" DOUBLE PRECISION,
    "operatingIncome" BIGINT,
    "operatingIncomeRatio" DOUBLE PRECISION,
    "totalOtherIncomeExpensesNet" BIGINT,
    "incomeBeforeTax" BIGINT,
    "incomeBeforeTaxRatio" DOUBLE PRECISION,
    "incomeTaxExpense" BIGINT,
    "netIncome" BIGINT,
    "netIncomeRatio" DOUBLE PRECISION,
    "eps" DOUBLE PRECISION,
    "epsdiluted" DOUBLE PRECISION,
    "weightedAverageShsOut" BIGINT,
    "weightedAverageShsOutDil" BIGINT,
    "link" TEXT,
    "finalLink" TEXT,
    PRIMARY KEY ("symbol", "date")
)
""")
conn.commit()

cur.execute("SELECT DISTINCT symbol FROM equity_profile")
symbols = [row[0] for row in cur.fetchall()]

cur.close()
conn.close()

with ThreadPoolExecutor(max_workers=20) as executor:
    futures = {executor.submit(process_symbol, symbol): symbol for symbol in symbols}
    for future in tqdm(as_completed(futures), total=len(symbols), desc="Processing symbols"):
        pass