"""
FMP CSV loader - loads pre-extracted FMP data from CSV files to PostgreSQL.
Fast deployment using PostgreSQL COPY FROM for maximum performance.
"""
import os
import psycopg2
import logging
import tempfile
import csv
import io
from typing import Dict, List
import sys

# Add current directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import config

# Import CSV downloader for Render deployments
try:
    from .download_csv_data import download_csv_files, check_files_exist
except ImportError:
    # Handle case where download_csv_data is not available
    def download_csv_files(force=False):
        logger.warning("CSV download functionality not available")
        return False
    def check_files_exist():
        return []

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database configuration
logger.info("Loading database configuration for FMP CSV loader...")
DB_CONFIG = config.db_config
DB_HOST = DB_CONFIG['host']
DB_PORT = DB_CONFIG['port']
DB_NAME = DB_CONFIG['database']
DB_USER = DB_CONFIG['user']
DB_PASSWORD = DB_CONFIG['password']

logger.info(f"Database config loaded: {DB_HOST}:{DB_PORT}/{DB_NAME} as {DB_USER}")

# FMP CSV file mappings
FMP_CSV_TABLES = {
    'equity_profile.csv': 'equity_profile',
    'equity_income.csv': 'equity_income',
    'equity_balance.csv': 'equity_balance',
    'equity_cash_flow.csv': 'equity_cashflow',
    'equity_earnings.csv': 'equity_earnings',
    'equity_peers.csv': 'equity_peers',
    'equity_ratios.csv': 'equity_financial_ratio',
    'equity_key_metrics.csv': 'equity_key_metrics',
    'equity_balance_growth.csv': 'equity_balance_growth',
    'equity_cashflow_growth.csv': 'equity_cashflow_growth',
    'equity_financial_growth.csv': 'equity_financial_growth',
    'equity_financial_scores.csv': 'equity_financial_scores',
    'equity_income_growth.csv': 'equity_income_growth',
    'etfs_profile.csv': 'etfs_profile',
    'etfs_peers.csv': 'etfs_peers',
    'etfs_data.csv': 'etfs_data',
}

def get_fmp_csv_directory():
    """Get the FMP CSV data directory path."""
    # Get the project root directory (parent of this script's directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(project_root, 'fmp_data')

def create_tables(conn):
    """Create FMP tables if they don't exist."""
    logger.info("Creating FMP tables if needed...")

    with conn.cursor() as cur:
        # Equity profile table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_profile (
                symbol VARCHAR(15) PRIMARY KEY,
                price DOUBLE PRECISION,
                beta DOUBLE PRECISION,
                vol_avg TEXT,
                mkt_cap TEXT,
                last_div DOUBLE PRECISION,
                range_str VARCHAR(50),
                changes DOUBLE PRECISION,
                company_name VARCHAR(255),
                currency VARCHAR(10),
                cik VARCHAR(20),
                isin VARCHAR(20),
                cusip VARCHAR(20),
                exchange VARCHAR(100),
                exchange_short_name VARCHAR(50),
                industry VARCHAR(100),
                website VARCHAR(255),
                description TEXT,
                ceo VARCHAR(100),
                sector VARCHAR(100),
                country VARCHAR(50),
                full_time_employees TEXT,
                phone VARCHAR(50),
                address VARCHAR(255),
                city VARCHAR(100),
                state VARCHAR(50),
                zip_code VARCHAR(20),
                dcf_diff DOUBLE PRECISION,
                dcf DOUBLE PRECISION,
                image VARCHAR(255),
                ipo_date DATE,
                default_image BOOLEAN,
                is_etf BOOLEAN,
                is_actively_trading BOOLEAN,
                is_adr BOOLEAN,
                is_fund BOOLEAN
            )
        """)

        # Equity income table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_income (
                date DATE,
                symbol VARCHAR(15),
                reportedCurrency VARCHAR(3),
                cik VARCHAR(20),
                fillingDate DATE,
                acceptedDate TIMESTAMP,
                calendarYear SMALLINT,
                period VARCHAR(3),
                revenue NUMERIC,
                costOfRevenue NUMERIC,
                grossProfit NUMERIC,
                grossProfitRatio DOUBLE PRECISION,
                researchAndDevelopmentExpenses NUMERIC,
                generalAndAdministrativeExpenses NUMERIC,
                sellingAndMarketingExpenses NUMERIC,
                sellingGeneralAndAdministrativeExpenses NUMERIC,
                otherExpenses NUMERIC,
                operatingExpenses NUMERIC,
                costAndExpenses NUMERIC,
                interestIncome NUMERIC,
                interestExpense NUMERIC,
                depreciationAndAmortization NUMERIC,
                ebitda NUMERIC,
                ebitdaratio DOUBLE PRECISION,
                operatingIncome NUMERIC,
                operatingIncomeRatio DOUBLE PRECISION,
                totalOtherIncomeExpensesNet NUMERIC,
                incomeBeforeTax NUMERIC,
                incomeBeforeTaxRatio DOUBLE PRECISION,
                incomeTaxExpense NUMERIC,
                netIncome NUMERIC,
                netIncomeRatio DOUBLE PRECISION,
                eps DOUBLE PRECISION,
                epsdiluted DOUBLE PRECISION,
                weightedAverageShsOut NUMERIC,
                weightedAverageShsOutDil NUMERIC,
                link TEXT,
                finalLink TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity balance table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_balance (
                date DATE,
                symbol VARCHAR(15),
                reportedCurrency VARCHAR(3),
                cik VARCHAR(20),
                fillingDate DATE,
                acceptedDate TIMESTAMP,
                calendarYear SMALLINT,
                period VARCHAR(3),
                cashAndCashEquivalents NUMERIC,
                shortTermInvestments NUMERIC,
                cashAndShortTermInvestments NUMERIC,
                netReceivables NUMERIC,
                inventory NUMERIC,
                otherCurrentAssets NUMERIC,
                totalCurrentAssets NUMERIC,
                propertyPlantEquipmentNet NUMERIC,
                goodwill NUMERIC,
                intangibleAssets NUMERIC,
                goodwillAndIntangibleAssets NUMERIC,
                longTermInvestments NUMERIC,
                taxAssets NUMERIC,
                otherNonCurrentAssets NUMERIC,
                totalNonCurrentAssets NUMERIC,
                otherAssets NUMERIC,
                totalAssets NUMERIC,
                accountPayables NUMERIC,
                shortTermDebt NUMERIC,
                taxPayables NUMERIC,
                deferredRevenue NUMERIC,
                otherCurrentLiabilities NUMERIC,
                totalCurrentLiabilities NUMERIC,
                longTermDebt NUMERIC,
                deferredRevenueNonCurrent NUMERIC,
                deferredTaxLiabilitiesNonCurrent NUMERIC,
                otherNonCurrentLiabilities NUMERIC,
                totalNonCurrentLiabilities NUMERIC,
                otherLiabilities NUMERIC,
                capitalLeaseObligations NUMERIC,
                totalLiabilities NUMERIC,
                preferredStock NUMERIC,
                commonStock NUMERIC,
                retainedEarnings NUMERIC,
                accumulatedOtherComprehensiveIncomeLoss NUMERIC,
                othertotalStockholdersEquity NUMERIC,
                totalStockholdersEquity NUMERIC,
                totalEquity NUMERIC,
                totalLiabilitiesAndStockholdersEquity NUMERIC,
                minorityInterest NUMERIC,
                totalLiabilitiesAndTotalEquity NUMERIC,
                totalInvestments NUMERIC,
                totalDebt NUMERIC,
                netDebt NUMERIC,
                link TEXT,
                finalLink TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity cashflow table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_cashflow (
                date DATE,
                symbol VARCHAR(15),
                reportedCurrency VARCHAR(3),
                cik VARCHAR(20),
                fillingDate DATE,
                acceptedDate TIMESTAMP,
                calendarYear SMALLINT,
                period VARCHAR(3),
                netIncome NUMERIC,
                depreciationAndAmortization NUMERIC,
                deferredIncomeTax NUMERIC,
                stockBasedCompensation NUMERIC,
                changeInWorkingCapital NUMERIC,
                accountsReceivables NUMERIC,
                inventory NUMERIC,
                accountsPayables NUMERIC,
                otherWorkingCapital NUMERIC,
                otherNonCashItems NUMERIC,
                netCashProvidedByOperatingActivities NUMERIC,
                investmentsInPropertyPlantAndEquipment NUMERIC,
                acquisitionsNet NUMERIC,
                purchasesOfInvestments NUMERIC,
                salesMaturitiesOfInvestments NUMERIC,
                otherInvestingActivites NUMERIC,
                netCashUsedForInvestingActivites NUMERIC,
                debtRepayment NUMERIC,
                commonStockIssued NUMERIC,
                commonStockRepurchased NUMERIC,
                dividendsPaid NUMERIC,
                otherFinancingActivites NUMERIC,
                netCashUsedProvidedByFinancingActivities NUMERIC,
                effectOfForexChangesOnCash NUMERIC,
                netChangeInCash NUMERIC,
                cashAtEndOfPeriod NUMERIC,
                cashAtBeginningOfPeriod NUMERIC,
                operatingCashFlow NUMERIC,
                capitalExpenditure NUMERIC,
                freeCashFlow NUMERIC,
                link TEXT,
                finalLink TEXT,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity earnings table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_earnings (
                symbol VARCHAR(15),
                date DATE,
                epsActual DOUBLE PRECISION,
                epsEstimated DOUBLE PRECISION,
                revenueActual NUMERIC,
                revenueEstimated NUMERIC,
                lastUpdated TIMESTAMP,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity financial ratios table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_financial_ratio (
                symbol VARCHAR(15),
                date DATE,
                calendarYear SMALLINT,
                period VARCHAR(3),
                currentRatio DOUBLE PRECISION,
                quickRatio DOUBLE PRECISION,
                cashRatio DOUBLE PRECISION,
                daysOfSalesOutstanding DOUBLE PRECISION,
                daysOfInventoryOutstanding DOUBLE PRECISION,
                operatingCycle DOUBLE PRECISION,
                daysOfPayablesOutstanding DOUBLE PRECISION,
                cashConversionCycle DOUBLE PRECISION,
                grossProfitMargin DOUBLE PRECISION,
                operatingProfitMargin DOUBLE PRECISION,
                pretaxProfitMargin DOUBLE PRECISION,
                netProfitMargin DOUBLE PRECISION,
                effectiveTaxRate DOUBLE PRECISION,
                returnOnAssets DOUBLE PRECISION,
                returnOnEquity DOUBLE PRECISION,
                returnOnCapitalEmployed DOUBLE PRECISION,
                netIncomePerEBT DOUBLE PRECISION,
                ebtPerEbit DOUBLE PRECISION,
                ebitPerRevenue DOUBLE PRECISION,
                debtRatio DOUBLE PRECISION,
                debtEquityRatio DOUBLE PRECISION,
                longTermDebtToCapitalization DOUBLE PRECISION,
                totalDebtToCapitalization DOUBLE PRECISION,
                interestCoverage DOUBLE PRECISION,
                cashFlowToDebtRatio DOUBLE PRECISION,
                companyEquityMultiplier DOUBLE PRECISION,
                receivablesTurnover DOUBLE PRECISION,
                payablesTurnover DOUBLE PRECISION,
                inventoryTurnover DOUBLE PRECISION,
                fixedAssetTurnover DOUBLE PRECISION,
                assetTurnover DOUBLE PRECISION,
                operatingCashFlowPerShare DOUBLE PRECISION,
                freeCashFlowPerShare DOUBLE PRECISION,
                cashPerShare DOUBLE PRECISION,
                payoutRatio DOUBLE PRECISION,
                operatingCashFlowSalesRatio DOUBLE PRECISION,
                freeCashFlowOperatingCashFlowRatio DOUBLE PRECISION,
                cashFlowCoverageRatios DOUBLE PRECISION,
                shortTermCoverageRatios DOUBLE PRECISION,
                capitalExpenditureCoverageRatio DOUBLE PRECISION,
                dividendPaidAndCapexCoverageRatio DOUBLE PRECISION,
                dividendPayoutRatio DOUBLE PRECISION,
                priceBookValueRatio DOUBLE PRECISION,
                priceToBookRatio DOUBLE PRECISION,
                priceToSalesRatio DOUBLE PRECISION,
                priceEarningsRatio DOUBLE PRECISION,
                priceToFreeCashFlowsRatio DOUBLE PRECISION,
                priceToOperatingCashFlowsRatio DOUBLE PRECISION,
                priceCashFlowRatio DOUBLE PRECISION,
                priceEarningsToGrowthRatio DOUBLE PRECISION,
                priceSalesRatio DOUBLE PRECISION,
                dividendYield DOUBLE PRECISION,
                enterpriseValueMultiple DOUBLE PRECISION,
                priceFairValue DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity key metrics table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_key_metrics (
                symbol VARCHAR(15),
                date DATE,
                calendarYear SMALLINT,
                period VARCHAR(3),
                revenuePerShare DOUBLE PRECISION,
                netIncomePerShare DOUBLE PRECISION,
                operatingCashFlowPerShare DOUBLE PRECISION,
                freeCashFlowPerShare DOUBLE PRECISION,
                cashPerShare DOUBLE PRECISION,
                bookValuePerShare DOUBLE PRECISION,
                tangibleBookValuePerShare DOUBLE PRECISION,
                shareholdersEquityPerShare DOUBLE PRECISION,
                interestDebtPerShare DOUBLE PRECISION,
                marketCap NUMERIC,
                enterpriseValue NUMERIC,
                peRatio DOUBLE PRECISION,
                priceToSalesRatio DOUBLE PRECISION,
                pocfratio DOUBLE PRECISION,
                pfcfRatio DOUBLE PRECISION,
                pbRatio DOUBLE PRECISION,
                ptbRatio DOUBLE PRECISION,
                evToSales DOUBLE PRECISION,
                enterpriseValueOverEBITDA DOUBLE PRECISION,
                evToOperatingCashFlow DOUBLE PRECISION,
                evToFreeCashFlow DOUBLE PRECISION,
                earningsYield DOUBLE PRECISION,
                freeCashFlowYield DOUBLE PRECISION,
                debtToEquity DOUBLE PRECISION,
                debtToAssets DOUBLE PRECISION,
                netDebtToEBITDA DOUBLE PRECISION,
                currentRatio DOUBLE PRECISION,
                interestCoverage DOUBLE PRECISION,
                incomeQuality DOUBLE PRECISION,
                dividendYield DOUBLE PRECISION,
                payoutRatio DOUBLE PRECISION,
                salesGeneralAndAdministrativeToRevenue DOUBLE PRECISION,
                researchAndDevelopmentToRevenue DOUBLE PRECISION,
                intangiblesToTotalAssets DOUBLE PRECISION,
                capexToOperatingCashFlow DOUBLE PRECISION,
                capexToRevenue DOUBLE PRECISION,
                capexToDepreciation DOUBLE PRECISION,
                stockBasedCompensationToRevenue DOUBLE PRECISION,
                grahamNumber DOUBLE PRECISION,
                roic DOUBLE PRECISION,
                returnOnTangibleAssets DOUBLE PRECISION,
                grahamNetNet NUMERIC,
                workingCapital NUMERIC,
                tangibleAssetValue NUMERIC,
                netCurrentAssetValue NUMERIC,
                investedCapital NUMERIC,
                averageReceivables NUMERIC,
                averagePayables NUMERIC,
                averageInventory NUMERIC,
                daysSalesOutstanding DOUBLE PRECISION,
                daysPayablesOutstanding DOUBLE PRECISION,
                daysOfInventoryOnHand DOUBLE PRECISION,
                receivablesTurnover DOUBLE PRECISION,
                payablesTurnover DOUBLE PRECISION,
                inventoryTurnover DOUBLE PRECISION,
                roe DOUBLE PRECISION,
                capexPerShare DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # ETFs profile table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_profile (
                symbol VARCHAR(15) PRIMARY KEY,
                price DOUBLE PRECISION,
                beta DOUBLE PRECISION,
                volAvg DOUBLE PRECISION,
                mktCap TEXT,
                lastDiv DOUBLE PRECISION,
                range VARCHAR(50),
                changes DOUBLE PRECISION,
                companyName VARCHAR(255),
                currency VARCHAR(10),
                cik VARCHAR(20),
                isin VARCHAR(20),
                cusip VARCHAR(20),
                exchange VARCHAR(100),
                exchangeShortName VARCHAR(50),
                industry VARCHAR(100),
                website VARCHAR(255),
                description TEXT,
                ceo VARCHAR(100),
                sector VARCHAR(100),
                country VARCHAR(50),
                fullTimeEmployees TEXT,
                phone VARCHAR(50),
                address VARCHAR(255),
                city VARCHAR(100),
                state VARCHAR(50),
                zip VARCHAR(20),
                dcfDiff DOUBLE PRECISION,
                dcf DOUBLE PRECISION,
                image VARCHAR(255),
                ipoDate DATE,
                defaultImage BOOLEAN,
                isEtf BOOLEAN,
                isActivelyTrading BOOLEAN,
                isAdr BOOLEAN,
                isFund BOOLEAN,
                assetClass VARCHAR(50),
                aum NUMERIC,
                avgVolume DOUBLE PRECISION,
                domicile VARCHAR(50),
                etfCompany VARCHAR(100),
                expenseRatio DOUBLE PRECISION,
                inceptionDate DATE,
                name VARCHAR(255),
                nav DOUBLE PRECISION,
                navCurrency VARCHAR(10),
                sectorsList TEXT,
                holdingsCount DOUBLE PRECISION
            )
        """)

        # ETFs data table - match CSV headers exactly
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_data (
                symbol VARCHAR(15),
                asset VARCHAR(100),
                name VARCHAR(255),
                isin VARCHAR(20),
                securityCusip VARCHAR(50),
                sharesNumber NUMERIC,
                weightPercentage DOUBLE PRECISION,
                marketValue NUMERIC,
                updatedAt TIMESTAMP,
                etf_symbol VARCHAR(15),
                data_type VARCHAR(50),
                country VARCHAR(50),
                weight_percentage DOUBLE PRECISION
            )
        """)

        # Equity quotes table - use TEXT for volume columns to handle decimal strings
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_quotes (
                date DATE,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                adjClose DOUBLE PRECISION,
                volume TEXT,
                unadjustedVolume TEXT,
                change DOUBLE PRECISION,
                changePercent DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                label VARCHAR(50),
                changeOverTime DOUBLE PRECISION,
                symbol VARCHAR(15),
                PRIMARY KEY (symbol, date)
            )
        """)

        # ETFs quotes table - use TEXT for volume columns to handle decimal strings
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_quotes (
                date DATE,
                open DOUBLE PRECISION,
                high DOUBLE PRECISION,
                low DOUBLE PRECISION,
                close DOUBLE PRECISION,
                adjClose DOUBLE PRECISION,
                volume TEXT,
                unadjustedVolume TEXT,
                change DOUBLE PRECISION,
                changePercent DOUBLE PRECISION,
                vwap DOUBLE PRECISION,
                label VARCHAR(50),
                changeOverTime DOUBLE PRECISION,
                symbol VARCHAR(15),
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity peers table - fix NOT NULL constraint issue
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_peers (
                symbol VARCHAR(15) NOT NULL,
                peer_symbol VARCHAR(15),
                PRIMARY KEY (symbol, peer_symbol)
            )
        """)

        # ETFs peers table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS etfs_peers (
                symbol VARCHAR(15) NOT NULL,
                peer_symbol VARCHAR(15),
                PRIMARY KEY (symbol, peer_symbol)
            )
        """)

        # Equity balance growth table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_balance_growth (
                date DATE,
                symbol VARCHAR(15),
                calendarYear SMALLINT,
                period VARCHAR(3),
                growthCashAndCashEquivalents DOUBLE PRECISION,
                growthShortTermInvestments DOUBLE PRECISION,
                growthCashAndShortTermInvestments DOUBLE PRECISION,
                growthNetReceivables DOUBLE PRECISION,
                growthInventory DOUBLE PRECISION,
                growthOtherCurrentAssets DOUBLE PRECISION,
                growthTotalCurrentAssets DOUBLE PRECISION,
                growthPropertyPlantEquipmentNet DOUBLE PRECISION,
                growthGoodwill DOUBLE PRECISION,
                growthIntangibleAssets DOUBLE PRECISION,
                growthGoodwillAndIntangibleAssets DOUBLE PRECISION,
                growthLongTermInvestments DOUBLE PRECISION,
                growthTaxAssets DOUBLE PRECISION,
                growthOtherNonCurrentAssets DOUBLE PRECISION,
                growthTotalNonCurrentAssets DOUBLE PRECISION,
                growthOtherAssets DOUBLE PRECISION,
                growthTotalAssets DOUBLE PRECISION,
                growthAccountPayables DOUBLE PRECISION,
                growthShortTermDebt DOUBLE PRECISION,
                growthTaxPayables DOUBLE PRECISION,
                growthDeferredRevenue DOUBLE PRECISION,
                growthOtherCurrentLiabilities DOUBLE PRECISION,
                growthTotalCurrentLiabilities DOUBLE PRECISION,
                growthLongTermDebt DOUBLE PRECISION,
                growthDeferredRevenueNonCurrent DOUBLE PRECISION,
                growthDeferrredTaxLiabilitiesNonCurrent DOUBLE PRECISION,
                growthOtherNonCurrentLiabilities DOUBLE PRECISION,
                growthTotalNonCurrentLiabilities DOUBLE PRECISION,
                growthOtherLiabilities DOUBLE PRECISION,
                growthTotalLiabilities DOUBLE PRECISION,
                growthCommonStock DOUBLE PRECISION,
                growthRetainedEarnings DOUBLE PRECISION,
                growthAccumulatedOtherComprehensiveIncomeLoss DOUBLE PRECISION,
                growthOthertotalStockholdersEquity DOUBLE PRECISION,
                growthTotalStockholdersEquity DOUBLE PRECISION,
                growthTotalLiabilitiesAndStockholdersEquity DOUBLE PRECISION,
                growthTotalInvestments DOUBLE PRECISION,
                growthTotalDebt DOUBLE PRECISION,
                growthNetDebt DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity cashflow growth table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_cashflow_growth (
                date DATE,
                symbol VARCHAR(15),
                calendarYear SMALLINT,
                period VARCHAR(3),
                growthNetIncome DOUBLE PRECISION,
                growthDepreciationAndAmortization DOUBLE PRECISION,
                growthDeferredIncomeTax DOUBLE PRECISION,
                growthStockBasedCompensation DOUBLE PRECISION,
                growthChangeInWorkingCapital DOUBLE PRECISION,
                growthAccountsReceivables DOUBLE PRECISION,
                growthInventory DOUBLE PRECISION,
                growthAccountsPayables DOUBLE PRECISION,
                growthOtherWorkingCapital DOUBLE PRECISION,
                growthOtherNonCashItems DOUBLE PRECISION,
                growthNetCashProvidedByOperatingActivites DOUBLE PRECISION,
                growthInvestmentsInPropertyPlantAndEquipment DOUBLE PRECISION,
                growthAcquisitionsNet DOUBLE PRECISION,
                growthPurchasesOfInvestments DOUBLE PRECISION,
                growthSalesMaturitiesOfInvestments DOUBLE PRECISION,
                growthOtherInvestingActivites DOUBLE PRECISION,
                growthNetCashUsedForInvestingActivites DOUBLE PRECISION,
                growthDebtRepayment DOUBLE PRECISION,
                growthCommonStockIssued DOUBLE PRECISION,
                growthCommonStockRepurchased DOUBLE PRECISION,
                growthDividendsPaid DOUBLE PRECISION,
                growthOtherFinancingActivites DOUBLE PRECISION,
                growthNetCashUsedProvidedByFinancingActivities DOUBLE PRECISION,
                growthEffectOfForexChangesOnCash DOUBLE PRECISION,
                growthNetChangeInCash DOUBLE PRECISION,
                growthCashAtEndOfPeriod DOUBLE PRECISION,
                growthCashAtBeginningOfPeriod DOUBLE PRECISION,
                growthOperatingCashFlow DOUBLE PRECISION,
                growthCapitalExpenditure DOUBLE PRECISION,
                growthFreeCashFlow DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity financial growth table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_financial_growth (
                symbol VARCHAR(15),
                date DATE,
                calendarYear SMALLINT,
                period VARCHAR(3),
                revenueGrowth DOUBLE PRECISION,
                grossProfitGrowth DOUBLE PRECISION,
                ebitgrowth DOUBLE PRECISION,
                operatingIncomeGrowth DOUBLE PRECISION,
                netIncomeGrowth DOUBLE PRECISION,
                epsgrowth DOUBLE PRECISION,
                epsdilutedGrowth DOUBLE PRECISION,
                weightedAverageSharesGrowth DOUBLE PRECISION,
                weightedAverageSharesDilutedGrowth DOUBLE PRECISION,
                dividendsperShareGrowth DOUBLE PRECISION,
                operatingCashFlowGrowth DOUBLE PRECISION,
                freeCashFlowGrowth DOUBLE PRECISION,
                tenYRevenueGrowthPerShare DOUBLE PRECISION,
                fiveYRevenueGrowthPerShare DOUBLE PRECISION,
                threeYRevenueGrowthPerShare DOUBLE PRECISION,
                tenYOperatingCFGrowthPerShare DOUBLE PRECISION,
                fiveYOperatingCFGrowthPerShare DOUBLE PRECISION,
                threeYOperatingCFGrowthPerShare DOUBLE PRECISION,
                tenYNetIncomeGrowthPerShare DOUBLE PRECISION,
                fiveYNetIncomeGrowthPerShare DOUBLE PRECISION,
                threeYNetIncomeGrowthPerShare DOUBLE PRECISION,
                tenYShareholdersEquityGrowthPerShare DOUBLE PRECISION,
                fiveYShareholdersEquityGrowthPerShare DOUBLE PRECISION,
                threeYShareholdersEquityGrowthPerShare DOUBLE PRECISION,
                tenYDividendperShareGrowthPerShare DOUBLE PRECISION,
                fiveYDividendperShareGrowthPerShare DOUBLE PRECISION,
                threeYDividendperShareGrowthPerShare DOUBLE PRECISION,
                receivablesGrowth DOUBLE PRECISION,
                inventoryGrowth DOUBLE PRECISION,
                assetGrowth DOUBLE PRECISION,
                bookValueperShareGrowth DOUBLE PRECISION,
                debtGrowth DOUBLE PRECISION,
                rdexpenseGrowth DOUBLE PRECISION,
                sgaexpensesGrowth DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        # Equity financial scores table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_financial_scores (
                symbol VARCHAR(15) PRIMARY KEY,
                altmanZScore DOUBLE PRECISION,
                piotroskiScore DOUBLE PRECISION,
                workingCapital DOUBLE PRECISION,
                totalAssets DOUBLE PRECISION,
                retainedEarnings DOUBLE PRECISION,
                ebit DOUBLE PRECISION,
                marketCap DOUBLE PRECISION,
                totalLiabilities DOUBLE PRECISION,
                revenue DOUBLE PRECISION
            )
        """)

        # Equity income growth table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS equity_income_growth (
                date DATE,
                symbol VARCHAR(15),
                calendarYear SMALLINT,
                period VARCHAR(3),
                growthRevenue DOUBLE PRECISION,
                growthCostOfRevenue DOUBLE PRECISION,
                growthGrossProfit DOUBLE PRECISION,
                growthGrossProfitRatio DOUBLE PRECISION,
                growthResearchAndDevelopmentExpenses DOUBLE PRECISION,
                growthGeneralAndAdministrativeExpenses DOUBLE PRECISION,
                growthSellingAndMarketingExpenses DOUBLE PRECISION,
                growthOtherExpenses DOUBLE PRECISION,
                growthOperatingExpenses DOUBLE PRECISION,
                growthCostAndExpenses DOUBLE PRECISION,
                growthInterestExpense DOUBLE PRECISION,
                growthDepreciationAndAmortization DOUBLE PRECISION,
                growthEBITDA DOUBLE PRECISION,
                growthEBITDARatio DOUBLE PRECISION,
                growthOperatingIncome DOUBLE PRECISION,
                growthOperatingIncomeRatio DOUBLE PRECISION,
                growthTotalOtherIncomeExpensesNet DOUBLE PRECISION,
                growthIncomeBeforeTax DOUBLE PRECISION,
                growthIncomeBeforeTaxRatio DOUBLE PRECISION,
                growthIncomeTaxExpense DOUBLE PRECISION,
                growthNetIncome DOUBLE PRECISION,
                growthNetIncomeRatio DOUBLE PRECISION,
                growthEPS DOUBLE PRECISION,
                growthEPSDiluted DOUBLE PRECISION,
                growthWeightedAverageShsOut DOUBLE PRECISION,
                growthWeightedAverageShsOutDil DOUBLE PRECISION,
                PRIMARY KEY (symbol, date)
            )
        """)

        conn.commit()
        logger.info("Tables created successfully")

def load_csv_to_table(conn, csv_file: str, table_name: str) -> bool:
    """Load a CSV file to PostgreSQL table using COPY FROM."""
    csv_path = os.path.join(get_fmp_csv_directory(), csv_file)

    if not os.path.exists(csv_path):
        logger.warning(f"CSV file not found: {csv_path}")
        return False

    logger.info(f"Loading {csv_file} to {table_name}...")

    try:
        with conn.cursor() as cur:
            # Special handling for tables with potential duplicate key issues
            if table_name in ['equity_peers', 'etfs_peers']:
                # Load into temp table first, then filter/handle duplicates
                # Create temp table without NOT NULL constraint on peer_symbol to allow loading empty values
                if table_name == 'equity_peers':
                    cur.execute("""
                        CREATE TEMP TABLE temp_equity_peers (
                            symbol VARCHAR(15),
                            peer_symbol VARCHAR(15)
                        )
                    """)
                else:  # etfs_peers
                    cur.execute("""
                        CREATE TEMP TABLE temp_etfs_peers (
                            symbol VARCHAR(15),
                            peer_symbol VARCHAR(15)
                        )
                    """)
                copy_sql = f"""
                    COPY temp_{table_name} FROM STDIN
                    WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                with open(csv_path, 'r', encoding='utf-8') as f:
                    cur.copy_expert(copy_sql, f)

                # Copy only rows with non-empty peer_symbol, handling duplicates
                cur.execute(f"""
                    INSERT INTO {table_name}
                    SELECT DISTINCT * FROM temp_{table_name}
                    WHERE peer_symbol IS NOT NULL AND peer_symbol != ''
                    ON CONFLICT DO NOTHING
                """)
                cur.execute(f"DROP TABLE temp_{table_name}")
            elif table_name == 'equity_income':
                # Special handling for equity_income to fix bigint casting issues with decimal values
                cur.execute("""
                    CREATE TEMP TABLE temp_equity_income (
                        date TEXT,
                        symbol TEXT,
                        reportedCurrency TEXT,
                        cik TEXT,
                        fillingDate TEXT,
                        acceptedDate TEXT,
                        calendarYear TEXT,
                        period TEXT,
                        revenue TEXT,
                        costOfRevenue TEXT,
                        grossProfit TEXT,
                        grossProfitRatio TEXT,
                        researchAndDevelopmentExpenses TEXT,
                        generalAndAdministrativeExpenses TEXT,
                        sellingAndMarketingExpenses TEXT,
                        sellingGeneralAndAdministrativeExpenses TEXT,
                        otherExpenses TEXT,
                        operatingExpenses TEXT,
                        costAndExpenses TEXT,
                        interestIncome TEXT,
                        interestExpense TEXT,
                        depreciationAndAmortization TEXT,
                        ebitda TEXT,
                        ebitdaratio TEXT,
                        operatingIncome TEXT,
                        operatingIncomeRatio TEXT,
                        totalOtherIncomeExpensesNet TEXT,
                        incomeBeforeTax TEXT,
                        incomeBeforeTaxRatio TEXT,
                        incomeTaxExpense TEXT,
                        netIncome TEXT,
                        netIncomeRatio TEXT,
                        eps TEXT,
                        epsdiluted TEXT,
                        weightedAverageShsOut TEXT,
                        weightedAverageShsOutDil TEXT,
                        link TEXT,
                        finalLink TEXT
                    )
                """)
                copy_sql = """
                    COPY temp_equity_income FROM STDIN
                    WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                with open(csv_path, 'r', encoding='utf-8') as f:
                    cur.copy_expert(copy_sql, f)

                # Insert with proper type casting for all numeric fields
                cur.execute("""
                    INSERT INTO equity_income (
                        date, symbol, reportedcurrency, cik, fillingdate, accepteddate, calendaryear, period,
                        revenue, costofrevenue, grossprofit, grossprofitratio, researchanddevelopmentexpenses,
                        generalandadministrativeexpenses, sellingandmarketingexpenses, sellinggeneralandadministrativeexpenses,
                        otherexpenses, operatingexpenses, costandexpenses, interestincome, interestexpense,
                        depreciationandamortization, ebitda, ebitdaratio, operatingincome, operatingincomeratio,
                        totalotherincomeexpensesnet, incomebeforetax, incomebeforetaxratio, incometaxexpense,
                        netincome, netincomeratio, eps, epsdiluted, weightedaverageshsout, weightedaverageshsoutdil,
                        link, finallink
                    )
                    SELECT DISTINCT ON (symbol, date)
                        CASE WHEN date = '' OR date IS NULL THEN NULL ELSE date::date END,
                        CASE WHEN symbol = '' OR symbol IS NULL THEN NULL ELSE symbol END,
                        CASE WHEN reportedCurrency = '' OR reportedCurrency IS NULL THEN NULL ELSE reportedCurrency END,
                        CASE WHEN cik = '' OR cik IS NULL THEN NULL ELSE cik END,
                        CASE WHEN fillingDate = '' OR fillingDate IS NULL THEN NULL ELSE fillingDate::date END,
                        CASE WHEN acceptedDate = '' OR acceptedDate IS NULL THEN NULL ELSE acceptedDate::timestamp END,
                        CASE WHEN calendarYear = '' OR calendarYear IS NULL THEN NULL ELSE calendarYear::smallint END,
                        CASE WHEN period = '' OR period IS NULL THEN NULL ELSE period END,
                        CASE WHEN revenue = '' OR revenue IS NULL THEN NULL ELSE revenue::numeric END,
                        CASE WHEN costOfRevenue = '' OR costOfRevenue IS NULL THEN NULL ELSE costOfRevenue::numeric END,
                        CASE WHEN grossProfit = '' OR grossProfit IS NULL THEN NULL ELSE grossProfit::numeric END,
                        CASE WHEN grossProfitRatio = '' OR grossProfitRatio IS NULL THEN NULL ELSE grossProfitRatio::double precision END,
                        CASE WHEN researchAndDevelopmentExpenses = '' OR researchAndDevelopmentExpenses IS NULL THEN NULL ELSE researchAndDevelopmentExpenses::numeric END,
                        CASE WHEN generalAndAdministrativeExpenses = '' OR generalAndAdministrativeExpenses IS NULL THEN NULL ELSE generalAndAdministrativeExpenses::numeric END,
                        CASE WHEN sellingAndMarketingExpenses = '' OR sellingAndMarketingExpenses IS NULL THEN NULL ELSE sellingAndMarketingExpenses::numeric END,
                        CASE WHEN sellingGeneralAndAdministrativeExpenses = '' OR sellingGeneralAndAdministrativeExpenses IS NULL THEN NULL ELSE sellingGeneralAndAdministrativeExpenses::numeric END,
                        CASE WHEN otherExpenses = '' OR otherExpenses IS NULL THEN NULL ELSE otherExpenses::numeric END,
                        CASE WHEN operatingExpenses = '' OR operatingExpenses IS NULL THEN NULL ELSE operatingExpenses::numeric END,
                        CASE WHEN costAndExpenses = '' OR costAndExpenses IS NULL THEN NULL ELSE costAndExpenses::numeric END,
                        CASE WHEN interestIncome = '' OR interestIncome IS NULL THEN NULL ELSE interestIncome::numeric END,
                        CASE WHEN interestExpense = '' OR interestExpense IS NULL THEN NULL ELSE interestExpense::numeric END,
                        CASE WHEN depreciationAndAmortization = '' OR depreciationAndAmortization IS NULL THEN NULL ELSE depreciationAndAmortization::numeric END,
                        CASE WHEN ebitda = '' OR ebitda IS NULL THEN NULL ELSE ebitda::numeric END,
                        CASE WHEN ebitdaratio = '' OR ebitdaratio IS NULL THEN NULL ELSE ebitdaratio::double precision END,
                        CASE WHEN operatingIncome = '' OR operatingIncome IS NULL THEN NULL ELSE operatingIncome::numeric END,
                        CASE WHEN operatingIncomeRatio = '' OR operatingIncomeRatio IS NULL THEN NULL ELSE operatingIncomeRatio::double precision END,
                        CASE WHEN totalOtherIncomeExpensesNet = '' OR totalOtherIncomeExpensesNet IS NULL THEN NULL ELSE totalOtherIncomeExpensesNet::numeric END,
                        CASE WHEN incomeBeforeTax = '' OR incomeBeforeTax IS NULL THEN NULL ELSE incomeBeforeTax::numeric END,
                        CASE WHEN incomeBeforeTaxRatio = '' OR incomeBeforeTaxRatio IS NULL THEN NULL ELSE incomeBeforeTaxRatio::double precision END,
                        CASE WHEN incomeTaxExpense = '' OR incomeTaxExpense IS NULL THEN NULL ELSE incomeTaxExpense::numeric END,
                        CASE WHEN netIncome = '' OR netIncome IS NULL THEN NULL ELSE netIncome::numeric END,
                        CASE WHEN netIncomeRatio = '' OR netIncomeRatio IS NULL THEN NULL ELSE netIncomeRatio::double precision END,
                        CASE WHEN eps = '' OR eps IS NULL THEN NULL ELSE eps::double precision END,
                        CASE WHEN epsdiluted = '' OR epsdiluted IS NULL THEN NULL ELSE epsdiluted::double precision END,
                        CASE WHEN weightedAverageShsOut = '' OR weightedAverageShsOut IS NULL THEN NULL ELSE weightedAverageShsOut::numeric END,
                        CASE WHEN weightedAverageShsOutDil = '' OR weightedAverageShsOutDil IS NULL THEN NULL ELSE weightedAverageShsOutDil::numeric END,
                        CASE WHEN link = '' OR link IS NULL THEN NULL ELSE link END,
                        CASE WHEN finalLink = '' OR finalLink IS NULL THEN NULL ELSE finalLink END
                    FROM temp_equity_income
                    WHERE symbol IS NOT NULL AND symbol != ''
                    ORDER BY symbol, date
                    ON CONFLICT (symbol, date) DO NOTHING
                """)
                cur.execute("DROP TABLE temp_equity_income")
            elif table_name == 'equity_key_metrics':
                # Special handling for equity_key_metrics to fix CSV structure issues
                cur.execute("""
                    CREATE TEMP TABLE temp_equity_key_metrics (
                        symbol TEXT,
                        date TEXT,
                        calendarYear TEXT,
                        period TEXT,
                        revenuePerShare TEXT,
                        netIncomePerShare TEXT,
                        operatingCashFlowPerShare TEXT,
                        freeCashFlowPerShare TEXT,
                        cashPerShare TEXT,
                        bookValuePerShare TEXT,
                        tangibleBookValuePerShare TEXT,
                        shareholdersEquityPerShare TEXT,
                        interestDebtPerShare TEXT,
                        marketCap TEXT,
                        enterpriseValue TEXT,
                        peRatio TEXT,
                        priceToSalesRatio TEXT,
                        pocfratio TEXT,
                        pfcfRatio TEXT,
                        pbRatio TEXT,
                        ptbRatio TEXT,
                        evToSales TEXT,
                        enterpriseValueOverEBITDA TEXT,
                        evToOperatingCashFlow TEXT,
                        evToFreeCashFlow TEXT,
                        earningsYield TEXT,
                        freeCashFlowYield TEXT,
                        debtToEquity TEXT,
                        debtToAssets TEXT,
                        netDebtToEBITDA TEXT,
                        currentRatio TEXT,
                        interestCoverage TEXT,
                        incomeQuality TEXT,
                        dividendYield TEXT,
                        payoutRatio TEXT,
                        salesGeneralAndAdministrativeToRevenue TEXT,
                        researchAndDdevelopementToRevenue TEXT,
                        intangiblesToTotalAssets TEXT,
                        capexToOperatingCashFlow TEXT,
                        capexToRevenue TEXT,
                        capexToDepreciation TEXT,
                        stockBasedCompensationToRevenue TEXT,
                        grahamNumber TEXT,
                        roic TEXT,
                        returnOnTangibleAssets TEXT,
                        grahamNetNet TEXT,
                        workingCapital TEXT,
                        tangibleAssetValue TEXT,
                        netCurrentAssetValue TEXT,
                        investedCapital TEXT,
                        averageReceivables TEXT,
                        averagePayables TEXT,
                        averageInventory TEXT,
                        daysSalesOutstanding TEXT,
                        daysPayablesOutstanding TEXT,
                        daysOfInventoryOnHand TEXT,
                        receivablesTurnover TEXT,
                        payablesTurnover TEXT,
                        inventoryTurnover TEXT,
                        roe TEXT,
                        capexPerShare TEXT
                    )
                """)
                copy_sql = """
                    COPY temp_equity_key_metrics FROM STDIN
                    WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                with open(csv_path, 'r', encoding='utf-8') as f:
                    cur.copy_expert(copy_sql, f)

                # Insert with proper type casting for all numeric fields
                cur.execute("""
                    INSERT INTO equity_key_metrics (
                        symbol, date, calendaryear, period, revenuepershare, netincomepershare,
                        operatingcashflowpershare, freecashflowpershare, cashpershare, bookvaluepershare,
                        tangiblebookvaluepershare, shareholdersequitypershare, interestdebtpershare,
                        marketcap, enterprisevalue, peratio, pricetosalesratio, pocfratio, pfcfratio,
                        pbratio, ptbratio, evtosales, enterprisevalueoverebitda, evtooperatingcashflow,
                        evtofreecashflow, earningsyield, freecashflowyield, debttoequity, debttoassets,
                        netdebttoebitda, currentratio, interestcoverage, incomequality, dividendyield,
                        payoutratio, salesgeneralandadministrativetorevenue, researchanddevelopmenttorevenue,
                        intangiblestototalassets, capextooperatingcashflow, capextorevenue, capextodepreciation,
                        stockbasedcompensationtorevenue, grahamnumber, roic, returnontangibleassets,
                        grahamnetnet, workingcapital, tangibleassetvalue, netcurrentassetvalue,
                        investedcapital, averagereceivables, averagepayables, averageinventory,
                        dayssalesoutstanding, dayspayablesoutstanding, daysofinventoryonhand,
                        receivablesturnover, payablesturnover, inventoryturnover, roe, capexpershare
                    )
                    SELECT DISTINCT ON (symbol, date)
                        CASE WHEN symbol = '' OR symbol IS NULL THEN NULL ELSE symbol END,
                        CASE WHEN date = '' OR date IS NULL THEN NULL ELSE date::date END,
                        CASE WHEN calendarYear = '' OR calendarYear IS NULL THEN NULL ELSE calendarYear::smallint END,
                        CASE WHEN period = '' OR period IS NULL THEN NULL ELSE period END,
                        CASE WHEN revenuePerShare = '' OR revenuePerShare IS NULL THEN NULL ELSE revenuePerShare::double precision END,
                        CASE WHEN netIncomePerShare = '' OR netIncomePerShare IS NULL THEN NULL ELSE netIncomePerShare::double precision END,
                        CASE WHEN operatingCashFlowPerShare = '' OR operatingCashFlowPerShare IS NULL THEN NULL ELSE operatingCashFlowPerShare::double precision END,
                        CASE WHEN freeCashFlowPerShare = '' OR freeCashFlowPerShare IS NULL THEN NULL ELSE freeCashFlowPerShare::double precision END,
                        CASE WHEN cashPerShare = '' OR cashPerShare IS NULL THEN NULL ELSE cashPerShare::double precision END,
                        CASE WHEN bookValuePerShare = '' OR bookValuePerShare IS NULL THEN NULL ELSE bookValuePerShare::double precision END,
                        CASE WHEN tangibleBookValuePerShare = '' OR tangibleBookValuePerShare IS NULL THEN NULL ELSE tangibleBookValuePerShare::double precision END,
                        CASE WHEN shareholdersEquityPerShare = '' OR shareholdersEquityPerShare IS NULL THEN NULL ELSE shareholdersEquityPerShare::double precision END,
                        CASE WHEN interestDebtPerShare = '' OR interestDebtPerShare IS NULL THEN NULL ELSE interestDebtPerShare::double precision END,
                        CASE WHEN marketCap = '' OR marketCap IS NULL THEN NULL ELSE marketCap::numeric END,
                        CASE WHEN enterpriseValue = '' OR enterpriseValue IS NULL THEN NULL ELSE enterpriseValue::numeric END,
                        CASE WHEN peRatio = '' OR peRatio IS NULL THEN NULL ELSE peRatio::double precision END,
                        CASE WHEN priceToSalesRatio = '' OR priceToSalesRatio IS NULL THEN NULL ELSE priceToSalesRatio::double precision END,
                        CASE WHEN pocfratio = '' OR pocfratio IS NULL THEN NULL ELSE pocfratio::double precision END,
                        CASE WHEN pfcfRatio = '' OR pfcfRatio IS NULL THEN NULL ELSE pfcfRatio::double precision END,
                        CASE WHEN pbRatio = '' OR pbRatio IS NULL THEN NULL ELSE pbRatio::double precision END,
                        CASE WHEN ptbRatio = '' OR ptbRatio IS NULL THEN NULL ELSE ptbRatio::double precision END,
                        CASE WHEN evToSales = '' OR evToSales IS NULL THEN NULL ELSE evToSales::double precision END,
                        CASE WHEN enterpriseValueOverEBITDA = '' OR enterpriseValueOverEBITDA IS NULL THEN NULL ELSE enterpriseValueOverEBITDA::double precision END,
                        CASE WHEN evToOperatingCashFlow = '' OR evToOperatingCashFlow IS NULL THEN NULL ELSE evToOperatingCashFlow::double precision END,
                        CASE WHEN evToFreeCashFlow = '' OR evToFreeCashFlow IS NULL THEN NULL ELSE evToFreeCashFlow::double precision END,
                        CASE WHEN earningsYield = '' OR earningsYield IS NULL THEN NULL ELSE earningsYield::double precision END,
                        CASE WHEN freeCashFlowYield = '' OR freeCashFlowYield IS NULL THEN NULL ELSE freeCashFlowYield::double precision END,
                        CASE WHEN debtToEquity = '' OR debtToEquity IS NULL THEN NULL ELSE debtToEquity::double precision END,
                        CASE WHEN debtToAssets = '' OR debtToAssets IS NULL THEN NULL ELSE debtToAssets::double precision END,
                        CASE WHEN netDebtToEBITDA = '' OR netDebtToEBITDA IS NULL THEN NULL ELSE netDebtToEBITDA::double precision END,
                        CASE WHEN currentRatio = '' OR currentRatio IS NULL THEN NULL ELSE currentRatio::double precision END,
                        CASE WHEN interestCoverage = '' OR interestCoverage IS NULL THEN NULL ELSE interestCoverage::double precision END,
                        CASE WHEN incomeQuality = '' OR incomeQuality IS NULL THEN NULL ELSE incomeQuality::double precision END,
                        CASE WHEN dividendYield = '' OR dividendYield IS NULL THEN NULL ELSE dividendYield::double precision END,
                        CASE WHEN payoutRatio = '' OR payoutRatio IS NULL THEN NULL ELSE payoutRatio::double precision END,
                        CASE WHEN salesGeneralAndAdministrativeToRevenue = '' OR salesGeneralAndAdministrativeToRevenue IS NULL THEN NULL ELSE salesGeneralAndAdministrativeToRevenue::double precision END,
                        CASE WHEN researchAndDdevelopementToRevenue = '' OR researchAndDdevelopementToRevenue IS NULL THEN NULL ELSE researchAndDdevelopementToRevenue::double precision END,
                        CASE WHEN intangiblesToTotalAssets = '' OR intangiblesToTotalAssets IS NULL THEN NULL ELSE intangiblesToTotalAssets::double precision END,
                        CASE WHEN capexToOperatingCashFlow = '' OR capexToOperatingCashFlow IS NULL THEN NULL ELSE capexToOperatingCashFlow::double precision END,
                        CASE WHEN capexToRevenue = '' OR capexToRevenue IS NULL THEN NULL ELSE capexToRevenue::double precision END,
                        CASE WHEN capexToDepreciation = '' OR capexToDepreciation IS NULL THEN NULL ELSE capexToDepreciation::double precision END,
                        CASE WHEN stockBasedCompensationToRevenue = '' OR stockBasedCompensationToRevenue IS NULL THEN NULL ELSE stockBasedCompensationToRevenue::double precision END,
                        CASE WHEN grahamNumber = '' OR grahamNumber IS NULL THEN NULL ELSE grahamNumber::double precision END,
                        CASE WHEN roic = '' OR roic IS NULL THEN NULL ELSE roic::double precision END,
                        CASE WHEN returnOnTangibleAssets = '' OR returnOnTangibleAssets IS NULL THEN NULL ELSE returnOnTangibleAssets::double precision END,
                        CASE WHEN grahamNetNet = '' OR grahamNetNet IS NULL THEN NULL ELSE grahamNetNet::numeric END,
                        CASE WHEN workingCapital = '' OR workingCapital IS NULL THEN NULL ELSE workingCapital::numeric END,
                        CASE WHEN tangibleAssetValue = '' OR tangibleAssetValue IS NULL THEN NULL ELSE tangibleAssetValue::numeric END,
                        CASE WHEN netCurrentAssetValue = '' OR netCurrentAssetValue IS NULL THEN NULL ELSE netCurrentAssetValue::numeric END,
                        CASE WHEN investedCapital = '' OR investedCapital IS NULL THEN NULL ELSE investedCapital::numeric END,
                        CASE WHEN averageReceivables = '' OR averageReceivables IS NULL THEN NULL ELSE averageReceivables::numeric END,
                        CASE WHEN averagePayables = '' OR averagePayables IS NULL THEN NULL ELSE averagePayables::numeric END,
                        CASE WHEN averageInventory = '' OR averageInventory IS NULL THEN NULL ELSE averageInventory::numeric END,
                        CASE WHEN daysSalesOutstanding = '' OR daysSalesOutstanding IS NULL THEN NULL ELSE daysSalesOutstanding::double precision END,
                        CASE WHEN daysPayablesOutstanding = '' OR daysPayablesOutstanding IS NULL THEN NULL ELSE daysPayablesOutstanding::double precision END,
                        CASE WHEN daysOfInventoryOnHand = '' OR daysOfInventoryOnHand IS NULL THEN NULL ELSE daysOfInventoryOnHand::double precision END,
                        CASE WHEN receivablesTurnover = '' OR receivablesTurnover IS NULL THEN NULL ELSE receivablesTurnover::double precision END,
                        CASE WHEN payablesTurnover = '' OR payablesTurnover IS NULL THEN NULL ELSE payablesTurnover::double precision END,
                        CASE WHEN inventoryTurnover = '' OR inventoryTurnover IS NULL THEN NULL ELSE inventoryTurnover::double precision END,
                        CASE WHEN roe = '' OR roe IS NULL THEN NULL ELSE roe::double precision END,
                        CASE WHEN capexPerShare = '' OR capexPerShare IS NULL THEN NULL ELSE capexPerShare::double precision END
                    FROM temp_equity_key_metrics
                    WHERE symbol IS NOT NULL AND symbol != ''
                    ORDER BY symbol, date
                    ON CONFLICT (symbol, date) DO NOTHING
                """)
                cur.execute("DROP TABLE temp_equity_key_metrics")
            elif table_name in ['equity_financial_ratio', 'equity_earnings'] or table_name.endswith('_growth'):
                # These tables may have duplicate key issues, use temp table approach
                cur.execute(f"CREATE TEMP TABLE temp_{table_name} (LIKE {table_name})")
                copy_sql = f"""
                    COPY temp_{table_name} FROM STDIN
                    WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                """
                with open(csv_path, 'r', encoding='utf-8') as f:
                    cur.copy_expert(copy_sql, f)

                # Insert handling duplicates - keep distinct records for all tables with (symbol, date) primary key
                cur.execute(f"""
                    INSERT INTO {table_name}
                    SELECT DISTINCT ON (symbol, date) *
                    FROM temp_{table_name}
                    ORDER BY symbol, date
                    ON CONFLICT (symbol, date) DO NOTHING
                """)
                cur.execute(f"DROP TABLE temp_{table_name}")
            else:
                # Preprocess CSV for tables with data type issues
                needs_preprocessing = table_name in [
                    'equity_profile', 'etfs_profile', 'equity_income', 'equity_key_metrics', 'etfs_data'
                ]

                if needs_preprocessing:
                    # Use preprocessing to clean data
                    temp_csv_path = preprocess_csv_data(csv_path)
                    try:
                        copy_sql = f"""
                            COPY {table_name} FROM STDIN
                            WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                        """
                        with open(temp_csv_path, 'r', encoding='utf-8') as f:
                            cur.copy_expert(copy_sql, f)
                    finally:
                        # Clean up temp file
                        try:
                            os.unlink(temp_csv_path)
                        except:
                            pass
                else:
                    # Use COPY FROM with CSV format to properly handle quoted fields
                    copy_sql = f"""
                        COPY {table_name} FROM STDIN
                        WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                    """
                    with open(csv_path, 'r', encoding='utf-8') as f:
                        cur.copy_expert(copy_sql, f)

            conn.commit()

            # Get row count
            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cur.fetchone()[0]
            logger.info(f"Successfully loaded {row_count} rows into {table_name}")

        return True

    except Exception as e:
        logger.error(f"Error loading {csv_file} to {table_name}: {e}")
        conn.rollback()
        return False

def clear_existing_data(conn):
    """Clear existing FMP data before loading."""
    logger.info("Clearing existing FMP data...")

    # Include all FMP tables plus the additional ones
    tables = list(FMP_CSV_TABLES.values()) + [
        'etfs_quotes', 'equity_quotes', 'equity_balance_growth', 'equity_cashflow_growth',
        'equity_financial_growth', 'equity_income_growth'
    ]

    with conn.cursor() as cur:
        for table in tables:
            try:
                cur.execute(f"TRUNCATE TABLE {table} CASCADE")
                logger.info(f"Cleared table: {table}")
            except Exception as e:
                logger.warning(f"Could not clear table {table}: {e}")

        conn.commit()

def load_etf_quotes_directory(conn) -> bool:
    """Load all ETF quote files from etfs_quotes directory."""
    logger.info("Loading ETF quotes from directory...")

    etf_quotes_dir = os.path.join(get_fmp_csv_directory(), 'etfs_quotes')

    if not os.path.exists(etf_quotes_dir):
        logger.warning("ETF quotes directory not found")
        return True  # Not an error if directory doesn't exist

    success_count = 0
    total_files = 0

    for filename in os.listdir(etf_quotes_dir):
        if filename.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(etf_quotes_dir, filename)

            logger.info(f"Loading ETF quotes from {filename}...")

            try:
                with conn.cursor() as cur:
                    # Use temp table approach to handle potential duplicates
                    # Create explicit temp table schema to ensure correct data types
                    cur.execute("""
                        CREATE TEMP TABLE temp_etfs_quotes (
                            date TEXT,
                            open TEXT,
                            high TEXT,
                            low TEXT,
                            close TEXT,
                            adjclose TEXT,
                            volume TEXT,
                            unadjustedvolume TEXT,
                            change TEXT,
                            changepercent TEXT,
                            vwap TEXT,
                            label TEXT,
                            changeovertime TEXT,
                            symbol TEXT
                        )
                    """)

                    # Clean the CSV data before loading
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                        with open(file_path, 'r', encoding='utf-8') as original_file:
                            header = original_file.readline().strip()
                            # Normalize header column names to lowercase for PostgreSQL compatibility
                            header_cols = header.split(',')
                            header_cols = [col.lower() for col in header_cols]
                            normalized_header = ','.join(header_cols)
                            temp_file.write(normalized_header + '\n')

                            line_count = 0
                            for line in original_file:
                                line_count += 1
                                try:
                                    cleaned_line = clean_csv_line(line, 14)
                                    temp_file.write(cleaned_line + '\n')
                                except Exception as e:
                                    logger.warning(f"Skipping malformed line {line_count} in {filename}: {str(e)}")
                                    continue

                        temp_file.flush()

                    copy_sql = """
                        COPY temp_etfs_quotes (date, open, high, low, close, adjclose, volume, unadjustedvolume, change, changepercent, vwap, label, changeovertime, symbol) FROM STDIN
                        WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                    """
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        cur.copy_expert(copy_sql, f)

                    # Clean up temp file
                    os.unlink(temp_file.name)

                    # Insert with duplicate handling and explicit casting
                    cur.execute("""
                        INSERT INTO etfs_quotes (date, open, high, low, close, adjclose, volume, unadjustedvolume, change, changepercent, vwap, label, changeovertime, symbol)
                        SELECT DISTINCT ON (symbol, date)
                            CASE WHEN date = '' OR date IS NULL THEN NULL ELSE date::date END,
                            CASE WHEN open = '' OR open IS NULL THEN NULL ELSE open::double precision END,
                            CASE WHEN high = '' OR high IS NULL THEN NULL ELSE high::double precision END,
                            CASE WHEN low = '' OR low IS NULL THEN NULL ELSE low::double precision END,
                            CASE WHEN close = '' OR close IS NULL THEN NULL ELSE close::double precision END,
                            CASE WHEN adjclose = '' OR adjclose IS NULL THEN NULL ELSE adjclose::double precision END,
                            CASE WHEN volume = '' OR volume IS NULL THEN NULL ELSE volume::double precision::bigint END,
                            CASE WHEN unadjustedvolume = '' OR unadjustedvolume IS NULL THEN NULL ELSE unadjustedvolume::double precision::bigint END,
                            CASE WHEN change = '' OR change IS NULL THEN NULL ELSE change::double precision END,
                            CASE WHEN changepercent = '' OR changepercent IS NULL THEN NULL ELSE changepercent::double precision END,
                            CASE WHEN vwap = '' OR vwap IS NULL THEN NULL ELSE vwap::double precision END,
                            label,
                            CASE WHEN changeovertime = '' OR changeovertime IS NULL THEN NULL ELSE changeovertime::double precision END,
                            symbol
                        FROM temp_etfs_quotes
                        ORDER BY symbol, date
                        ON CONFLICT (symbol, date) DO NOTHING
                    """)
                    cur.execute("DROP TABLE temp_etfs_quotes")

                conn.commit()
                success_count += 1
                logger.info(f"Successfully loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")

    if total_files > 0:
        logger.info(f"ETF quotes loading: {success_count}/{total_files} files loaded")

    return success_count == total_files

def preprocess_csv_data(csv_path: str) -> str:
    """Preprocess CSV to handle empty values and invalid data types."""
    import tempfile
    import csv

    temp_fd, temp_path = tempfile.mkstemp(suffix='.csv')

    try:
        with open(csv_path, 'r', encoding='utf-8') as infile, os.fdopen(temp_fd, 'w', encoding='utf-8', newline='') as outfile:
            reader = csv.reader(infile)
            writer = csv.writer(outfile)

            # Copy header and normalize to lowercase for PostgreSQL compatibility
            header = next(reader)
            header = [col.lower() for col in header]
            writer.writerow(header)

            # Process each row
            for row_num, row in enumerate(reader, start=2):
                cleaned_row = []
                for cell in row:
                    # Clean empty cells and invalid values
                    cell = cell.strip()
                    if cell in ['', ' ', 'null', 'NULL', 'None']:
                        cell = ''
                    # Handle common problematic patterns
                    elif cell == '""':
                        cell = ''
                    cleaned_row.append(cell)

                # Ensure correct number of columns
                while len(cleaned_row) < len(header):
                    cleaned_row.append('')
                if len(cleaned_row) > len(header):
                    cleaned_row = cleaned_row[:len(header)]

                writer.writerow(cleaned_row)

        return temp_path

    except Exception as e:
        try:
            os.unlink(temp_path)
        except:
            pass
        raise e

def clean_csv_line(line: str, expected_columns: int = None) -> str:
    """Clean a CSV line to fix common issues like unterminated quotes and missing columns."""
    line = line.strip()
    if not line:
        return line

    # Special handling for lines with "date, year" pattern that break CSV parsing
    # Example: "October 29, 12" should be "October 29 12" or properly escaped
    if '"' in line and ',' in line:
        # Find patterns like "Month DD, YY" and fix them
        import re
        pattern = r'"([A-Za-z]+ \d+), (\d+)"'
        match = re.search(pattern, line)
        if match:
            # Replace comma with space within quotes
            old_text = match.group(0)
            new_text = f'"{match.group(1)} {match.group(2)}"'
            line = line.replace(old_text, new_text)
            logger.warning(f"Fixed comma within quotes: {old_text} -> {new_text}")

    # Count quotes in the line
    quote_count = line.count('"')

    # If odd number of quotes, we likely have an unterminated quote
    if quote_count % 2 == 1:
        # Find the last quote and check if it needs closing
        last_quote_pos = line.rfind('"')
        if last_quote_pos == len(line) - 1:
            # Line ends with a quote, likely fine
            return line
        else:
            # Add closing quote at the end
            line = line + '"'
            logger.warning(f"Fixed unterminated quote in line: {line[:50]}...")

    # If expected_columns is provided, ensure the line has enough columns
    if expected_columns:
        # Count actual columns (handle quoted commas)
        try:
            reader = csv.reader(io.StringIO(line))
            row = next(reader)
            actual_columns = len(row)

            if actual_columns < expected_columns:
                # Pad with empty values
                missing_count = expected_columns - actual_columns
                line = line + ',' * missing_count
                logger.warning(f"Padded line with {missing_count} missing columns: {line[:50]}...")

        except Exception as e:
            # If CSV parsing fails, try to fix common issues and retry
            logger.warning(f"CSV parsing failed, attempting to fix: {str(e)}")

            # Last resort: if there are still issues, replace problematic quotes
            if 'unterminated quoted field' in str(e).lower():
                # Try to fix by escaping internal quotes
                line = line.replace('""', '"')  # Remove double quotes
                # If still odd number of quotes, add one at the end
                if line.count('"') % 2 == 1:
                    line = line + '"'
                    logger.warning(f"Applied last resort quote fix: {line[:50]}...")

    return line

def load_equity_quotes_directory(conn) -> bool:
    """Load all equity quote files from equity_quotes directory."""
    logger.info("Loading equity quotes from directory...")

    equity_quotes_dir = os.path.join(get_fmp_csv_directory(), 'equity_quotes')

    if not os.path.exists(equity_quotes_dir):
        logger.warning("Equity quotes directory not found")
        return True  # Not an error if directory doesn't exist

    success_count = 0
    total_files = 0

    for filename in os.listdir(equity_quotes_dir):
        if filename.endswith('.csv'):
            total_files += 1
            file_path = os.path.join(equity_quotes_dir, filename)

            logger.info(f"Loading equity quotes from {filename}...")

            try:
                with conn.cursor() as cur:
                    # Use temp table approach to handle potential duplicates
                    # Explicitly define schema to ensure correct data types (especially TEXT for volume)
                    cur.execute("""
                        CREATE TEMP TABLE temp_equity_quotes (
                            date TEXT,
                            open TEXT,
                            high TEXT,
                            low TEXT,
                            close TEXT,
                            adjclose TEXT,
                            volume TEXT,
                            unadjustedvolume TEXT,
                            change TEXT,
                            changepercent TEXT,
                            vwap TEXT,
                            label TEXT,
                            changeovertime TEXT,
                            symbol TEXT
                        )
                    """)

                    # Clean the CSV data before loading
                    with tempfile.NamedTemporaryFile(mode='w+', suffix='.csv', delete=False) as temp_file:
                        with open(file_path, 'r', encoding='utf-8') as original_file:
                            header = original_file.readline().strip()
                            # Normalize header column names to lowercase for PostgreSQL compatibility
                            header_cols = header.split(',')
                            header_cols = [col.lower() for col in header_cols]
                            normalized_header = ','.join(header_cols)
                            temp_file.write(normalized_header + '\n')

                            line_count = 0
                            for line in original_file:
                                line_count += 1
                                try:
                                    cleaned_line = clean_csv_line(line, 14)
                                    temp_file.write(cleaned_line + '\n')
                                except Exception as e:
                                    logger.warning(f"Skipping malformed line {line_count} in {filename}: {str(e)}")
                                    continue

                        temp_file.flush()

                    copy_sql = """
                        COPY temp_equity_quotes (date, open, high, low, close, adjclose, volume, unadjustedvolume, change, changepercent, vwap, label, changeovertime, symbol) FROM STDIN
                        WITH (FORMAT CSV, HEADER true, DELIMITER ',', QUOTE '"', ESCAPE '"')
                    """
                    with open(temp_file.name, 'r', encoding='utf-8') as f:
                        cur.copy_expert(copy_sql, f)

                    # Clean up temp file
                    os.unlink(temp_file.name)

                    # Insert with duplicate handling and explicit casting
                    cur.execute("""
                        INSERT INTO equity_quotes (date, symbol, open, high, low, close, adjclose, volume, unadjustedvolume, change, changepercent, vwap, label, changeovertime)
                        SELECT DISTINCT ON (symbol, date)
                            CASE WHEN date = '' OR date IS NULL THEN NULL ELSE date::date END,
                            symbol,
                            CASE WHEN open = '' OR open IS NULL THEN NULL ELSE open::double precision END,
                            CASE WHEN high = '' OR high IS NULL THEN NULL ELSE high::double precision END,
                            CASE WHEN low = '' OR low IS NULL THEN NULL ELSE low::double precision END,
                            CASE WHEN close = '' OR close IS NULL THEN NULL ELSE close::double precision END,
                            CASE WHEN adjclose = '' OR adjclose IS NULL THEN NULL ELSE adjclose::double precision END,
                            CASE WHEN volume = '' OR volume IS NULL THEN NULL ELSE volume::double precision::bigint END,
                            CASE WHEN unadjustedvolume = '' OR unadjustedvolume IS NULL THEN NULL ELSE unadjustedvolume::double precision::bigint END,
                            CASE WHEN change = '' OR change IS NULL THEN NULL ELSE change::double precision END,
                            CASE WHEN changepercent = '' OR changepercent IS NULL THEN NULL ELSE changepercent::double precision END,
                            CASE WHEN vwap = '' OR vwap IS NULL THEN NULL ELSE vwap::double precision END,
                            label,
                            CASE WHEN changeovertime = '' OR changeovertime IS NULL THEN NULL ELSE changeovertime::double precision END
                        FROM temp_equity_quotes
                        ORDER BY symbol, date
                        ON CONFLICT (symbol, date) DO NOTHING
                    """)
                    cur.execute("DROP TABLE temp_equity_quotes")

                conn.commit()
                success_count += 1
                logger.info(f"Successfully loaded {filename}")

            except Exception as e:
                logger.error(f"Error loading {filename}: {e}")
                conn.rollback()  # Rollback this file's transaction to prevent cascade failures
                continue  # Continue to next file

    if total_files > 0:
        logger.info(f"Equity quotes loading: {success_count}/{total_files} files loaded")

    return success_count == total_files

def load_all_fmp_csvs() -> bool:
    """Load all FMP CSV files to PostgreSQL."""
    logger.info("Starting FMP CSV loading process...")

    try:
        # Connect to database
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        # Create tables
        create_tables(conn)

        # Clear existing data
        clear_existing_data(conn)

        # Load each CSV file
        success_count = 0
        for csv_file, table_name in FMP_CSV_TABLES.items():
            if load_csv_to_table(conn, csv_file, table_name):
                success_count += 1

        # Load ETF quotes directory
        etf_quotes_success = load_etf_quotes_directory(conn)

        # Load equity quotes directory
        equity_quotes_success = load_equity_quotes_directory(conn)

        total_expected = len(FMP_CSV_TABLES)
        logger.info(f"FMP CSV loading completed: {success_count}/{total_expected} files loaded successfully")

        if etf_quotes_success:
            logger.info("ETF quotes loading completed successfully")

        if equity_quotes_success:
            logger.info("Equity quotes loading completed successfully")

        # Define core tables that are essential for basic functionality (growth tables are optional)
        core_csv_files = {
            'equity_profile.csv', 'equity_income.csv', 'equity_balance.csv',
            'equity_cash_flow.csv', 'equity_earnings.csv', 'equity_peers.csv',
            'equity_ratios.csv', 'equity_key_metrics.csv', 'equity_financial_scores.csv',
            'etfs_profile.csv', 'etfs_peers.csv', 'etfs_data.csv'
        }

        # Count how many core files were successfully loaded
        core_loaded = 0
        for csv_file, table_name in FMP_CSV_TABLES.items():
            if csv_file in core_csv_files:
                # Check if this table was loaded (check if it has data)
                try:
                    cursor = conn.cursor()
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cursor.fetchone()[0]
                    if count > 0:
                        core_loaded += 1
                    cursor.close()
                except:
                    pass

        conn.close()

        # Success criteria:
        # 1. All core files loaded (12/12) OR
        # 2. At least 75% of all files loaded (12/16 = 75%) AND quotes successful
        core_success = core_loaded >= len(core_csv_files)
        percentage_success = (success_count / total_expected) >= 0.75
        quotes_success = etf_quotes_success and equity_quotes_success

        overall_success = core_success or (percentage_success and quotes_success)

        if overall_success:
            logger.info(f"FMP loading successful: {core_loaded}/{len(core_csv_files)} core files, {success_count}/{total_expected} total files")
        else:
            logger.warning(f"FMP loading below threshold: {core_loaded}/{len(core_csv_files)} core files, {success_count}/{total_expected} total files")

        return overall_success

    except Exception as e:
        logger.error(f"Error in FMP CSV loading process: {e}")
        return False

def get_loading_status() -> Dict[str, int]:
    """Get row counts for all FMP tables."""
    try:
        conn = psycopg2.connect(
            host=DB_HOST,
            port=DB_PORT,
            dbname=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD
        )

        status = {}
        with conn.cursor() as cur:
            # Check main FMP tables
            for table_name in FMP_CSV_TABLES.values():
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    status[table_name] = count
                except Exception as e:
                    status[table_name] = f"Error: {e}"

            # Check directory-based tables
            for table_name in ['etfs_quotes', 'equity_quotes']:
                try:
                    cur.execute(f"SELECT COUNT(*) FROM {table_name}")
                    count = cur.fetchone()[0]
                    status[table_name] = count
                except Exception as e:
                    status[table_name] = f"Error: {e}"

        conn.close()
        return status

    except Exception as e:
        logger.error(f"Error getting loading status: {e}")
        return {}

if __name__ == "__main__":
    # Check for missing CSV files and download if needed (for Render deployment)
    missing_files = check_files_exist()
    if missing_files:
        logger.info(f"Found {len(missing_files)} missing CSV files. Attempting to download...")
        if not download_csv_files():
            logger.error("Failed to download required CSV files")
            logger.info("If running locally, ensure CSV files are present or set CSV_BASE_URL environment variable")
            # Continue anyway - might be local development

    success = load_all_fmp_csvs()

    if success:
        print("\n=== FMP CSV Loading Complete ===")
        status = get_loading_status()
        for table, count in status.items():
            print(f"{table}: {count} rows")
    else:
        print("FMP CSV loading failed - check logs for details")
        exit(1)