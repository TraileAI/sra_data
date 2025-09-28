#!/bin/bash

# Core FMP files
echo "Downloading core FMP files..."
b2 file download b2://sra-data-csv/equity_income.csv fmp_data/equity_income.csv &
b2 file download b2://sra-data-csv/equity_balance.csv fmp_data/equity_balance.csv &
b2 file download b2://sra-data-csv/equity_cash_flow.csv fmp_data/equity_cash_flow.csv &
b2 file download b2://sra-data-csv/equity_peers.csv fmp_data/equity_peers.csv &
b2 file download b2://sra-data-csv/equity_ratios.csv fmp_data/equity_ratios.csv &
b2 file download b2://sra-data-csv/equity_key_metrics.csv fmp_data/equity_key_metrics.csv &

# Growth files
echo "Downloading growth files..."
b2 file download b2://sra-data-csv/equity_balance_growth.csv fmp_data/equity_balance_growth.csv &
b2 file download b2://sra-data-csv/equity_cashflow_growth.csv fmp_data/equity_cashflow_growth.csv &
b2 file download b2://sra-data-csv/equity_financial_growth.csv fmp_data/equity_financial_growth.csv &
b2 file download b2://sra-data-csv/equity_income_growth.csv fmp_data/equity_income_growth.csv &

# ETF files
echo "Downloading ETF files..."
b2 file download b2://sra-data-csv/etfs_profile.csv fmp_data/etfs_profile.csv &
b2 file download b2://sra-data-csv/etfs_peers.csv fmp_data/etfs_peers.csv &
b2 file download b2://sra-data-csv/etfs_data.csv fmp_data/etfs_data.csv &

# Wait for all downloads to complete
wait

echo "All FMP files downloaded!"