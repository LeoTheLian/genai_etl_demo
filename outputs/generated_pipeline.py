# =============================================================================
# Generated ETL Pipeline
# Target table : fraud_transactions
# Source file  : data/raw/sample_transactions.csv
# Framework    : pandas
# Generated at : 2026-06-16 17:09:40 UTC
# Mode         : llm_only
# =============================================================================

import sys
from pathlib import Path

# Make the project root importable when this file is run from any cwd
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.etl_utilities import (
    load_csv,
    save_csv,
    rename_columns,
    filter_non_negative,
    filter_positive,
    filter_range,
    filter_valid_values,
    filter_not_null,
    cast_column,
    impute_nulls,
    replace_blank_with,
    convert_seconds_to_timestamp,
    add_ingestion_timestamp,
    add_transaction_date,
    add_transaction_hour,
    add_is_weekend,
    add_amount_category,
    add_utilization_rate,
    add_is_high_risk_context,
    standardize_country_codes,
)

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_PATH  = str(_PROJECT_ROOT / "data/raw/sample_transactions.csv")
OUTPUT_PATH = str(_PROJECT_ROOT / "data/processed/fraud_transactions.csv")


def run():
    # Load source data
    df = load_csv(INPUT_PATH)

    # Rename columns
    rename_map = {
        "Time": "transaction_timestamp",
        "Amount": "transaction_amount",
        "Class": "is_fraud"
    }
    df = rename_columns(df, rename_map)

    # Convert seconds to timestamp
    df = convert_seconds_to_timestamp(df, 'transaction_timestamp', 'transaction_timestamp')

    # Replace blank merchant names
    df['merchant_name'] = replace_blank_with(df, 'merchant_name', 'UNKNOWN')

    # Standardize merchant category to lowercase
    df['merchant_category'] = df['merchant_category'].str.lower()

    # Standardize channel to lowercase
    df['channel'] = df['channel'].str.lower()

    # Standardize pos_entry_mode to lowercase
    df['pos_entry_mode'] = df['pos_entry_mode'].str.lower()

    # Standardize merchant country codes
    df['merchant_country'] = standardize_country_codes(df, 'merchant_country')

    # Add derived columns
    df = add_ingestion_timestamp(df)
    df = add_transaction_date(df, 'transaction_timestamp')
    df = add_transaction_hour(df, 'transaction_timestamp')
    df = add_is_weekend(df, 'transaction_timestamp')
    df = add_amount_category(df, 'transaction_amount')
    df = add_utilization_rate(df, 'account_balance', 'credit_limit')
    df = add_is_high_risk_context(df, 'is_foreign_transaction', 'num_declined_7d', 'distance_from_home_km')

    # Type casting
    df = cast_column(df, 'transaction_amount', 'float64')
    df = cast_column(df, 'credit_limit', 'float64')
    df = cast_column(df, 'account_balance', 'float64')
    df = cast_column(df, 'avg_amount_30d', 'float64')
    df = cast_column(df, 'is_fraud', 'integer')
    df = cast_column(df, 'account_age_days', 'integer')
    df = cast_column(df, 'cardholder_age', 'integer')
    df = cast_column(df, 'is_foreign_transaction', 'integer')
    df = cast_column(df, 'num_transactions_24h', 'integer')
    df = cast_column(df, 'num_declined_7d', 'integer')
    df = cast_column(df, 'transaction_hour', 'integer')
    df = cast_column(df, 'is_weekend', 'integer')
    df = cast_column(df, 'is_high_risk_context', 'integer')

    # Save results
    save_csv(df, OUTPUT_PATH)

    # Print completion message
    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    run()
