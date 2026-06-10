# =============================================================================
# Generated ETL Pipeline
# Target table : fraud_transactions
# Source file  : data/raw/sample_transactions.csv
# Framework    : pandas
# Generated at : 2026-06-10 15:01:38 UTC
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
    generate_validation_report,
)

INPUT_PATH  = "data/raw/sample_transactions.csv"
OUTPUT_PATH = "data/processed/fraud_transactions.csv"
REPORT_PATH = "data/processed/validation_report.txt"


def run():
    df = load_csv(INPUT_PATH)
    df_input = df.copy()
    rejected_counts = {}

    rename_map = {
        "Time": "transaction_timestamp",
        "Amount": "transaction_amount",
        "Class": "is_fraud",
        "merchant_name": "merchant_name"
    }
    df = rename_columns(df, rename_map)

    df, rejected_df = filter_non_negative(df, "transaction_amount")
    rejected_counts["non_negative_transaction_amount"] = len(rejected_df)

    df, rejected_df = filter_not_null(df, ["account_balance"])
    rejected_counts["not_null_account_balance"] = len(rejected_df)

    df = convert_seconds_to_timestamp(df, "transaction_timestamp", "transaction_timestamp", "2020-01-01 00:00:00")

    df["merchant_name"] = replace_blank_with(df, "merchant_name", "UNKNOWN")
    df["merchant_country"] = standardize_country_codes(df, "merchant_country", {
        "USA": "US", "us": "US", "U.S.": "US", "United States": "US"
    })

    df = add_transaction_date(df, "transaction_timestamp")
    df = add_transaction_hour(df, "transaction_timestamp")
    df = add_is_weekend(df, "transaction_timestamp")
    df = add_amount_category(df, "transaction_amount")
    df = add_utilization_rate(df, "account_balance", "credit_limit")
    df = add_is_high_risk_context(df, "is_foreign_transaction", "num_declined_7d", "distance_from_home_km")

    df = cast_column(df, "transaction_amount", "decimal")
    df = cast_column(df, "credit_limit", "decimal")
    df = cast_column(df, "account_balance", "decimal")
    df = cast_column(df, "avg_amount_30d", "decimal")
    df = cast_column(df, "distance_from_home_km", "decimal")

    df = cast_column(df, "is_fraud", "int32")
    df = cast_column(df, "is_foreign_transaction", "int32")
    df = cast_column(df, "is_weekend", "int32")
    df = cast_column(df, "is_high_risk_context", "int32")
    df = cast_column(df, "cardholder_age", "int32")
    df = cast_column(df, "account_age_days", "int32")
    df = cast_column(df, "num_transactions_24h", "int32")
    df = cast_column(df, "num_declined_7d", "int32")

    df = add_ingestion_timestamp(df)

    save_csv(df, OUTPUT_PATH)
    generate_validation_report(df_input, df, rejected_counts, REPORT_PATH)
    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    run()
