# =============================================================================
# Generated ETL Pipeline
# Target table : fraud_transactions
# Source file  : data/raw/sample_transactions.csv
# Framework    : pandas
# Generated at : 2026-04-29 19:42:38 UTC
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
    df_input = load_csv(INPUT_PATH)
    rejected_counts = {}

    rename_map = {
        "Time": "transaction_timestamp",
        "Amount": "transaction_amount",
        "Class": "is_fraud"
    }
    df = rename_columns(df_input, rename_map)

    df, rejected_df = filter_non_negative(df, "transaction_amount")
    rejected_counts["transaction_amount"] = len(rejected_df)

    df, rejected_df = filter_positive(df, "transaction_amount")
    rejected_counts["transaction_amount_positive"] = len(rejected_df)

    df, rejected_df = filter_not_null(df, ["account_balance"])
    rejected_counts["account_balance"] = len(rejected_df)

    df = impute_nulls(df, "account_balance", strategy="mean")

    df = convert_seconds_to_timestamp(df, "transaction_timestamp", "transaction_timestamp")
    df = add_ingestion_timestamp(df)

    df = add_transaction_date(df, "transaction_timestamp")
    df = add_transaction_hour(df, "transaction_timestamp")
    df = add_is_weekend(df, "transaction_timestamp")
    df = add_amount_category(df, "transaction_amount")
    df = add_utilization_rate(df, "account_balance", "credit_limit")
    df = add_is_high_risk_context(df, "is_foreign_transaction", "num_declined_7d", "distance_from_home_km")

    df["cardholder_state"] = df["cardholder_state"].str.upper()
    df["merchant_category"] = df["merchant_category"].str.lower()
    df["channel"] = df["channel"].str.lower()
    df["pos_entry_mode"] = df["pos_entry_mode"].str.lower()
    df["merchant_country"] = df["merchant_country"].str.upper()

    save_csv(df, OUTPUT_PATH)
    generate_validation_report(df_input, df, rejected_counts, REPORT_PATH)
    print("ETL pipeline completed successfully.")

if __name__ == "__main__":
    run()
