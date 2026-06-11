# agents/etl_utilities.py
"""
Generic ETL utility functions.

All functions operate on pandas DataFrames and return DataFrames (or tuples).
Functions here are designed to be called directly from generated pipeline code.

PySpark note:
  Each function is documented with the equivalent PySpark pattern so the
  developer_agent's LLM prompt can guide generation of PySpark pipelines.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd


REFERENCE_TIMESTAMP = "2020-01-01 00:00:00"

COUNTRY_STANDARD_MAP = {
    "usa": "US",
    "us": "US",
    "u.s.": "US",
    "u.s.a.": "US",
    "united states": "US",
    "united states of america": "US",
    "ca": "CA",
    "mx": "MX",
    "gb": "GB",
    "de": "DE",
    "fr": "FR",
}


def load_csv(path: str) -> pd.DataFrame:
    """Load a CSV file into a DataFrame."""
    return pd.read_csv(path)


def save_csv(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame to a CSV file, creating parent directories as needed."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def rename_columns(df: pd.DataFrame, rename_map: dict) -> pd.DataFrame:
    """Rename columns using a source->target dict. Silently skips missing source columns.
    Returns DataFrame — assign back: df = rename_columns(df, {"OldName": "new_name"}).
    """
    existing = {k: v for k, v in rename_map.items() if k in df.columns}
    return df.rename(columns=existing)


def filter_non_negative(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows where column >= 0 (nulls are rejected).
    Returns (clean_df, rejected_df) — always unpack both.
    """
    mask = df[column].notna() & (df[column] >= 0)
    return df[mask].copy(), df[~mask].copy()


def filter_positive(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows where column > 0 (nulls and zeros are rejected).
    Returns (clean_df, rejected_df) — always unpack both.
    """
    mask = df[column].notna() & (df[column] > 0)
    return df[mask].copy(), df[~mask].copy()


def filter_range(df: pd.DataFrame, column: str, min_val, max_val) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows where min_val <= column <= max_val (nulls are rejected).
    Returns (clean_df, rejected_df) — always unpack both.
    """
    mask = df[column].notna() & (df[column] >= min_val) & (df[column] <= max_val)
    return df[mask].copy(), df[~mask].copy()


def filter_valid_values(df: pd.DataFrame, column: str, valid_values: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep rows where column value is in valid_values list.
    Returns (clean_df, rejected_df) — always unpack both.
    """
    mask = df[column].isin(valid_values)
    return df[mask].copy(), df[~mask].copy()


def filter_not_null(df: pd.DataFrame, columns: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows where ANY of the listed columns is null.
    Returns (clean_df, rejected_df) — always unpack both.
    """
    mask = df[columns].notna().all(axis=1)
    return df[mask].copy(), df[~mask].copy()


def cast_column(df: pd.DataFrame, column: str, target_type: str) -> pd.DataFrame:
    """Cast a column to a target type. Supported target_type values:
    float64 | decimal — coerces to float64 (use for all monetary/decimal columns).
    int32 | int8 | integer — coerces to Int64 (nullable integer, preserves NaN).
    string — strips whitespace and casts to pandas StringDtype.
    timestamp | datetime — parses to datetime64.
    date — parses to datetime then formats as YYYY-MM-DD string.
    Returns DataFrame — assign back: df = cast_column(df, col, type).
    """
    df = df.copy()
    t = target_type.lower()
    if t in ("float", "float64", "decimal", "decimal(10,2)"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("float64")
    elif t in ("int", "int32", "int8", "integer"):
        df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
    elif t in ("string", "str"):
        df[column] = df[column].astype("string").str.strip()
    elif t in ("timestamp", "datetime"):
        df[column] = pd.to_datetime(df[column], errors="coerce")
    elif t == "date":
        df[column] = pd.to_datetime(df[column], errors="coerce").dt.strftime("%Y-%m-%d")
    return df


def impute_nulls(df: pd.DataFrame, column: str, strategy: str = "median") -> pd.Series:
    """Fill null values in a column. strategy: 'median' | 'mean' | 'zero'.
    Returns Series — assign directly: df[col] = impute_nulls(df, col, 'median').
    """
    if strategy == "median":
        return df[column].fillna(df[column].median())
    elif strategy == "mean":
        return df[column].fillna(df[column].mean())
    elif strategy == "zero":
        return df[column].fillna(0)
    return df[column]


def replace_blank_with(df: pd.DataFrame, column: str, replacement: str = "UNKNOWN") -> pd.Series:
    """Replace empty-string or whitespace-only values with a replacement string.
    NaN/None values are also replaced. Valid non-empty strings are left unchanged.
    Returns Series — assign directly: df[col] = replace_blank_with(df, col, 'UNKNOWN').
    """
    normalized = df[column].astype("string").str.strip()
    return normalized.mask(normalized == "", replacement).fillna(replacement)


def standardize_country_codes(df: pd.DataFrame, column: str, country_map: dict = None) -> pd.Series:
    """Standardize country code variants in a column to ISO 2-letter codes.

    Keys in country_map are matched case-insensitively (normalized to lowercase
    internally at call time), so the caller does not need to worry about key case.
    Unrecognized values are preserved as-is (trimmed).
    Uses COUNTRY_STANDARD_MAP by default, which covers common US variants:
      {'usa', 'us', 'u.s.', 'u.s.a.', 'united states', 'united states of america'} -> 'US'
    Returns Series — assign directly: df[col] = standardize_country_codes(df, col).
    """
    raw_map = country_map or COUNTRY_STANDARD_MAP
    mapping = {k.lower().strip(): v for k, v in raw_map.items()}
    return df[column].apply(
        lambda v: mapping.get(str(v).lower().strip(), str(v).strip()) if pd.notna(v) else v
    )


def convert_seconds_to_timestamp(
    df: pd.DataFrame,
    source_column: str,
    target_column: str,
    reference_start: str = REFERENCE_TIMESTAMP,
) -> pd.DataFrame:
    """Convert an elapsed-seconds column to a formatted timestamp string.

    Adds seconds from source_column to reference_start (default '2020-01-01 00:00:00')
    and writes the result as 'YYYY-MM-DD HH:MM:SS' into target_column.
    Returns DataFrame — assign back: df = convert_seconds_to_timestamp(df, 'Time', 'transaction_timestamp').
    """
    df = df.copy()
    ref = datetime.strptime(reference_start, "%Y-%m-%d %H:%M:%S")
    df[target_column] = df[source_column].apply(
        lambda s: (ref + timedelta(seconds=float(s))).strftime("%Y-%m-%d %H:%M:%S")
        if pd.notna(s)
        else None
    )
    return df


def add_ingestion_timestamp(df: pd.DataFrame, target_column: str = "ingestion_timestamp") -> pd.DataFrame:
    """Add a UTC ingestion timestamp column with the current run time.
    Returns DataFrame — assign back: df = add_ingestion_timestamp(df).
    """
    df = df.copy()
    df[target_column] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    return df


def add_transaction_date(df: pd.DataFrame, timestamp_col: str, target_col: str = "transaction_date") -> pd.DataFrame:
    """Extract the date part (YYYY-MM-DD) from a timestamp column into a new column.
    Returns DataFrame — assign back: df = add_transaction_date(df, 'transaction_timestamp').
    """
    df = df.copy()
    df[target_col] = pd.to_datetime(df[timestamp_col]).dt.strftime("%Y-%m-%d")
    return df


def add_transaction_hour(df: pd.DataFrame, timestamp_col: str, target_col: str = "transaction_hour") -> pd.DataFrame:
    """Extract the hour (0-23) from a timestamp column into a new integer column.
    Returns DataFrame — assign back: df = add_transaction_hour(df, 'transaction_timestamp').
    """
    df = df.copy()
    df[target_col] = pd.to_datetime(df[timestamp_col]).dt.hour
    return df


def add_is_weekend(df: pd.DataFrame, timestamp_col: str, target_col: str = "is_weekend") -> pd.DataFrame:
    """Add a binary flag (1=weekend, 0=weekday) derived from a timestamp column.
    Returns DataFrame — assign back: df = add_is_weekend(df, 'transaction_timestamp').
    """
    df = df.copy()
    df[target_col] = pd.to_datetime(df[timestamp_col]).dt.dayofweek.isin([5, 6]).astype(int)
    return df


def add_amount_category(df: pd.DataFrame, amount_col: str, target_col: str = "amount_category") -> pd.DataFrame:
    """Bin transaction amounts into 'low' / 'medium' / 'high' / 'outlier' using quartiles.
    Thresholds are computed from the data: p25=low, p75=medium, p95=high, above=outlier.
    Returns DataFrame — assign back: df = add_amount_category(df, 'transaction_amount').
    """
    df = df.copy()
    p25 = df[amount_col].quantile(0.25)
    p75 = df[amount_col].quantile(0.75)
    p95 = df[amount_col].quantile(0.95)

    def _categorize(value):
        if pd.isna(value):
            return None
        if value < p25:
            return "low"
        if value < p75:
            return "medium"
        if value < p95:
            return "high"
        return "outlier"

    df[target_col] = df[amount_col].apply(_categorize)
    return df


def add_utilization_rate(df: pd.DataFrame, balance_col: str, limit_col: str, target_col: str = "utilization_rate") -> pd.DataFrame:
    """Compute credit utilization as balance/limit, clipped to [0.0, 1.0].
    Returns DataFrame — assign back: df = add_utilization_rate(df, 'account_balance', 'credit_limit').
    """
    df = df.copy()
    df[target_col] = (df[balance_col] / df[limit_col]).clip(0.0, 1.0)
    return df


def add_is_high_risk_context(
    df: pd.DataFrame,
    foreign_col: str,
    declined_col: str,
    distance_col: str,
    target_col: str = "is_high_risk_context",
    declined_threshold: int = 2,
    distance_threshold: float = 500.0,
) -> pd.DataFrame:
    """Add a binary high-risk flag (1=high-risk, 0=normal).
    Flags a row if it is a foreign transaction, has more than declined_threshold
    recent declines, or the merchant distance exceeds distance_threshold km.
    Returns DataFrame — assign back: df = add_is_high_risk_context(df, 'is_foreign_transaction', 'num_declined_7d', 'merchant_distance_km').
    """
    df = df.copy()
    df[target_col] = (
        (df[foreign_col] == 1)
        | (df[declined_col] > declined_threshold)
        | (df[distance_col] > distance_threshold)
    ).astype(int)
    return df


def validate_schema(df: pd.DataFrame, expected_columns: list) -> tuple[bool, list]:
    """Check that all expected_columns are present in df. Returns (passed, missing_list)."""
    missing = [column for column in expected_columns if column not in df.columns]
    return len(missing) == 0, missing


def generate_validation_report(
    df_in: pd.DataFrame,
    df_out: pd.DataFrame,
    rejected_counts: dict,
    report_path: str,
    expected_columns: list | None = None,
) -> None:
    """Write a plain-text validation summary to report_path.
    Must always be called at the end of run() with df_in (original loaded DataFrame
    before any filtering), df_out (final output), and a rejected_counts dict that
    maps filter step names to row counts removed at that step.
    """
    expected_columns = expected_columns or list(df_out.columns)
    schema_ok, missing_columns = validate_schema(df_out, expected_columns)
    fraud_counts = df_out["is_fraud"].value_counts(dropna=False).to_dict() if "is_fraud" in df_out.columns else {}
    amount_counts = df_out["amount_category"].value_counts(dropna=False).to_dict() if "amount_category" in df_out.columns else {}

    lines = [
        "=== Validation Report ===",
        f"Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
        "",
        f"total_records_input:   {len(df_in)}",
        f"total_records_output:  {len(df_out)}",
        f"total_records_removed: {len(df_in) - len(df_out)}",
        "",
        "--- Filtering Counts ---",
    ]
    for key, value in rejected_counts.items():
        lines.append(f"  {key}: {value}")

    lines.extend([
        "",
        "--- Fraud Distribution (output) ---",
        f"  legitimate (0): {fraud_counts.get(0, 0)}",
        f"  fraudulent  (1): {fraud_counts.get(1, 0)}",
        "",
        "--- Amount Category Distribution ---",
    ])
    for bucket in ("low", "medium", "high", "outlier"):
        lines.append(f"  {bucket}: {amount_counts.get(bucket, 0)}")

    lines.extend([
        "",
        "--- Schema Check ---",
        f"  pass: {schema_ok}",
    ])
    if missing_columns:
        lines.append(f"  missing_columns: {', '.join(missing_columns)}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")