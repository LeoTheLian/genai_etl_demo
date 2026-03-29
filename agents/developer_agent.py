# agents/developer_agent.py
"""
Developer Agent — generates executable ETL pipeline code from a
source-to-target mapping JSON produced by requirements_parser.py.

This agent is LLM-only by design.

Usage:
  python agents/developer_agent.py
    python agents/developer_agent.py --mapping outputs/source_to_target_mapping.json
  python agents/developer_agent.py --output outputs/generated_pipeline.py
  python agents/developer_agent.py --framework pyspark
"""

import argparse
import json
import textwrap
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.llm_client import call_llm_text, llm_available


# ---------------------------------------------------------------------------
# Code template constants
# ---------------------------------------------------------------------------

_PIPELINE_HEADER = """\
# =============================================================================
# Generated ETL Pipeline
# Target table : {target_table}
# Source file  : {source_file}
# Framework    : {framework}
# Generated at : {generated_at}
# Mode         : {mode}
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

INPUT_PATH  = "{source_file}"
OUTPUT_PATH = "data/processed/{target_table}.csv"
REPORT_PATH = "data/processed/validation_report.txt"
"""

_PIPELINE_FOOTER = """\

if __name__ == "__main__":
    run()
"""


# ---------------------------------------------------------------------------
# ETL utility signatures exposed to the LLM in its system prompt
# ---------------------------------------------------------------------------

_UTILITY_SIGNATURES = """
Available functions from etl_utilities (pandas DataFrame-based):

  load_csv(path: str) -> DataFrame
  save_csv(df, path: str) -> None
  rename_columns(df, rename_map: dict) -> DataFrame
  filter_non_negative(df, column: str) -> (clean_df, rejected_df)
  filter_positive(df, column: str) -> (clean_df, rejected_df)
  filter_range(df, column: str, min_val, max_val) -> (clean_df, rejected_df)
  filter_valid_values(df, column: str, valid_values: list) -> (clean_df, rejected_df)
  filter_not_null(df, columns: list) -> (clean_df, rejected_df)
  cast_column(df, column: str, target_type: str) -> DataFrame
      target_type: float64 | int32 | int8 | decimal | string | timestamp
  impute_nulls(df, column: str, strategy: str) -> DataFrame
      strategy: median | mean | zero
  replace_blank_with(df, column: str, replacement: str) -> DataFrame
  convert_seconds_to_timestamp(df, source_column, target_column,
                               reference_start="2020-01-01 00:00:00") -> DataFrame
  add_ingestion_timestamp(df, target_column="ingestion_timestamp") -> DataFrame
  add_transaction_date(df, timestamp_col, target_col="transaction_date") -> DataFrame
  add_transaction_hour(df, timestamp_col, target_col="transaction_hour") -> DataFrame
  add_is_weekend(df, timestamp_col, target_col="is_weekend") -> DataFrame
  add_amount_category(df, amount_col, target_col="amount_category") -> DataFrame
  add_utilization_rate(df, balance_col, limit_col, target_col="utilization_rate") -> DataFrame
  add_is_high_risk_context(df, foreign_col, declined_col, distance_col,
                           target_col="is_high_risk_context",
                           declined_threshold=2, distance_threshold=500.0) -> DataFrame
  standardize_country_codes(df, column: str, country_map: dict=None) -> DataFrame
  generate_validation_report(df_in, df_out, rejected_counts: dict, report_path: str) -> None
"""


# ---------------------------------------------------------------------------
# LLM-based code generator
# ---------------------------------------------------------------------------


def _build_developer_prompt(
    mapping: dict,
    framework: str,
    data_dictionary: dict | None = None,
) -> str:
    mapping_json = json.dumps(mapping, indent=2)
    dictionary_json = json.dumps(data_dictionary, indent=2) if data_dictionary else None
    framework_note = (
        "Use pandas DataFrames and the etl_utilities functions listed above."
        if framework == "pandas"
        else (
            "Write PySpark code. Use the PySpark equivalents documented in the "
            "etl_utilities docstrings. Import SparkSession at the top."
        )
    )

    return textwrap.dedent(
        f"""
        You are a senior data engineer generating an ETL pipeline script.

        ## Available utility functions
        {_UTILITY_SIGNATURES}

        ## Framework
        {framework_note}

        ## Full source-to-target mapping JSON
        ```json
        {mapping_json}
        ```

        ## Source data dictionary
        {f'```json\n{dictionary_json}\n```' if dictionary_json else 'Not provided.'}

        ## Task
        Generate ONLY valid Python code for a function called `run()`.

        The function must be fully mapping-driven and generic:
        1. Load source data from INPUT_PATH using load_csv().
        2. Apply transformations defined by the mapping JSON (renames, filters,
           cleaning, type casting, derived or target-only fields).
        3. Maintain a rejected_counts dictionary when any filtering/rejections occur.
        4. Save results to OUTPUT_PATH using save_csv().
        5. Write validation summary using
           generate_validation_report(df_input, df, rejected_counts, REPORT_PATH).
        6. Print a completion message.

        Rules:
        - Start with `def run():` and indent all code by 4 spaces.
        - Use only the listed etl_utilities functions. Do not import pandas directly.
        - Preserve the original loaded DataFrame in a variable named `df_input` before any filtering or mutation.
        - Do not overwrite the variables INPUT_PATH, OUTPUT_PATH, REPORT_PATH. Use existing ones provided in the pipeline header.
        - Always call `generate_validation_report(df_input, df, rejected_counts, REPORT_PATH)`.
        - Do not hardcode dataset-specific assumptions beyond what exists in mapping JSON.
        - Return only Python code. No markdown, no explanations.
        """
    ).strip()


def _generate_code_llm(
    mapping: dict,
    framework: str,
    data_dictionary: dict | None = None,
) -> str:
    system_prompt = (
        "You are a senior data engineer. Generate clean, executable Python "
        "ETL pipeline code. Return only code, no markdown fences, no explanation."
    )
    user_prompt = _build_developer_prompt(mapping, framework, data_dictionary)
    return call_llm_text(system_prompt=system_prompt, user_prompt=user_prompt)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_pipeline_code(
    mapping: dict,
    output_path: str,
    framework: str = "pandas",
    data_dictionary: dict | None = None,
) -> str:
    """
    Generate an executable ETL pipeline Python file from source-to-target mapping.

    Returns the path of the generated file.
    """
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running developer_agent.py."
        )

    target_table = mapping.get("target_table", "output_table")
    source_file = mapping.get("source_file", "data/raw/sample_transactions.csv")

    run_body = _generate_code_llm(mapping, framework, data_dictionary)

    header = _PIPELINE_HEADER.format(
        target_table=target_table,
        source_file=source_file,
        framework=framework,
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        mode="llm_only",
    )

    full_code = header + "\n\n" + run_body + "\n" + _PIPELINE_FOOTER

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(full_code, encoding="utf-8")

    return str(out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Developer Agent: read source_to_target_mapping.json and generate "
            "an executable ETL pipeline Python file using an LLM."
        )
    )
    parser.add_argument(
        "--mapping",
        default="outputs/source_to_target_mapping.json",
        help="Source-to-target mapping JSON produced by requirements_parser.py",
    )
    parser.add_argument(
        "--output",
        default="outputs/generated_pipeline.py",
        help="Path where the generated pipeline script will be written",
    )
    parser.add_argument(
        "--framework",
        choices=["pandas", "pyspark"],
        default="pandas",
        help="Target execution framework for the generated code",
    )
    args = parser.parse_args()

    mapping_data = json.loads(Path(args.mapping).read_text(encoding="utf-8"))

    out_path = generate_pipeline_code(
        mapping=mapping_data,
        output_path=args.output,
        framework=args.framework,
    )

    print(f"Generated pipeline written to: {out_path}")
    print(f"Target table : {mapping_data.get('target_table')}")
    print(f"Framework    : {args.framework}")
    print("Mode         : llm_only")
