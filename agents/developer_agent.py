# agents/developer_agent.py
"""
Developer Agent — generates executable ETL pipeline code from a
source-to-target mapping JSON produced by analyst_agent.py.

This agent is LLM-based and can optionally validate generated code by executing it.

Usage:
  python agents/developer_agent.py
  python agents/developer_agent.py --mapping outputs/source_to_target_mapping.json
  python agents/developer_agent.py --output outputs/generated_pipeline.py
  python agents/developer_agent.py --framework pyspark
  python agents/developer_agent.py --validate --max-retries 3
"""

import argparse
import json
import re
import subprocess
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
  impute_nulls(df, column: str, strategy: str) -> Series
      strategy: median | mean | zero
  replace_blank_with(df, column: str, replacement: str) -> Series
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
  standardize_country_codes(df, column: str, country_map: dict=None) -> Series
"""


# ---------------------------------------------------------------------------
# LLM-based code generator
# ---------------------------------------------------------------------------


def _build_developer_prompt(
    mapping: dict,
    framework: str,
    data_dictionary: dict | None = None,
    test_feedback: dict | None = None,
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

    base_prompt = textwrap.dedent(
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
        - Functions returning Series (replace_blank_with, impute_nulls, standardize_country_codes):
          Assign directly to column: df[col] = function(df, col, ...)
        - Functions returning DataFrame (cast_column, add_* functions):
          Assign back to df: df = function(df, ...)
        - Always call `generate_validation_report(df_input, df, rejected_counts, REPORT_PATH)`.
        - Do not hardcode dataset-specific assumptions beyond what exists in mapping JSON.
        - Return only Python code. No markdown, no explanations.
        """
    ).strip()

    # If test feedback is provided, append it to the prompt
    if test_feedback:
        test_results = test_feedback.get("test_results", [])
        generated_code = test_feedback.get("generated_code", "")
        requirements = test_feedback.get("requirements", "")
        
        feedback_section = textwrap.dedent(
            f"""

            ## CRITICAL: Fix Test Failures
            Your previous code attempt failed the following tests. You MUST fix these issues:

            ### Failed Tests:
            """
        ).strip()
        
        for test in test_results:
            test_name = test.get("name", "unknown")
            test_details = test.get("details", "no details")
            feedback_section += f"\n  - **{test_name}**: {test_details}"
        
        feedback_section += textwrap.dedent(
            f"""

            ### Previous Generated Code (NEEDS FIXES):
            ```python
            {generated_code}
            ```

            ### Original Requirements:
            {requirements}

            ### Fix Strategy:
            1. Analyze the test failures above to understand what went wrong.
            2. Review your previous code and identify the root cause.
            3. Rewrite the code to fix the failures while maintaining compliance with requirements.
            4. Use only the listed etl_utilities functions.
            5. Return ONLY the fixed Python code. No explanations.
            """
        ).strip()
        
        return base_prompt + "\n" + feedback_section
    
    return base_prompt


def _generate_code_llm(
    mapping: dict,
    framework: str,
    data_dictionary: dict | None = None,
    error_feedback: str | None = None,
    test_feedback: dict | None = None,
) -> str:
    system_prompt = (
        "You are a senior data engineer. Generate clean, executable Python "
        "ETL pipeline code. Return only code, no markdown fences, no explanation."
    )
    user_prompt = _build_developer_prompt(mapping, framework, data_dictionary, test_feedback)
    
    if error_feedback:
        user_prompt += f"\n\n## Previous execution error (FIX THIS):\n{error_feedback}"
    
    return call_llm_text(system_prompt=system_prompt, user_prompt=user_prompt)


def _execute_pipeline(pipeline_path: str) -> tuple[bool, str]:
    """
    Execute the generated pipeline and return (success: bool, output: str).
    """
    try:
        result = subprocess.run(
            ["python", str(pipeline_path)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode == 0:
            return True, result.stdout
        else:
            # Extract error from stderr
            error_msg = result.stderr or result.stdout or "Unknown error"
            return False, error_msg
    except subprocess.TimeoutExpired:
        return False, "Pipeline execution timed out (>60 seconds)"
    except Exception as e:
        return False, f"Execution failed: {str(e)}"


def _format_error_for_feedback(error_output: str) -> str:
    """
    Parse execution error and format it for LLM feedback.
    Extracts the most relevant error information.
    """
    # Try to extract the actual error line
    lines = error_output.split('\n')
    
    # Look for traceback
    for i, line in enumerate(lines):
        if 'Traceback' in line or 'Error' in line or 'error' in line:
            # Collect 5-10 lines from the error
            relevant_lines = lines[i:min(i + 10, len(lines))]
            return '\n'.join(relevant_lines)
    
    # Fallback: return last 5 non-empty lines
    non_empty = [l for l in lines if l.strip()]
    return '\n'.join(non_empty[-5:])


def _generate_pipeline_with_retry(
    mapping: dict,
    output_path: str,
    framework: str = "pandas",
    data_dictionary: dict | None = None,
    max_retries: int = 3,
    test_feedback: dict | None = None,
) -> tuple[str, bool]:
    """
    Generate and validate a pipeline, with auto-troubleshooting for up to max_retries.
    Returns (output_path, success: bool)
    """
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running developer_agent.py."
        )

    target_table = mapping.get("target_table", "output_table")
    source_file = mapping.get("source_file", "data/raw/sample_transactions.csv")

    for attempt in range(max_retries):
        error_feedback = None if attempt == 0 else _format_error_for_feedback(last_error)
        # Use test_feedback on first attempt (if provided), switch to error_feedback on retries from execution
        current_test_feedback = test_feedback if attempt == 0 else None

        run_body = _generate_code_llm(
            mapping, framework, data_dictionary, error_feedback, current_test_feedback
        )

        header = _PIPELINE_HEADER.format(
            target_table=target_table,
            source_file=source_file,
            framework=framework,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            mode=f"llm_only (retry {attempt})" if attempt > 0 else "llm_only",
        )

        full_code = header + "\n\n" + run_body + "\n" + _PIPELINE_FOOTER

        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full_code, encoding="utf-8")

        # Try to execute
        success, output = _execute_pipeline(str(out))

        if success:
            print(f"[Attempt {attempt + 1}/{max_retries}] Pipeline execution succeeded")
            return str(out), True

        last_error = output
        print(
            f"[Attempt {attempt + 1}/{max_retries}] Pipeline execution failed. "
            f"Error: {last_error[:200]}..."
        )

        if attempt < max_retries - 1:
            print(f"  Retrying with feedback...")

    # Failed all retries
    print(f"\n[FAILED] Pipeline did not pass validation after {max_retries} attempts")
    print(f"Last error: {last_error}")
    return str(out), False


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def generate_pipeline_code(
    mapping: dict,
    output_path: str,
    framework: str = "pandas",
    data_dictionary: dict | None = None,
    validate: bool = False,
    max_retries: int = 3,
    test_feedback: dict | None = None,
    requirements_text: str | None = None,
) -> str:
    """
    Generate an executable ETL pipeline Python file from source-to-target mapping.

    Args:
        mapping: Source-to-target mapping JSON
        output_path: Where to write the generated pipeline
        framework: "pandas" or "pyspark"
        data_dictionary: Optional data dictionary for context
        validate: If True, execute pipeline and retry up to max_retries times on error
        max_retries: Maximum number of retry attempts (only used if validate=True)
        test_feedback: Optional dict with test_results, generated_code, requirements for test-driven regeneration
        requirements_text: Optional requirements document text (used in test_feedback context)

    Returns the path of the generated file.
    """
    # Integrate requirements_text into test_feedback if provided
    if test_feedback and requirements_text:
        test_feedback = dict(test_feedback)  # Make a copy to avoid mutation
        test_feedback["requirements"] = requirements_text
    
    if validate:
        path, success = _generate_pipeline_with_retry(
            mapping=mapping,
            output_path=output_path,
            framework=framework,
            data_dictionary=data_dictionary,
            max_retries=max_retries,
            test_feedback=test_feedback,
        )
        return path

    # Non-validating path (original behavior)
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running developer_agent.py."
        )

    target_table = mapping.get("target_table", "output_table")
    source_file = mapping.get("source_file", "data/raw/sample_transactions.csv")

    run_body = _generate_code_llm(
        mapping, framework, data_dictionary, error_feedback=None, test_feedback=test_feedback
    )

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
        help="Source-to-target mapping JSON produced by analyst_agent.py",
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Execute the pipeline and auto-fix errors (up to 3 attempts)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retry attempts when --validate is enabled",
    )
    args = parser.parse_args()

    mapping_data = json.loads(Path(args.mapping).read_text(encoding="utf-8"))

    out_path = generate_pipeline_code(
        mapping=mapping_data,
        output_path=args.output,
        framework=args.framework,
        validate=args.validate,
        max_retries=args.max_retries,
    )

    print(f"Generated pipeline written to: {out_path}")
    print(f"Target table : {mapping_data.get('target_table')}")
    print(f"Framework    : {args.framework}")
    print(f"Mode         : {'validated' if args.validate else 'llm_only'}")
    if args.validate:
        print(f"Max retries  : {args.max_retries}")
