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
import inspect
import json
import re
import textwrap
from datetime import datetime, timezone
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import agents.etl_utilities as _etl_utils

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from agents.llm_client import llm_available, make_llm
from agents.prompt_logger import PromptResponseLogger


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
)

INPUT_PATH  = "{source_file}"
OUTPUT_PATH = "data/processed/{target_table}.csv"
"""

_PIPELINE_FOOTER = """\

if __name__ == "__main__":
    run()
"""


# ---------------------------------------------------------------------------
# RAG: Dynamic utility documentation extracted from etl_utilities at runtime
# ---------------------------------------------------------------------------

def _load_utility_docs() -> str:
    """Build the utility reference block by inspecting etl_utilities at call time.

    This replaces the old hardcoded _UTILITY_SIGNATURES string. The docstrings in
    etl_utilities.py are now the single source of truth — no manual sync needed.
    """
    lines = ["Available functions from etl_utilities (pandas DataFrame-based):\n"]
    for name, fn in inspect.getmembers(_etl_utils, inspect.isfunction):
        if name.startswith("_"):
            continue
        sig = inspect.signature(fn)
        doc = inspect.getdoc(fn) or ""
        lines.append(f"  {name}{sig}")
        if doc:
            for doc_line in doc.splitlines():
                lines.append(f"      {doc_line}")
        lines.append("")

    lines.append(
        "  COUNTRY_STANDARD_MAP default keys (already lowercase): "
        + str(list(_etl_utils.COUNTRY_STANDARD_MAP.keys()))
    )
    return "\n".join(lines)


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
        {_load_utility_docs()}

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
        3. Save results to OUTPUT_PATH using save_csv().
        4. Print a completion message.

        Rules:
        - Start with `def run():` and indent all code by 4 spaces.
        - Use only the listed etl_utilities functions. Do not import pandas directly.
        - Do not overwrite the variables INPUT_PATH, OUTPUT_PATH. Use existing ones provided in the pipeline header.
        - Functions returning Series (replace_blank_with, impute_nulls, standardize_country_codes):
          Assign directly to column: df[col] = function(df, col, ...)
        - Functions returning DataFrame (cast_column, add_* functions):
          Assign back to df: df = function(df, ...)
        - Do not hardcode dataset-specific assumptions beyond what exists in mapping JSON.
        - Return only Python code. No markdown, no explanations.
        - Rename columns (REQUIRED): call rename_columns() ONLY for mapping entries where
          source_column != target_column. NEVER include identity mappings (same name on
          both sides) in the rename_columns() call — they cause rename_mapping_applied
          test failures.
        - Type casting (REQUIRED): scan EVERY entry in source_to_target_mapping. For each
          column where target_type contains 'decimal', 'float', 'numeric', or 'double',
          call `df = cast_column(df, 'column_name', 'float64')`. For each column where
          target_type contains 'integer', 'int', 'bigint', 'tinyint', or 'smallint',
          call `df = cast_column(df, 'column_name', 'integer')`. Apply these casts even
          if the source column already appears numeric — CSV round-trips lose dtype
          precision and downstream type checks will fail without explicit casting.
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


_DEVELOPER_SYSTEM_PROMPT = (
    "You are a senior data engineer. Generate clean, executable Python "
    "ETL pipeline code. Return only code, no markdown fences, no explanation."
)


def _strip_code_fences(text: str) -> str:
    """Strip markdown fenced code block wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        start = 1
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end]).strip()
    return text


def _generate_code_llm(
    mapping: dict,
    framework: str,
    data_dictionary: dict | None = None,
    error_feedback: str | None = None,
    test_feedback: dict | None = None,
) -> str:
    user_prompt = _build_developer_prompt(mapping, framework, data_dictionary, test_feedback)
    if error_feedback:
        user_prompt += f"\n\n## Previous execution error (FIX THIS):\n{error_feedback}"

    prompt = ChatPromptTemplate.from_messages([
        ("system", _DEVELOPER_SYSTEM_PROMPT),
        ("human", "{user_prompt}"),
    ])
    chain = (prompt | make_llm() | StrOutputParser()).with_retry(stop_after_attempt=2)
    raw_response = chain.invoke({"user_prompt": user_prompt})
    try:
        PromptResponseLogger().log(
            agent="developer",
            call_type="code_generation",
            system_prompt=_DEVELOPER_SYSTEM_PROMPT,
            human_prompt=user_prompt,
            raw_response=raw_response,
        )
    except Exception:
        pass
    return _strip_code_fences(raw_response)


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

    Retry orchestration (validate=True) is handled by the orchestrator's LangGraph
    state machine. This function always performs a single code-generation pass.

    Args:
        mapping: Source-to-target mapping JSON
        output_path: Where to write the generated pipeline
        framework: "pandas" or "pyspark"
        data_dictionary: Optional data dictionary for context
        validate: Kept for backward compatibility; retry loop is in orchestrator
        max_retries: Kept for backward compatibility
        test_feedback: Optional dict with test_results and generated_code for
                       test-driven regeneration feedback
        requirements_text: Optional requirements text; merged into test_feedback

    Returns the path of the generated file.
    """
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running developer_agent.py."
        )

    if test_feedback and requirements_text:
        test_feedback = dict(test_feedback)
        test_feedback["requirements"] = requirements_text

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
