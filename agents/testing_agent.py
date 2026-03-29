"""
Testing Agent — generates and runs data-quality tests from requirements.

This agent is LLM-driven by design. It reads the requirements document,
source dataset, and processed dataset; asks an LLM to generate deterministic
test functions; executes those tests; and writes a human-readable report.
"""

import argparse
import json
import math
import re
import textwrap
from pathlib import Path

if __package__ is None or __package__ == "":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from agents.llm_client import call_llm_text, llm_available
from agents.requirements_parser import _parse_requirements_rule_based


def _load_requirements(path: str) -> tuple[str, dict]:
    text = Path(path).read_text(encoding="utf-8")
    parsed = _parse_requirements_rule_based(text)
    return text, parsed


def _validate_input_paths(
    requirements_path: str,
    source_data_path: str,
    processed_data_path: str,
) -> list[dict]:
    issues: list[dict] = []

    req_path = Path(requirements_path)
    if not req_path.exists():
        issues.append(_result("requirements_path_exists", False, f"missing path: {req_path}"))
    elif not req_path.is_file():
        issues.append(_result("requirements_path_exists", False, f"not a file: {req_path}"))

    source_path = Path(source_data_path)
    if not source_path.exists():
        issues.append(_result("source_data_path_exists", False, f"missing path: {source_path}"))
    elif not source_path.is_file():
        issues.append(_result("source_data_path_exists", False, f"not a file: {source_path}"))
    elif source_path.suffix.lower() != ".csv":
        issues.append(
            _result(
                "source_data_path_csv",
                False,
                f"expected .csv file extension, got: {source_path.suffix or '<none>'}",
            )
        )

    data_path = Path(processed_data_path)
    if not data_path.exists():
        issues.append(_result("processed_data_path_exists", False, f"missing path: {data_path}"))
    elif not data_path.is_file():
        issues.append(_result("processed_data_path_exists", False, f"not a file: {data_path}"))
    elif data_path.suffix.lower() != ".csv":
        issues.append(
            _result(
                "processed_data_path_csv",
                False,
                f"expected .csv file extension, got: {data_path.suffix or '<none>'}",
            )
        )

    return issues


def _result(name: str, passed: bool, details: str) -> dict:
    return {"name": str(name), "passed": bool(passed), "details": str(details)}


def _build_testing_prompt(
    requirements_text: str,
    parsed_requirements: dict,
    source_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> str:
    requirements_json = json.dumps(parsed_requirements, indent=2)
    source_snapshot = json.dumps(
        {
            "row_count": int(len(source_df)),
            "columns": list(source_df.columns),
            "dtypes": {column: str(dtype) for column, dtype in source_df.dtypes.items()},
        },
        indent=2,
    )
    processed_snapshot = json.dumps(
        {
            "row_count": int(len(processed_df)),
            "columns": list(processed_df.columns),
            "dtypes": {column: str(dtype) for column, dtype in processed_df.dtypes.items()},
        },
        indent=2,
    )

    return textwrap.dedent(
        f"""
        You are a senior data quality engineer.

        Generate ONLY executable Python code for a function named:
                    generate_test_results(source_df, df, parsed_requirements)

                Notes:
                - `source_df` is the raw input dataset before ETL.
                - `df` is the processed output dataset.
                - For transformation validations (renames, type casting, filtering,
                    and derivations), compare source_df vs df whenever applicable.

                The function must:
        1. Return a list of dictionaries, each with keys:
           - name (str)
           - passed (bool)
           - details (str)
        2. Implement checks inferred from the requirements.
        3. Be deterministic, pure (no file I/O), and robust to missing columns.
        4. Use pandas operations only (assume pd is already imported).
        5. Catch edge cases and return failing test rows as summary text, not stack traces.

                Required baseline checks:
                Tier 1 (always include):
                - dataset_non_empty
                - row_count_sanity
                - expected_columns_present
                - required_fields_not_null
                - id_uniqueness
                - duplicate_rows_absent
                - numeric_type_coercion_success
                - timestamp_parse_success

                Tier 2 (requirements-driven, extract Testing Requirements from requirements_document if available):
                Precedence rules:
                - Requirement-specific constraints override generic defaults.
                - If a check is not inferable from requirements/schema, skip it gracefully.

                Required implementation details for two key checks:

                1) expected_columns_present
                - Must compare output columns against parsed_requirements["target_columns_expected"]
                    (target schema), not raw source columns.
                - Must account for rename intent in requirements (for example Time ->
                    transaction_timestamp, Amount -> transaction_amount, Class -> is_fraud).
                - In details, explicitly include:
                    - expected_target_count
                    - present_target_count
                    - missing_target_columns (if any)
                    - rename_coverage summary using parsed_requirements.get("rename_rules", {{}})
                    - whether legacy source names still appear in output (informational)

                                2) row_count_sanity
                - Must coordinate with requirements-based filtering expectations.
                                - Use source_df row count as first-class input baseline; use requirements
                                    text row count as a secondary cross-check when available (for example
                                    "Records: 1,000").
                - Infer required-removal expectations from required cleaning rules and
                    estimated counts (for example negative amount, negative distance,
                    invalid age).
                - Treat optional filtering/imputation rules as informational unless the
                    requirements make them mandatory.
                - In details, explicitly include:
                                        - source_row_count
                    - input_row_count_from_requirements (or "unknown")
                    - output_row_count
                    - expected_required_removals_estimate
                                        - observed_removed_count (source_row_count - output_row_count)
                    - pass/fail rationale tied to required vs optional filtering guidance

                                3) transformation_comparison_sanity
                                - Include at least one dedicated comparison test that validates key
                                    transformation intent from source_df to df (for example rename mapping
                                    coverage, castability from source types to target expectations, and
                                    required filtering effect).
                                - In details, summarize compared columns and mismatches.

                Reporting requirements for each test result:
                - Keep test names stable and snake_case.
                - Include actionable details: fail_count (if relevant), affected columns,
                    and sample invalid values or sample failing row indexes.
                - Avoid stack traces in details; return concise business-readable summaries.

                Failure handling defaults:
                - Tier 1 checks should fail when violated.
                - outlier_rate_sanity should be informational unless requirements explicitly
                    define a hard threshold.

        Rules:
        - Start with `def generate_test_results(source_df, df, parsed_requirements):`
        - Do not include markdown fences or explanations.
        - Do not import modules.
        - Use only variables available in the function scope and `pd`.
        - Ensure the function always returns a list, even on partial failures.

        Requirements document:
        {requirements_text}

        Parsed requirements JSON:
        {requirements_json}

        Source data snapshot:
        {source_snapshot}

        Processed data snapshot:
        {processed_snapshot}
        """
    ).strip()


def _generate_tests_code_llm(
    requirements_text: str,
    parsed_requirements: dict,
    source_df: pd.DataFrame,
    processed_df: pd.DataFrame,
) -> str:
    system_prompt = (
        "You are a senior Python data-quality engineer. Return only Python code "
        "for the requested function with no markdown."
    )
    user_prompt = _build_testing_prompt(
        requirements_text,
        parsed_requirements,
        source_df,
        processed_df,
    )
    return call_llm_text(system_prompt=system_prompt, user_prompt=user_prompt)


def _compile_generated_tests(code: str):
    namespace: dict = {}
    globals_dict = {
        "__builtins__": __builtins__,
        "pd": pd,
        "math": math,
        "re": re,
    }
    exec(code, globals_dict, namespace)

    fn = namespace.get("generate_test_results") or globals_dict.get("generate_test_results")
    if not callable(fn):
        raise RuntimeError(
            "Generated test code must define callable "
            "generate_test_results(source_df, df, parsed_requirements)"
        )
    return fn


def _validate_generated_results(results: object) -> list[dict]:
    if not isinstance(results, list):
        raise RuntimeError("Generated test function must return a list of result dictionaries")

    normalized: list[dict] = []
    for item in results:
        if not isinstance(item, dict):
            raise RuntimeError("Each generated test result must be a dictionary")
        normalized.append(
            _result(
                item.get("name", "unnamed_test"),
                item.get("passed", False),
                item.get("details", ""),
            )
        )
    return normalized


def run_tests(
    requirements_path: str,
    source_data_path: str,
    processed_data_path: str,
    report_path: str,
    generated_tests_output_path: str = "outputs/generated_tests.py",
) -> dict:
    path_issues = _validate_input_paths(
        requirements_path,
        source_data_path,
        processed_data_path,
    )
    if path_issues:
        all_passed = all(item["passed"] for item in path_issues)
        summary = {
            "requirements_path": requirements_path,
            "source_data_path": source_data_path,
            "processed_data_path": processed_data_path,
            "generated_tests_output_path": generated_tests_output_path,
            "row_count": 0,
            "all_passed": all_passed,
            "test_results": path_issues,
        }

        lines = [
            "=== Test Report ===",
            f"requirements_path: {requirements_path}",
            f"source_data_path: {source_data_path}",
            f"processed_data_path: {processed_data_path}",
            f"generated_tests_output_path: {generated_tests_output_path}",
            "row_count: 0",
            f"all_passed: {all_passed}",
            "",
            "--- Test Results ---",
        ]
        for result in path_issues:
            status = "PASS" if result["passed"] else "FAIL"
            lines.append(f"{status} | {result['name']} | {result['details']}")

        Path(report_path).parent.mkdir(parents=True, exist_ok=True)
        Path(report_path).write_text("\n".join(lines), encoding="utf-8")
        return summary

    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running testing_agent.py."
        )

    requirements_text, parsed_requirements = _load_requirements(requirements_path)
    source_df = pd.read_csv(source_data_path)
    df = pd.read_csv(processed_data_path)

    generated_code = _generate_tests_code_llm(
        requirements_text,
        parsed_requirements,
        source_df,
        df,
    )

    generated_tests_path = Path(generated_tests_output_path)
    generated_tests_path.parent.mkdir(parents=True, exist_ok=True)
    generated_tests_path.write_text(generated_code + "\n", encoding="utf-8")

    test_fn = _compile_generated_tests(generated_code)

    try:
        raw_results = test_fn(source_df, df, parsed_requirements)
    except Exception as exc:
        raw_results = [
            _result(
                "generated_tests_runtime",
                False,
                f"generated test execution failed: {type(exc).__name__}: {exc}",
            )
        ]

    results = _validate_generated_results(raw_results)

    all_passed = all(result["passed"] for result in results)
    summary = {
        "requirements_path": requirements_path,
        "source_data_path": source_data_path,
        "processed_data_path": processed_data_path,
        "generated_tests_output_path": str(generated_tests_path),
        "source_row_count": int(len(source_df)),
        "row_count": int(len(df)),
        "all_passed": all_passed,
        "test_results": results,
    }

    lines = [
        "=== Test Report ===",
        f"requirements_path: {requirements_path}",
        f"source_data_path: {source_data_path}",
        f"processed_data_path: {processed_data_path}",
        f"generated_tests_output_path: {generated_tests_path}",
        f"source_row_count: {len(source_df)}",
        f"row_count: {len(df)}",
        f"all_passed: {all_passed}",
        "",
        "--- Test Results ---",
    ]
    for result in results:
        status = "PASS" if result["passed"] else "FAIL"
        lines.append(f"{status} | {result['name']} | {result['details']}")

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text("\n".join(lines), encoding="utf-8")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate processed data against the requirements document."
    )
    parser.add_argument(
        "--requirements",
        default="data/raw/requirements_document.txt",
        help="Path to requirements document text file",
    )
    parser.add_argument(
        "--source-data",
        default="data/raw/sample_transactions.csv",
        help="Path to source/raw CSV used for transformation comparison checks",
    )
    parser.add_argument(
        "--input",
        default="data/processed/fraud_transactions.csv",
        help="Path to processed CSV to validate",
    )
    parser.add_argument(
        "--report",
        default="data/processed/test_report.txt",
        help="Path to test report output file",
    )
    parser.add_argument(
        "--generated-tests-output",
        default="outputs/generated_tests.py",
        help="Path to generated Python test script output file",
    )
    args = parser.parse_args()

    result = run_tests(
        args.requirements,
        args.source_data,
        args.input,
        args.report,
        generated_tests_output_path=args.generated_tests_output,
    )
    print(json.dumps(result, indent=2))