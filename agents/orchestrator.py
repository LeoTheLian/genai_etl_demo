"""
End-to-end orchestration for the GenAI ETL demo.

This script runs all agent stages in sequence:
1) requirements_parser (LLM mode) -> source_to_target_mapping.json
2) data_profiler      (LLM mode) -> source_data_dictionary.json
3) developer_agent    (LLM mode) -> generated_pipeline.py
4) optional execution of generated pipeline
5) testing_agent      (LLM-generated tests) -> data/processed/test_report.txt

Usage:
  python agents/orchestrator.py
  python agents/orchestrator.py --framework pyspark
  python agents/orchestrator.py --skip-run
"""

import argparse
import json
import subprocess
import sys

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.data_profiler import build_data_dictionary
from agents.developer_agent import generate_pipeline_code
from agents.llm_client import llm_available
from agents.requirements_parser import save_source_to_target_mapping
from agents.testing_agent import run_tests


def _run_generated_pipeline(pipeline_path: str) -> None:
    cmd = [sys.executable, pipeline_path]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True)

    if result.stdout:
        print("\n[generated_pipeline stdout]")
        print(result.stdout.strip())
    if result.stderr:
        print("\n[generated_pipeline stderr]")
        print(result.stderr.strip())

    if result.returncode != 0:
        raise RuntimeError(
            f"Generated pipeline failed with exit code {result.returncode}: {pipeline_path}"
        )


def orchestrate(
    requirements_path: str,
    source_csv_path: str,
    mapping_output_path: str,
    dictionary_output_path: str,
    pipeline_output_path: str,
    test_report_path: str,
    framework: str,
    run_generated: bool,
) -> dict:
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running the orchestrator."
        )

    print("Step 1/5: Generating source-to-target mapping with requirements_parser (LLM)...")
    mapping = save_source_to_target_mapping(
        requirements_path=requirements_path,
        output_path=mapping_output_path,
        use_llm=True,
    )
    print(f"  Done: {mapping_output_path}")

    print("Step 2/5: Profiling source data with data_profiler (LLM enrichment)...")
    dictionary = build_data_dictionary(
        file_path=source_csv_path,
        output_path=dictionary_output_path,
        use_llm=True,
    )
    print(f"  Done: {dictionary_output_path}")

    print("Step 3/5: Generating pipeline code with developer_agent (LLM-only)...")
    pipeline_path = generate_pipeline_code(
        mapping=mapping,
        output_path=pipeline_output_path,
        framework=framework,
        data_dictionary=dictionary,
    )
    print(f"  Done: {pipeline_path}")

    if run_generated:
        print("Step 4/5: Running generated pipeline...")
        _run_generated_pipeline(pipeline_path)
        print("  Done: generated pipeline executed successfully")

        print("Step 5/5: Validating processed data with testing_agent...")
        test_summary = run_tests(
            requirements_path=requirements_path,
            source_data_path=source_csv_path,
            processed_data_path="data/processed/fraud_transactions.csv",
            report_path=test_report_path,
        )
        print(f"  Done: {test_report_path}")
    else:
        print("Step 4/5: Skipped generated pipeline execution (--skip-run)")
        print("Step 5/5: Skipped testing because generated pipeline was not executed")
        test_summary = None

    return {
        "mapping_output": mapping_output_path,
        "dictionary_output": dictionary_output_path,
        "pipeline_output": pipeline_path,
        "test_report": test_report_path if test_summary is not None else None,
        "target_table": mapping.get("target_table"),
        "row_count": dictionary.get("row_count"),
        "column_count": dictionary.get("column_count"),
        "tests_passed": None if test_summary is None else test_summary.get("all_passed"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the GenAI ETL flow end-to-end in LLM mode."
    )
    parser.add_argument(
        "--requirements",
        default="data/raw/requirements_document.txt",
        help="Path to requirements document text file",
    )
    parser.add_argument(
        "--input",
        default="data/raw/sample_transactions.csv",
        help="Path to source CSV for profiling and pipeline input",
    )
    parser.add_argument(
        "--mapping-output",
        default="outputs/source_to_target_mapping.json",
        help="Path for LLM-generated source-to-target mapping JSON",
    )
    parser.add_argument(
        "--dictionary-output",
        default="outputs/source_data_dictionary.json",
        help="Path for LLM-enriched source data dictionary JSON",
    )
    parser.add_argument(
        "--pipeline-output",
        default="outputs/generated_pipeline.py",
        help="Path for generated ETL pipeline Python script",
    )
    parser.add_argument(
        "--test-report",
        default="data/processed/test_report.txt",
        help="Path for processed-data test report",
    )
    parser.add_argument(
        "--framework",
        choices=["pandas", "pyspark"],
        default="pandas",
        help="Target execution framework for generated pipeline",
    )
    parser.add_argument(
        "--skip-run",
        action="store_true",
        help="Generate artifacts but do not execute the generated pipeline",
    )
    args = parser.parse_args()

    summary = orchestrate(
        requirements_path=args.requirements,
        source_csv_path=args.input,
        mapping_output_path=args.mapping_output,
        dictionary_output_path=args.dictionary_output,
        pipeline_output_path=args.pipeline_output,
        test_report_path=args.test_report,
        framework=args.framework,
        run_generated=not args.skip_run,
    )

    print("\nOrchestration complete")
    print(json.dumps(summary, indent=2))
