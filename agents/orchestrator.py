"""
End-to-end orchestration for the GenAI ETL demo.

This script runs all agent stages in sequence:
1) analyst_agent      (LLM mode) -> source_to_target_mapping.json + source_data_dictionary.json
2) developer_agent    (LLM mode) -> generated_pipeline.py
3) optional execution of generated pipeline
4) tester_agent       (LLM-generated tests) -> data/processed/test_report.txt

Usage:
  python agents/orchestrator.py
  python agents/orchestrator.py --framework pyspark
  python agents/orchestrator.py --skip-run

  # Skip analyst and run only the developer-tester feedback loop
  # (requires existing mapping/dictionary outputs from a prior run):
  python agents/orchestrator.py --skip-analyst --validate
  python agents/orchestrator.py --skip-analyst --validate --max-retries 5
"""

import argparse
import json
import os
import subprocess
import sys

from pathlib import Path

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.analyst_agent import analyze_requirements_and_data
from agents.developer_agent import generate_pipeline_code
from agents.llm_client import llm_available
from agents.tester_agent import run_tests


def _run_generated_pipeline(pipeline_path: str) -> None:
    env = os.environ.copy()
    python_executable = sys.executable
    virtual_env = env.get("VIRTUAL_ENV")

    if virtual_env:
        candidate = Path(virtual_env) / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        if candidate.exists():
            python_executable = str(candidate)

    print(f"Running generated pipeline with Python: {python_executable}")
    if os.name == "nt" and python_executable.lower().endswith("py.exe"):
        cmd = [python_executable, "-3", pipeline_path]
    else:
        cmd = [python_executable, pipeline_path]
    result = subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)

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


def _attempt_pipeline_with_testing(
    mapping: dict,
    dictionary: dict,
    requirements_text: str,
    requirements_path: str,
    pipeline_output_path: str,
    test_report_path: str,
    source_csv_path: str,
    framework: str,
    max_retries: int = 3,
) -> dict:
    """
    Attempt to generate, execute, and test a pipeline with test-driven retries.
    
    Returns dict with:
    {
        "success": bool,
        "pipeline_path": str,
        "test_summary": dict or None,
        "attempts": int,
    }
    """
    processed_data_path = "data/processed/fraud_transactions.csv"
    test_feedback = None
    
    for attempt in range(max_retries):
        print(f"[DEVELOPER] Generating ETL pipeline code (attempt {attempt + 1}/{max_retries})...")
        if attempt > 0:
            print(f"[DEVELOPER] ...incorporating test feedback from {attempt} previous attempt(s)...")
        
        # Generate code with optional test feedback
        pipeline_path = generate_pipeline_code(
            mapping=mapping,
            output_path=pipeline_output_path,
            framework=framework,
            data_dictionary=dictionary,
            validate=False,  # Let orchestrator handle execution
            test_feedback=test_feedback,
            requirements_text=requirements_text,
        )
        
        code_length = Path(pipeline_path).read_text(encoding="utf-8").count('\n')
        print(f"[DEVELOPER] ✓ Generated {code_length} lines of Python code")
        
        # Execute the pipeline
        print(f"[EXECUTOR] Running pipeline...")
        try:
            _run_generated_pipeline(pipeline_path)
            print(f"[EXECUTOR] ✓ Pipeline execution succeeded")
        except RuntimeError as e:
            print(f"[EXECUTOR] ✗ Pipeline execution failed: {str(e)[:100]}...")
            if attempt < max_retries - 1:
                print(f"[ORCHESTRATOR] [Retry {attempt + 2}/{max_retries}] Execution failed, regenerating code with error feedback...")
                # Format execution error for feedback
                test_feedback = {
                    "test_results": [{"name": "execution_error", "passed": False, "details": str(e)[:300]}],
                    "generated_code": Path(pipeline_path).read_text(encoding="utf-8"),
                }
                continue
            else:
                print(f"[ORCHESTRATOR] ✗ Pipeline validation failed after {max_retries} attempts")
                return {
                    "success": False,
                    "pipeline_path": pipeline_path,
                    "test_summary": None,
                    "attempts": attempt + 1,
                }
        
        # Run tests
        print(f"[TESTER] Running test suite...")
        test_summary = run_tests(
            requirements_path=requirements_path,
            source_data_path=source_csv_path,
            processed_data_path=processed_data_path,
            report_path=test_report_path,
        )
        
        total_tests = len(test_summary.get("test_results", []))
        passed_tests = sum(1 for t in test_summary.get("test_results", []) if t.get("passed"))
        
        if test_summary.get("all_passed"):
            print(f"[TESTER] ✓ All tests passed! ({passed_tests}/{total_tests})")
            print(f"[ORCHESTRATOR] ✓ Pipeline validation completed successfully")
            return {
                "success": True,
                "pipeline_path": pipeline_path,
                "test_summary": test_summary,
                "attempts": attempt + 1,
            }
        else:
            failed_tests = [t for t in test_summary.get("test_results", []) if not t.get("passed")]
            print(f"[TESTER] ✗ Test failures: {len(failed_tests)} failed / {total_tests} total")
            
            if attempt < max_retries - 1:
                print(f"[ORCHESTRATOR] [Retry {attempt + 2}/{max_retries}] Test failures detected, regenerating code with feedback...")
                # Prepare test feedback for next iteration
                test_feedback = {
                    "test_results": failed_tests,
                    "generated_code": Path(pipeline_path).read_text(encoding="utf-8"),
                }
                continue
            else:
                print(f"[ORCHESTRATOR] ✗ Pipeline validation failed after {max_retries} attempts")
                return {
                    "success": False,
                    "pipeline_path": pipeline_path,
                    "test_summary": test_summary,
                    "attempts": attempt + 1,
                }
    
    # Should not reach here, but as fallback
    return {
        "success": False,
        "pipeline_path": pipeline_output_path,
        "test_summary": None,
        "attempts": max_retries,
    }



def orchestrate(
    requirements_path: str,
    source_csv_path: str,
    mapping_output_path: str,
    dictionary_output_path: str,
    pipeline_output_path: str,
    test_report_path: str,
    framework: str,
    run_generated: bool,
    validate: bool,
    max_retries: int,
    skip_analyst: bool = False,
) -> dict:
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running the orchestrator."
        )

    print(f"[ORCHESTRATOR] Starting ETL pipeline orchestration (max_retries: {max_retries}, validate: {validate}, skip_analyst: {skip_analyst})")
    print()

    requirements_text = Path(requirements_path).read_text(encoding="utf-8")

    if skip_analyst:
        mapping_path = Path(mapping_output_path)
        dictionary_path = Path(dictionary_output_path)
        missing = [str(p) for p in (mapping_path, dictionary_path) if not p.exists()]
        if missing:
            raise FileNotFoundError(
                f"--skip-analyst requires existing analysis files. Missing: {missing}\n"
                "Run without --skip-analyst first to generate them."
            )
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        dictionary = json.loads(dictionary_path.read_text(encoding="utf-8"))
        print(f"[ANALYST] Skipping — loading cached artifacts from disk")
        print(f"[ANALYST] ✓ Loaded mapping:     {mapping_output_path}")
        print(f"[ANALYST] ✓ Loaded dictionary:  {dictionary_output_path}")
    else:
        print("[ANALYST] Analyzing requirements and data...")
        analysis = analyze_requirements_and_data(
            requirements_path=requirements_path,
            source_csv_path=source_csv_path,
            output_mapping_path=mapping_output_path,
            output_dictionary_path=dictionary_output_path,
            use_llm=True,
        )
        mapping = analysis["mapping"]
        dictionary = analysis["dictionary"]
        print(f"[ANALYST] ✓ Extracted mapping: target_table={mapping.get('target_table')}, {len(mapping.get('source_to_target_mapping', []))} source→target transformations, {len(mapping.get('source_columns', []))} source columns")
        print(f"[ANALYST] ✓ Profiled data: {dictionary.get('row_count')} rows, {dictionary.get('column_count')} columns, generated data dictionary")
    print()

    if run_generated:
        if validate:
            # Test-driven retry loop
            print("[ORCHESTRATOR] Attempting pipeline generation with test-driven validation...")
            result = _attempt_pipeline_with_testing(
                mapping=mapping,
                dictionary=dictionary,
                requirements_text=requirements_text,
                requirements_path=requirements_path,
                pipeline_output_path=pipeline_output_path,
                test_report_path=test_report_path,
                source_csv_path=source_csv_path,
                framework=framework,
                max_retries=max_retries,
            )
            
            pipeline_path = result["pipeline_path"]
            test_summary = result["test_summary"]
            tests_passed = result["success"]
            
            print()
            if tests_passed:
                print(f"[ORCHESTRATOR] ✓ SUCCESS: Pipeline generated and validated after {result['attempts']} attempt(s)")
            else:
                print(f"[ORCHESTRATOR] ✗ FAILED: Pipeline validation did not pass after {result['attempts']} attempt(s)")
        else:
            # Simple generation without test-driven loop (legacy mode)
            print("[DEVELOPER] Generating ETL pipeline code...")
            pipeline_path = generate_pipeline_code(
                mapping=mapping,
                output_path=pipeline_output_path,
                framework=framework,
                data_dictionary=dictionary,
                validate=False,
            )
            print(f"[DEVELOPER] ✓ Generated pipeline: {pipeline_output_path}")
            
            print("[EXECUTOR] Running pipeline...")
            _run_generated_pipeline(pipeline_path)
            print("[EXECUTOR] ✓ Pipeline execution succeeded")
            
            print("[TESTER] Running test suite...")
            test_summary = run_tests(
                requirements_path=requirements_path,
                source_data_path=source_csv_path,
                processed_data_path="data/processed/fraud_transactions.csv",
                report_path=test_report_path,
            )
            tests_passed = test_summary.get("all_passed")
            print(f"[TESTER] {'✓' if tests_passed else '✗'} Tests {'passed' if tests_passed else 'failed'}")
            print()
    else:
        print("[ORCHESTRATOR] Skipping pipeline generation and execution (--skip-run)")
        print("[DEVELOPER] Generating ETL pipeline code (no execution)...")
        pipeline_path = generate_pipeline_code(
            mapping=mapping,
            output_path=pipeline_output_path,
            framework=framework,
            data_dictionary=dictionary,
            validate=False,
        )
        print(f"[DEVELOPER] ✓ Generated pipeline: {pipeline_output_path}")
        test_summary = None
        tests_passed = None
        print()

    return {
        "mapping_output": mapping_output_path,
        "dictionary_output": dictionary_output_path,
        "pipeline_output": pipeline_path,
        "test_report": test_report_path if test_summary is not None else None,
        "target_table": mapping.get("target_table"),
        "row_count": dictionary.get("row_count"),
        "column_count": dictionary.get("column_count"),
        "tests_passed": tests_passed,
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
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Execute pipeline with test-driven validation (retries on test failures)",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum number of retries for each ETL step",
    )
    parser.add_argument(
        "--skip-analyst",
        action="store_true",
        help=(
            "Skip the analyst step and load existing mapping/dictionary from disk. "
            "Runs only the developer-tester feedback loop, saving analyst LLM calls. "
            "Requires outputs from a prior full run."
        ),
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
        validate=args.validate,
        max_retries=args.max_retries,
        skip_analyst=args.skip_analyst,
    )

    print("\nOrchestration complete")
    print(json.dumps(summary, indent=2))
