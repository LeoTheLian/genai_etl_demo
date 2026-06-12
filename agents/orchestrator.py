"""
End-to-end orchestration for the GenAI ETL demo.

Uses a LangGraph StateGraph with human-in-the-loop confirmation checkpoints:

  analyst_node -> [CONFIRM] -> developer_node -> [CONFIRM]
       -> executor_node -> [CONFIRM] -> tester_node
              ^                               |
              └──── retry on test failure ────┘
                         [CONFIRM before retry]

Usage:
  python agents/orchestrator.py
  python agents/orchestrator.py --framework pyspark
  python agents/orchestrator.py --skip-run
  python agents/orchestrator.py --skip-analyst --validate
  python agents/orchestrator.py --skip-analyst --validate --max-retries 5
  python agents/orchestrator.py --no-approval   # skip human checkpoints
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from typing_extensions import TypedDict

from langgraph.graph import END, StateGraph

from agents.analyst_agent import analyze_requirements_and_data
from agents.developer_agent import generate_pipeline_code
from agents.llm_client import llm_available
from agents.tester_agent import run_tests


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------

class ETLState(TypedDict):
    # ── inputs ────────────────────────────────────────────────────────────
    requirements_path: str
    source_csv_path: str
    mapping_output_path: str
    pipeline_output_path: str
    test_report_path: str
    framework: str
    max_retries: int
    skip_analyst: bool
    run_generated: bool
    validate: bool
    require_approval: bool
    # ── produced by analyst_node ──────────────────────────────────────────
    mapping: Optional[dict]
    requirements_text: Optional[str]
    processed_data_path: Optional[str]
    # ── loop control ──────────────────────────────────────────────────────
    attempt: int
    test_feedback: Optional[dict]
    # ── produced by developer_node / executor_node ────────────────────────
    pipeline_path: Optional[str]
    # ── produced by tester_node ───────────────────────────────────────────
    all_passed: Optional[bool]
    test_summary: Optional[dict]


# ---------------------------------------------------------------------------
# Subprocess helper (shared by executor_node)
# ---------------------------------------------------------------------------

def _run_pipeline_subprocess(pipeline_path: str) -> None:
    """Run a generated pipeline script as a subprocess; raise RuntimeError on failure."""
    env = os.environ.copy()
    python_executable = sys.executable
    virtual_env = env.get("VIRTUAL_ENV")

    if virtual_env:
        candidate = Path(virtual_env) / ("Scripts" if os.name == "nt" else "bin") / (
            "python.exe" if os.name == "nt" else "python"
        )
        if candidate.exists():
            python_executable = str(candidate)

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
            f"Generated pipeline failed (exit {result.returncode}): "
            f"{result.stderr or result.stdout or 'unknown error'}"
        )


# ---------------------------------------------------------------------------
# Human-in-the-loop confirmation helper
# ---------------------------------------------------------------------------

def _human_confirm(stage: str, lines: list, require_approval: bool) -> None:
    """Print a checkpoint summary and wait for human approval.

    Raises SystemExit(0) if the user declines to continue.
    Does nothing when require_approval is False.
    """
    if not require_approval:
        return
    print(f"\n{'='*60}")
    print(f"[CHECKPOINT] {stage}")
    print("="*60)
    for line in lines:
        print(f"  {line}")
    print()
    try:
        answer = input("  Continue? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = "n"
    if answer in ("n", "no"):
        print(f"\n[STOPPED] Run aborted by user at: {stage}")
        raise SystemExit(0)


# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def analyst_node(state: ETLState) -> dict:
    print("[ANALYST] Starting analysis...")
    requirements_text = Path(state["requirements_path"]).read_text(encoding="utf-8")

    if state["skip_analyst"]:
        mapping_path = Path(state["mapping_output_path"])
        if not mapping_path.exists():
            raise FileNotFoundError(
                f"--skip-analyst requires an existing mapping file. Missing: {mapping_path}\n"
                "Run without --skip-analyst first to generate it."
            )
        mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        print("[ANALYST] Skipping -- loaded cached mapping from disk")
    else:
        analysis = analyze_requirements_and_data(
            requirements_path=state["requirements_path"],
            output_mapping_path=state["mapping_output_path"],
            use_llm=True,
        )
        mapping = analysis["mapping"]
        print(
            f"[ANALYST] target_table={mapping.get('target_table')}, "
            f"{len(mapping.get('source_to_target_mapping', []))} transformations"
        )

    target_table = mapping.get("target_table", "output_table")
    rename_rules = mapping.get("rename_rules", {})
    rename_summary = ", ".join(f"{s} -> {t}" for s, t in rename_rules.items()) or "none"

    _human_confirm(
        "Analyst -> Developer",
        [
            f"Target table : {target_table}",
            f"Mappings     : {len(mapping.get('source_to_target_mapping', []))} transformations",
            f"Rename rules : {rename_summary}",
            f"Output saved : {state['mapping_output_path']}",
        ],
        state["require_approval"],
    )

    return {
        "mapping": mapping,
        "requirements_text": requirements_text,
        "processed_data_path": f"data/processed/{target_table}.csv",
    }


def developer_node(state: ETLState) -> dict:
    attempt = state.get("attempt", 0) + 1
    print(f"[DEVELOPER] Generating pipeline (attempt {attempt}/{state['max_retries']})...")

    pipeline_path = generate_pipeline_code(
        mapping=state["mapping"],
        output_path=state["pipeline_output_path"],
        framework=state["framework"],
        data_dictionary=None,
        validate=False,
        test_feedback=state.get("test_feedback"),
        requirements_text=state.get("requirements_text"),
    )
    code = Path(pipeline_path).read_text(encoding="utf-8")
    line_count = code.count("\n")
    print(f"[DEVELOPER] Generated {line_count} lines -> {pipeline_path}")

    # Show a brief preview: find the def run(): block and grab the first 8 non-empty lines
    run_lines = [ln for ln in code.splitlines() if ln.strip()]
    preview = run_lines[:8] if len(run_lines) >= 8 else run_lines

    _human_confirm(
        "Developer -> Executor",
        [
            f"Pipeline file : {pipeline_path}",
            f"Lines         : {line_count}",
            "Preview:",
        ] + [f"  {ln}" for ln in preview],
        state["require_approval"],
    )

    return {"pipeline_path": pipeline_path, "attempt": attempt}


def executor_node(state: ETLState) -> dict:
    print(f"[EXECUTOR] Running {state['pipeline_path']}...")
    try:
        _run_pipeline_subprocess(state["pipeline_path"])
        print("[EXECUTOR] Pipeline execution succeeded")

        output_path = state.get("processed_data_path", "data/processed/output.csv")
        _human_confirm(
            "Executor -> Tester",
            [
                "Pipeline executed successfully.",
                f"Output file  : {output_path}",
                "Next step    : run data-quality test suite",
            ],
            state["require_approval"],
        )

        # Clear any failure state carried over from a previous attempt
        return {"all_passed": None, "test_feedback": None}
    except RuntimeError as exc:
        error_msg = str(exc)[:400]
        print(f"[EXECUTOR] Pipeline execution failed: {error_msg[:120]}")
        return {
            "all_passed": False,
            "test_feedback": {
                "test_results": [{"name": "execution_error", "passed": False, "details": error_msg}],
                "generated_code": Path(state["pipeline_path"]).read_text(encoding="utf-8"),
            },
        }


def tester_node(state: ETLState) -> dict:
    print("[TESTER] Running test suite...")
    summary = run_tests(
        requirements_path=state["requirements_path"],
        source_data_path=state["source_csv_path"],
        processed_data_path=state["processed_data_path"],
        report_path=state["test_report_path"],
    )

    all_passed = summary.get("all_passed", False)
    passed = sum(1 for t in summary.get("test_results", []) if t.get("passed"))
    total = len(summary.get("test_results", []))
    print(f"[TESTER] {'PASS' if all_passed else 'FAIL'}: {passed}/{total} tests")

    feedback = None
    if not all_passed:
        failed = [t for t in summary.get("test_results", []) if not t.get("passed")]
        feedback = {
            "test_results": failed,
            "generated_code": Path(state["pipeline_path"]).read_text(encoding="utf-8"),
        }

        can_retry = state["validate"] and state["attempt"] < state["max_retries"]
        if can_retry:
            failure_lines = [f"FAIL | {t['name']} | {t['details']}" for t in failed]
            _human_confirm(
                f"Tester -> Developer (retry {state['attempt'] + 1}/{state['max_retries']})",
                [f"Tests passed : {passed}/{total}"] + failure_lines + [
                    "",
                    "The developer will regenerate the pipeline with this feedback.",
                ],
                state["require_approval"],
            )

    return {"all_passed": all_passed, "test_summary": summary, "test_feedback": feedback}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def _route_after_developer(state: ETLState) -> str:
    if not state["run_generated"]:
        return END
    return "executor_node"


def _route_after_executor(state: ETLState) -> str:
    if state.get("all_passed") is False:
        if state["validate"] and state["attempt"] < state["max_retries"]:
            return "developer_node"
        return END
    return "tester_node"


def _route_after_tester(state: ETLState) -> str:
    if state.get("all_passed"):
        print(f"[ORCHESTRATOR] SUCCESS after {state['attempt']} attempt(s)")
        return END
    if state["validate"] and state["attempt"] < state["max_retries"]:
        print(f"[ORCHESTRATOR] Tests failed — retrying ({state['attempt'] + 1}/{state['max_retries']})")
        return "developer_node"
    print(f"[ORCHESTRATOR] Tests failed — no retries remaining")
    return END


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_graph() -> "CompiledGraph":
    graph = StateGraph(ETLState)

    graph.add_node("analyst_node", analyst_node)
    graph.add_node("developer_node", developer_node)
    graph.add_node("executor_node", executor_node)
    graph.add_node("tester_node", tester_node)

    graph.set_entry_point("analyst_node")
    graph.add_edge("analyst_node", "developer_node")

    graph.add_conditional_edges(
        "developer_node",
        _route_after_developer,
        {"executor_node": "executor_node", END: END},
    )
    graph.add_conditional_edges(
        "executor_node",
        _route_after_executor,
        {"tester_node": "tester_node", "developer_node": "developer_node", END: END},
    )
    graph.add_conditional_edges(
        "tester_node",
        _route_after_tester,
        {"developer_node": "developer_node", END: END},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Public entry point (same signature as before)
# ---------------------------------------------------------------------------

def orchestrate(
    requirements_path: str,
    source_csv_path: str,
    mapping_output_path: str,
    pipeline_output_path: str,
    test_report_path: str,
    framework: str,
    run_generated: bool,
    validate: bool,
    max_retries: int,
    skip_analyst: bool = False,
    require_approval: bool = True,
) -> dict:
    if not llm_available():
        raise RuntimeError(
            "LLM configuration is missing. Set OPENAI_API_KEY (and optionally "
            "OPENAI_BASE_URL / OPENAI_MODEL) before running the orchestrator."
        )

    print(
        f"[ORCHESTRATOR] Starting ETL pipeline orchestration "
        f"(max_retries={max_retries}, validate={validate}, "
        f"skip_analyst={skip_analyst}, require_approval={require_approval})\n"
    )

    initial_state: ETLState = {
        "requirements_path": requirements_path,
        "source_csv_path": source_csv_path,
        "mapping_output_path": mapping_output_path,
        "pipeline_output_path": pipeline_output_path,
        "test_report_path": test_report_path,
        "framework": framework,
        "max_retries": max_retries,
        "skip_analyst": skip_analyst,
        "run_generated": run_generated,
        "validate": validate,
        "require_approval": require_approval,
        "mapping": None,
        "requirements_text": None,
        "processed_data_path": None,
        "attempt": 0,
        "test_feedback": None,
        "pipeline_path": None,
        "all_passed": None,
        "test_summary": None,
    }

    app = build_graph()
    final_state = app.invoke(initial_state)

    mapping = final_state.get("mapping") or {}
    test_summary = final_state.get("test_summary")

    return {
        "mapping_output": mapping_output_path,
        "pipeline_output": final_state.get("pipeline_path"),
        "test_report": test_report_path if test_summary is not None else None,
        "target_table": mapping.get("target_table"),
        "tests_passed": final_state.get("all_passed"),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the GenAI ETL flow end-to-end via LangGraph orchestration."
    )
    parser.add_argument(
        "--requirements",
        default="data/raw/requirements_document.txt",
    )
    parser.add_argument(
        "--input",
        default="data/raw/sample_transactions.csv",
    )
    parser.add_argument(
        "--mapping-output",
        default="outputs/source_to_target_mapping.json",
    )
    parser.add_argument(
        "--pipeline-output",
        default="outputs/generated_pipeline.py",
    )
    parser.add_argument(
        "--test-report",
        default="data/processed/test_report.txt",
    )
    parser.add_argument(
        "--framework",
        choices=["pandas", "pyspark"],
        default="pandas",
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
    )
    parser.add_argument(
        "--skip-analyst",
        action="store_true",
        help="Load cached mapping from disk; skip re-running the analyst",
    )
    parser.add_argument(
        "--no-approval",
        action="store_true",
        help="Skip human confirmation checkpoints (useful for automated runs)",
    )
    args = parser.parse_args()

    summary = orchestrate(
        requirements_path=args.requirements,
        source_csv_path=args.input,
        mapping_output_path=args.mapping_output,
        pipeline_output_path=args.pipeline_output,
        test_report_path=args.test_report,
        framework=args.framework,
        run_generated=not args.skip_run,
        validate=args.validate,
        max_retries=args.max_retries,
        skip_analyst=args.skip_analyst,
        require_approval=not args.no_approval,
    )

    print("\nOrchestration complete")
    print(json.dumps(summary, indent=2))
