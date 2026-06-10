from pathlib import Path
import json
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.orchestrator import orchestrate


def test_orchestrator_iterates_until_all_tests_pass(tmp_path, monkeypatch):
    requirements_path = tmp_path / "requirements_document.txt"
    source_csv_path = tmp_path / "sample_transactions.csv"
    mapping_output_path = tmp_path / "outputs" / "source_to_target_mapping.json"
    pipeline_output_path = tmp_path / "outputs" / "generated_pipeline.py"
    test_report_path = tmp_path / "data" / "processed" / "test_report.txt"
    activity_log_path = tmp_path / "outputs" / "agent_activity_log.json"

    requirements_path.write_text("dummy requirements", encoding="utf-8")
    source_csv_path.write_text("col1,col2\n1,2\n", encoding="utf-8")

    monkeypatch.setattr("agents.llm_client.llm_available", lambda: True)
    monkeypatch.setattr(
        "agents.analyst_agent.save_source_to_target_mapping",
        lambda requirements_path, output_path, use_llm: {
            "target_table": "fraud_transactions",
            "source_file": str(source_csv_path),
            "source_columns": [{"name": "col1", "type": "int"}],
            "target_columns": [{"name": "col1", "type": "int"}],
        },
    )

    def fake_generate_pipeline_code(
        mapping,
        output_path,
        framework,
        data_dictionary,
        feedback=None,
        iteration=None,
    ):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"# iteration={iteration}\n", encoding="utf-8")
        return str(path)

    monkeypatch.setattr("agents.developer_agent.generate_pipeline_code", fake_generate_pipeline_code)

    def fake_run_generated_pipeline(pipeline_path: str):
        out_path = tmp_path / "data" / "processed" / "fraud_transactions.csv"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("col1,col2\n1,2\n", encoding="utf-8")

    monkeypatch.setattr("agents.orchestrator._run_generated_pipeline", fake_run_generated_pipeline)

    call_count = {"runs": 0}

    def fake_run_tests(
        requirements_path,
        source_data_path,
        processed_data_path,
        report_path,
        generated_tests_output_path="outputs/generated_tests.py",
        summary_output_path="outputs/test_summary.json",
        iteration=None,
    ):
        call_count["runs"] += 1
        report_file = Path(report_path)
        report_file.parent.mkdir(parents=True, exist_ok=True)
        report_file.write_text(f"iteration={call_count['runs']}\n", encoding="utf-8")
        summary = {
            "requirements_path": str(requirements_path),
            "source_data_path": str(source_data_path),
            "processed_data_path": str(processed_data_path),
            "generated_tests_output_path": generated_tests_output_path,
            "source_row_count": 1,
            "row_count": 1,
            "all_passed": call_count["runs"] >= 2,
            "test_results": [
                {
                    "name": "dummy_test",
                    "passed": call_count["runs"] >= 2,
                    "details": "ok" if call_count["runs"] >= 2 else "needs fix",
                }
            ],
        }
        summary_file = Path(summary_output_path)
        summary_file.parent.mkdir(parents=True, exist_ok=True)
        summary_file.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    monkeypatch.setattr("agents.testing_agent.run_tests", fake_run_tests)

    summary = orchestrate(
        requirements_path=str(requirements_path),
        source_csv_path=str(source_csv_path),
        mapping_output_path=str(mapping_output_path),
        pipeline_output_path=str(pipeline_output_path),
        test_report_path=str(test_report_path),
        framework="pandas",
        run_generated=True,
        iterate=True,
        max_iterations=3,
        activity_log_path=str(activity_log_path),
    )

    assert summary["tests_passed"] is True
    assert summary["iterations"] == 2
    assert (tmp_path / "outputs" / "generated_pipeline_iteration_1.py").exists()
    assert (tmp_path / "outputs" / "generated_pipeline_iteration_2.py").exists()
    assert (tmp_path / "data" / "processed" / "test_report_iteration_2.txt").exists()
    assert activity_log_path.exists()
    log_entries = json.loads(activity_log_path.read_text(encoding="utf-8"))
    assert any(entry["agent"] == "orchestrator" and entry["stage"] == "complete" for entry in log_entries)
