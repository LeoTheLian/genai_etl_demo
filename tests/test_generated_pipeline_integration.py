from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_generated_pipeline_module():
    pipeline_path = ROOT / "outputs" / "generated_pipeline.py"
    spec = importlib.util.spec_from_file_location("generated_pipeline_under_test", pipeline_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _parse_report(report_path: Path) -> dict[str, str]:
    parsed = {}
    for line in report_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        parsed[key.strip()] = value.strip()
    return parsed


def test_generated_pipeline_runs_and_writes_expected_outputs(tmp_path: Path):
    module = _load_generated_pipeline_module()
    output_path = tmp_path / "fraud_transactions.csv"
    report_path = tmp_path / "validation_report.txt"

    module.INPUT_PATH = str(ROOT / "data" / "raw" / "sample_transactions.csv")
    module.OUTPUT_PATH = str(output_path)
    module.REPORT_PATH = str(report_path)

    module.run()

    assert output_path.exists()
    assert report_path.exists()


def test_validation_report_counts_match_expected_regression_values(tmp_path: Path):
    module = _load_generated_pipeline_module()
    output_path = tmp_path / "fraud_transactions.csv"
    report_path = tmp_path / "validation_report.txt"

    module.INPUT_PATH = str(ROOT / "data" / "raw" / "sample_transactions.csv")
    module.OUTPUT_PATH = str(output_path)
    module.REPORT_PATH = str(report_path)

    module.run()

    report = _parse_report(report_path)

    assert report["total_records_input"] == "1000"
    assert report["total_records_output"] == "937"
    assert report["total_records_removed"] == "63"
    assert report["transaction_amount"] == "20"
    assert report["account_balance"] == "15"
    assert report["cardholder_age"] == "10"
    assert report["distance_from_home_km"] == "8"
    assert report["pass"] == "True"