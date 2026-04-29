from pathlib import Path
import os
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.llm_client import call_llm_json, llm_available


@pytest.mark.skipif(not llm_available(), reason="OPENAI_API_KEY is not set")
def test_llm_connection_returns_expected_json_shape():
    response = call_llm_json(
        system_prompt="You are a connectivity test assistant. Return strict JSON only.",
        user_prompt=(
            "Return a JSON object with exactly these keys: "
            "status, provider, model_check. "
            "Set status to ok, provider to openai_compatible, and model_check to passed."
        ),
        timeout=30,
    )

    assert isinstance(response, dict)
    assert response["status"] == "ok"
    assert response["provider"] == "openai_compatible"
    assert response["model_check"] == "passed"


def test_llm_environment_configuration_is_visible():
    assert os.getenv("OPENAI_MODEL", "gpt-4o-mini")