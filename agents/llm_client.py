import json
import os
import urllib.error
import urllib.request


def llm_available():
    return bool(os.getenv("OPENAI_API_KEY"))


def _post_json(url, payload, headers, timeout):
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8")
    return json.loads(raw)


def _extract_json_block(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    return text


def call_llm_json(system_prompt, user_prompt, timeout=60):
    """
    Call an OpenAI-compatible chat endpoint and return parsed JSON.

    Expected environment variables:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL (optional, default: gpt-4o-mini)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = _post_json(
            url=f"{base_url}/chat/completions",
            payload=payload,
            headers=headers,
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP error: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM URL error: {exc}") from exc

    content = response["choices"][0]["message"]["content"]
    content = _extract_json_block(content)

    try:
        return json.loads(content)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM returned non-JSON content: {content[:300]}") from exc


def _extract_code_block(text):
    """Strip markdown fenced code block wrappers if present."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # Drop opening fence (```python or ```)
        start = 1
        # Drop closing fence
        end = len(lines) - 1 if lines[-1].strip() == "```" else len(lines)
        return "\n".join(lines[start:end]).strip()
    return text


def call_llm_text(system_prompt, user_prompt, timeout=120):
    """
    Call an OpenAI-compatible chat endpoint and return raw text content.
    Use this for code generation (as opposed to call_llm_json for structured output).

    Expected environment variables:
    - OPENAI_API_KEY
    - OPENAI_BASE_URL (optional, default: https://api.openai.com/v1)
    - OPENAI_MODEL (optional, default: gpt-4o-mini)
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    try:
        response = _post_json(
            url=f"{base_url}/chat/completions",
            payload=payload,
            headers=headers,
            timeout=timeout,
        )
    except urllib.error.HTTPError as exc:
        details = exc.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"LLM HTTP error: {exc.code} {details}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"LLM URL error: {exc}") from exc

    content = response["choices"][0]["message"]["content"]
    return _extract_code_block(content)