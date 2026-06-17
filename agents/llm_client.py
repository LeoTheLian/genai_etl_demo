import os

from langchain_openai import ChatOpenAI

_REASONING_MODEL_PREFIXES = ("o1", "o3", "o4")


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def make_llm(model: str | None = None, seed: int | None = 42) -> ChatOpenAI:
    """Return a ChatOpenAI instance configured from environment variables.

    Environment variables:
    - OPENAI_API_KEY  (required)
    - OPENAI_BASE_URL (optional; defaults to OpenAI)
    - OPENAI_MODEL    (optional; defaults to gpt-4o-mini)

    Args:
        model: Override the model name. Falls back to OPENAI_MODEL env var.
        seed: RNG seed for best-effort deterministic output. Ignored for
              reasoning models (o1/o3/o4) which manage sampling internally.
    """
    resolved = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    is_reasoning = any(resolved.startswith(p) for p in _REASONING_MODEL_PREFIXES)
    kwargs: dict = {
        "model": resolved,
        "base_url": os.getenv("OPENAI_BASE_URL") or None,
    }
    if not is_reasoning:
        kwargs["temperature"] = 0
        if seed is not None:
            kwargs["seed"] = seed
    return ChatOpenAI(**kwargs)
