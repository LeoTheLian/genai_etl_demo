import os

from langchain_openai import ChatOpenAI


def llm_available() -> bool:
    return bool(os.getenv("OPENAI_API_KEY"))


def make_llm() -> ChatOpenAI:
    """Return a ChatOpenAI instance configured from environment variables.

    Environment variables:
    - OPENAI_API_KEY  (required)
    - OPENAI_BASE_URL (optional; defaults to OpenAI)
    - OPENAI_MODEL    (optional; defaults to gpt-4o-mini)
    """
    return ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        base_url=os.getenv("OPENAI_BASE_URL") or None,
        temperature=0,
    )
