"""LLM factory.

The lab defaults to OpenRouter, but this helper also accepts a normal OpenAI
project key so the app can run with whichever key is available in `.env`.
"""

import os

from langchain_openai import ChatOpenAI


def get_llm(temperature: float = 0.2) -> ChatOpenAI:
    openrouter_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini").strip()

    # OpenAI project keys start with sk-proj-. If one was pasted into
    # OPENROUTER_API_KEY, use the native OpenAI endpoint and normalize the model.
    if openai_key or openrouter_key.startswith("sk-proj-"):
        api_key = openai_key or openrouter_key
        if model.startswith("openai/"):
            model = model.removeprefix("openai/")
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            temperature=temperature,
        )

    if not openrouter_key:
        raise RuntimeError(
            "Missing LLM key. Set OPENROUTER_API_KEY=sk-or-v1-... or "
            "OPENAI_API_KEY=sk-proj-... in .env"
        )

    return ChatOpenAI(
        model=model,
        base_url=base_url,
        api_key=openrouter_key,
        temperature=temperature,
    )
