"""Benchmark-scoped no-op LLM metrics."""


def record_llm_request(*args, **kwargs) -> None:
    return None

