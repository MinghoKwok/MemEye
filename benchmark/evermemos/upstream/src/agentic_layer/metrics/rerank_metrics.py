"""Benchmark-scoped no-op rerank metrics."""


def record_rerank_request(*args, **kwargs) -> None:
    return None


def record_rerank_fallback(*args, **kwargs) -> None:
    return None


def record_rerank_error(*args, **kwargs) -> None:
    return None

