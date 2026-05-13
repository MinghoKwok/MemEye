"""Benchmark-scoped no-op vectorize metrics."""


def record_vectorize_request(*args, **kwargs) -> None:
    return None


def record_vectorize_fallback(*args, **kwargs) -> None:
    return None


def record_vectorize_error(*args, **kwargs) -> None:
    return None

