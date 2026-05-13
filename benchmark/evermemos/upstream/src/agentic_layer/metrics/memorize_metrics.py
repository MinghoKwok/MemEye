"""Benchmark-scoped no-op memorize metrics."""


def get_space_id_for_metrics() -> str:
    return "benchmark"


def record_memorize_request(*args, **kwargs) -> None:
    return None


def record_memorize_error(*args, **kwargs) -> None:
    return None


def record_memorize_message(*args, **kwargs) -> None:
    return None


def record_boundary_detection(*args, **kwargs) -> None:
    return None


def record_memcell_extracted(*args, **kwargs) -> None:
    return None


def record_extract_memory_call(*args, **kwargs) -> None:
    return None
