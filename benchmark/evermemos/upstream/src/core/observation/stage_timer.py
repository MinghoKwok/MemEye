from __future__ import annotations

import functools
from contextlib import contextmanager
from typing import Optional


class StageTimer:
    """Minimal benchmark-scoped timer placeholder."""

    def __init__(self, endpoint: str) -> None:
        self.endpoint = endpoint

    def summary(self) -> dict:
        return {"endpoint": self.endpoint}

    def log_summary(self) -> None:
        return None


def start_timer(endpoint: str) -> None:
    return None


def log_timer() -> None:
    return None


def stage_timed(endpoint: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)

        return wrapper

    return decorator


def get_current_timer() -> Optional[StageTimer]:
    return None


@contextmanager
def timed(name: str):
    yield


@contextmanager
def timed_parallel(name: str):
    yield

