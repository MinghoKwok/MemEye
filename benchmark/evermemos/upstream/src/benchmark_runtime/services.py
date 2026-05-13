"""Explicit singleton services for the benchmark-scoped EverMemOS runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import tiktoken


class TokenUsageCollector(ABC):
    @abstractmethod
    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        call_type: str = "llm",
        request_id: Optional[str] = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_totals(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class NoopTokenUsageCollector(TokenUsageCollector):
    def add(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        *,
        call_type: str = "llm",
        request_id: Optional[str] = None,
    ) -> None:
        return None

    def get_totals(self) -> dict:
        return {
            "input_tokens": 0,
            "output_tokens": 0,
            "embedding_calls": 0,
            "call_count": 0,
        }

    def reset(self) -> None:
        return None


class TokenizerFactory:
    def __init__(self) -> None:
        self._tokenizers: Dict[str, Any] = {}

    def get_tokenizer_from_tiktoken(self, encoding_name: str) -> tiktoken.Encoding:
        cache_key = f"tiktoken:{encoding_name}"
        if cache_key not in self._tokenizers:
            self._tokenizers[cache_key] = tiktoken.get_encoding(encoding_name)
        return self._tokenizers[cache_key]

    def load_default_encodings(self) -> None:
        for encoding_name in ("o200k_base", "cl100k_base"):
            self.get_tokenizer_from_tiktoken(encoding_name)

    def get_cached_tokenizer_count(self) -> int:
        return len(self._tokenizers)

    def clear_cache(self) -> None:
        self._tokenizers.clear()


_TOKENIZER_FACTORY = TokenizerFactory()
_TOKEN_USAGE_COLLECTOR = NoopTokenUsageCollector()


def get_tokenizer_factory() -> TokenizerFactory:
    return _TOKENIZER_FACTORY


def get_token_usage_collector() -> TokenUsageCollector:
    return _TOKEN_USAGE_COLLECTOR

