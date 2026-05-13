"""Prompt registry for the benchmark-scoped English EverMemOS runtime."""

import os
from typing import Any, Optional

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES = [DEFAULT_LANGUAGE]

_PROMPT_REGISTRY = {
    "CONV_BOUNDARY_DETECTION_PROMPT": ("memory_layer.prompts.en.conv_prompts", False),
    "CONV_BATCH_BOUNDARY_DETECTION_PROMPT": ("memory_layer.prompts.en.conv_prompts", False),
    "CONV_SUMMARY_PROMPT": ("memory_layer.prompts.en.conv_prompts", False),
    "EPISODE_GENERATION_PROMPT": ("memory_layer.prompts.en.episode_mem_prompts", False),
    "GROUP_EPISODE_GENERATION_PROMPT": ("memory_layer.prompts.en.episode_mem_prompts", False),
    "DEFAULT_CUSTOM_INSTRUCTIONS": ("memory_layer.prompts.en.episode_mem_prompts", False),
    "PROFILE_UPDATE_PROMPT": ("memory_layer.prompts.en.profile_prompts", False),
    "TEAM_PROFILE_UPDATE_PROMPT": ("memory_layer.prompts.en.profile_prompts", False),
    "PROFILE_INITIAL_EXTRACTION_PROMPT": ("memory_layer.prompts.en.profile_prompts", False),
    "PROFILE_COMPACT_PROMPT": ("memory_layer.prompts.en.profile_prompts", False),
    "FORESIGHT_GENERATION_PROMPT": ("memory_layer.prompts.en.foresight_prompts", False),
    "ATOMIC_FACT_PROMPT": ("memory_layer.prompts.en.atomic_fact_prompts", False),
}


# ============================================================================
# PromptManager - Dynamic prompt loader with caching
# ============================================================================


class PromptManager:
    """Prompt manager for dynamic English prompt loading."""

    def __init__(self):
        self._module_cache: dict[str, Any] = {}

    def _load_module(self, module_path: str) -> Any:
        """Load module dynamically with caching."""
        if module_path not in self._module_cache:
            import importlib

            self._module_cache[module_path] = importlib.import_module(module_path)
        return self._module_cache[module_path]

    def get_prompt(self, prompt_name: str, language: Optional[str] = None) -> Any:
        """Get a prompt by name for the supported benchmark language."""
        if language is None:
            language = _get_prompt_language()
        language = language.lower()
        if language not in SUPPORTED_LANGUAGES:
            raise ValueError(
                f"Language '{language}' is not supported in the local benchmark runtime. "
                f"Supported languages: {SUPPORTED_LANGUAGES}"
            )

        if prompt_name not in _PROMPT_REGISTRY:
            raise ValueError(
                f"Unknown prompt: {prompt_name}. Available: {list(_PROMPT_REGISTRY.keys())}"
            )

        module_path, _ = _PROMPT_REGISTRY[prompt_name]
        module = self._load_module(module_path)
        return getattr(module, prompt_name)

    def list_prompts(self) -> list[str]:
        """List all available prompt names."""
        return list(_PROMPT_REGISTRY.keys())

    def get_supported_languages(self, prompt_name: str) -> list[str]:
        """Get supported languages for a prompt."""
        if prompt_name not in _PROMPT_REGISTRY:
            return []
        return list(SUPPORTED_LANGUAGES)


# Global PromptManager instance
_prompt_manager = PromptManager()


def get_prompt_by(prompt_name: str, language: Optional[str] = None) -> Any:
    """Get a prompt by name and language (English-only in this benchmark)."""
    return _prompt_manager.get_prompt(prompt_name, language)


# ============================================================================
# Exported constants (for backward compatibility)
# ============================================================================


def _get_prompt_language() -> str:
    """Return the supported prompt language for the benchmark runtime."""
    language = os.getenv("MEMORY_LANGUAGE", DEFAULT_LANGUAGE).lower()
    if language not in SUPPORTED_LANGUAGES:
        return DEFAULT_LANGUAGE
    return language


CURRENT_LANGUAGE = _get_prompt_language()
MEMORY_LANGUAGE = CURRENT_LANGUAGE


def get_current_language() -> str:
    """Get current language setting."""
    return _get_prompt_language()
