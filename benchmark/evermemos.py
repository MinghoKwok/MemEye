import asyncio
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

from .common import SCRIPT_DIR, write_json
from .dataset import MemoryBenchmarkDataset
from .methods import HistoryMethod


_EVERMEMOS_ROOT = (
    Path(__file__).resolve().parent
    / "evermemos"
    / "upstream"
).resolve()
_EVERMEMOS_SRC = (_EVERMEMOS_ROOT / "src").resolve()

for _path in (_EVERMEMOS_ROOT, _EVERMEMOS_SRC):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

_ENV_LOCAL = (SCRIPT_DIR / ".env.local").resolve()
if _ENV_LOCAL.exists():
    load_dotenv(_ENV_LOCAL, override=False)


def _seed_evermemos_service_env() -> None:
    vectorize_fallback_key = os.getenv("VECTORIZE_FALLBACK_API_KEY", "").strip()
    if vectorize_fallback_key and not os.getenv("VECTORIZE_PROVIDER", "").strip():
        os.environ["VECTORIZE_PROVIDER"] = "deepinfra"
    if vectorize_fallback_key and not os.getenv("VECTORIZE_API_KEY", "").strip():
        os.environ["VECTORIZE_API_KEY"] = vectorize_fallback_key
    if os.getenv("VECTORIZE_PROVIDER", "").strip().lower() == "deepinfra" and not os.getenv(
        "VECTORIZE_BASE_URL", ""
    ).strip():
        os.environ["VECTORIZE_BASE_URL"] = "https://api.deepinfra.com/v1/openai"

    rerank_key = (
        os.getenv("RERANK_API_KEY", "").strip()
        or os.getenv("RERANK_FALLBACK_API_KEY", "").strip()
        or vectorize_fallback_key
    )
    if rerank_key and not os.getenv("RERANK_PROVIDER", "").strip():
        os.environ["RERANK_PROVIDER"] = "deepinfra"
    if rerank_key and not os.getenv("RERANK_API_KEY", "").strip():
        os.environ["RERANK_API_KEY"] = rerank_key
    if os.getenv("RERANK_PROVIDER", "").strip().lower() == "deepinfra" and not os.getenv(
        "RERANK_BASE_URL", ""
    ).strip():
        os.environ["RERANK_BASE_URL"] = "https://api.deepinfra.com/v1/inference"


_seed_evermemos_service_env()

def _run_async(coro: Any) -> Any:
    try:
        return asyncio.run(coro)
    except RuntimeError as exc:
        if "asyncio.run() cannot be called from a running event loop" not in str(exc):
            raise
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _resolve_secret(
    primary_value: str,
    env_name: str,
) -> Optional[str]:
    raw = str(primary_value).strip()
    if raw:
        return raw
    env_key = str(env_name).strip()
    if env_key:
        value = os.getenv(env_key, "").strip()
        if value:
            return value
    return None


def _load_evermemos_components() -> Dict[str, Any]:
    try:
        from evaluation.src.adapters.evermemos_adapter import EverMemOSAdapter  # type: ignore
        from evaluation.src.adapters.evermemos import stage4_response  # type: ignore
        from evaluation.src.core.data_models import Conversation, Message  # type: ignore
        from memory_layer.llm.llm_provider import LLMProvider  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "EverMemOS dependencies are not fully installed in the active environment. "
            f"Missing module: {exc.name}. Install the dependencies required by the vendored "
            f"EverMemOS runtime in the active 'memorybench' conda env."
        ) from exc
    return {
        "EverMemOSAdapter": EverMemOSAdapter,
        "stage4_response": stage4_response,
        "Conversation": Conversation,
        "Message": Message,
        "LLMProvider": LLMProvider,
    }


def _round_image_blocks(raw_dialogue: Dict[str, Any]) -> List[str]:
    input_images = raw_dialogue.get("input_image", []) or []
    captions = raw_dialogue.get("image_caption", []) or []
    if len(captions) != len(input_images):
        raise ValueError(
            "EverMemOS text-only adaptation requires Mem-Gallery-style image captions "
            "for every image-bearing round."
        )
    lines: List[str] = []
    for caption in captions:
        caption_text = str(caption).strip()
        if not caption_text:
            raise ValueError(
                "EverMemOS text-only adaptation requires non-empty image_caption values."
            )
        lines.extend(["image:", f"image_caption: {caption_text}"])
    return lines


def _message_text(text: str, *, image_blocks: Optional[List[str]] = None) -> str:
    body = str(text).strip()
    lines: List[str] = [body] if body else []
    if image_blocks:
        lines.extend(image_blocks)
    return "\n".join(lines).strip()


def _round_messages(
    round_payload: Dict[str, Any],
    speaker_a: str,
    speaker_b: str,
) -> List[Dict[str, str]]:
    raw_dialogue = round_payload.get("raw", {}) or {}
    image_blocks = _round_image_blocks(raw_dialogue)
    user_text = _message_text(str(round_payload.get("user", "")), image_blocks=image_blocks)
    assistant_text = _message_text(str(round_payload.get("assistant", "")))

    messages: List[Dict[str, str]] = []
    if user_text:
        messages.append(
            {
                "sender_id": "benchmark_user",
                "sender_name": speaker_a,
                "role": "user",
                "content": user_text,
            }
        )
    if assistant_text:
        messages.append(
            {
                "sender_id": "benchmark_assistant",
                "sender_name": speaker_b,
                "role": "assistant",
                "content": assistant_text,
            }
        )
    return messages


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    query = question.strip()
    question_caption = qa.get("image_caption")
    question_image = qa.get("question_image")
    question_images = qa.get("question_images")
    has_question_image = bool(question_image) or bool(question_images)
    if not has_question_image or not question_caption:
        return query

    if isinstance(question_caption, list):
        caption_text = " ".join(
            str(item).strip() for item in question_caption if str(item).strip()
        )
    else:
        caption_text = str(question_caption).strip()
    if not caption_text:
        return query
    return f"{query}\nimage:\nimage_caption: {caption_text}"


def _raw_question_text(qa: Dict[str, Any], fallback_question: str) -> str:
    raw_question = str(qa.get("question", "") or "").strip()
    return raw_question or str(fallback_question).strip()


def _safe_task_name(dataset: MemoryBenchmarkDataset) -> str:
    task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
    safe = re.sub(r"[^a-z0-9]+", "_", task_name.lower())
    return safe.strip("_") or dataset.dialog_json_path.stem.lower()


def _parse_session_base(session_date: str, session_index: int) -> datetime:
    session_date = str(session_date).strip()
    base_dt = datetime(2025, 1, 1, 0, 0, 0)
    if not session_date:
        return base_dt + timedelta(days=session_index)
    try:
        return datetime.strptime(session_date, "%Y-%m-%d")
    except ValueError:
        return base_dt + timedelta(days=session_index)


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            key_text = str(key).lower()
            if "api_key" in key_text or "token" in key_text or "secret" in key_text:
                continue
            redacted[str(key)] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return [_redact_secrets(item) for item in value]
    return value


def _provider_env_name(provider_type: str, suffix: str) -> str:
    provider = str(provider_type or "openai").strip().upper() or "OPENAI"
    return f"{provider}_{suffix}"


def _resolve_provider_secret(
    primary_value: str,
    env_name: str,
    provider_type: str,
    *,
    legacy_env_name: str,
) -> Optional[str]:
    raw = str(primary_value).strip()
    if raw:
        return raw

    explicit_env = str(env_name).strip()
    if explicit_env:
        value = os.getenv(explicit_env, "").strip()
        if value:
            return value

    provider_env = _provider_env_name(provider_type, "API_KEY")
    value = os.getenv(provider_env, "").strip()
    if value:
        return value

    legacy_value = os.getenv(legacy_env_name, "").strip()
    return legacy_value or None


def _resolve_provider_base_url(
    primary_value: str,
    env_name: str,
    provider_type: str,
    *,
    legacy_env_name: str,
    default: str,
) -> str:
    raw = str(primary_value).strip()
    if raw:
        return raw

    explicit_env = str(env_name).strip()
    if explicit_env:
        value = os.getenv(explicit_env, "").strip()
        if value:
            return value

    provider_env = _provider_env_name(provider_type, "BASE_URL")
    value = os.getenv(provider_env, "").strip()
    if value:
        return value

    legacy_value = os.getenv(legacy_env_name, "").strip()
    if legacy_value:
        return legacy_value

    return default


class EverMemOSMethod(HistoryMethod):
    name = "evermemos"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._adapter: Any = None
        self._answer_llm: Any = None
        self._answer_client: Optional[OpenAI] = None
        self._conversation: Any = None
        self._index_metadata: Optional[Dict[str, Any]] = None
        self._debug_rows: List[Dict[str, Any]] = []
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._components: Optional[Dict[str, Any]] = None
        self._answer_model_name: str = str(self.config.get("answer_model", "gpt-4.1-nano")).strip() or "gpt-4.1-nano"
        self._runtime_signature_payload_cache: Dict[int, Dict[str, Any]] = {}
        self._runtime_signature_cache: Dict[int, str] = {}

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return self._runtime_dir(dataset)

    def _runtime_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (
            SCRIPT_DIR
            / "runs"
            / _safe_task_name(dataset)
            / self.name
            / self._runtime_signature(dataset)
        ).resolve()

    def _runtime_signature_payload(self, dataset: MemoryBenchmarkDataset) -> Dict[str, Any]:
        dataset_id = id(dataset)
        cached = self._runtime_signature_payload_cache.get(dataset_id)
        if cached is not None:
            return cached
        dialog_path = dataset.dialog_json_path.resolve()
        try:
            dialog_sha256 = hashlib.sha256(dialog_path.read_bytes()).hexdigest()
        except OSError:
            dialog_sha256 = "unavailable"

        method_config = _redact_secrets(dict(self.config))
        model_cfg = _redact_secrets(dict(self.config.get("_model_cfg", {})))
        payload = {
            "task_name": str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem,
            "dialog_json_path": str(dialog_path),
            "dialog_sha256": dialog_sha256,
            "method_name": self.name,
            "method_config": method_config,
            "model_config": model_cfg,
            "code_version": 11,
        }
        self._runtime_signature_payload_cache[dataset_id] = payload
        return payload

    def _runtime_signature(self, dataset: MemoryBenchmarkDataset) -> str:
        dataset_id = id(dataset)
        cached = self._runtime_signature_cache.get(dataset_id)
        if cached is not None:
            return cached
        payload = self._runtime_signature_payload(dataset)
        serialized = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]
        self._runtime_signature_cache[dataset_id] = digest
        return digest

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "runtime_signature": self._runtime_signature(dataset),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _build_internal_llm_config(self) -> Dict[str, Any]:
        llm_provider = str(self.config.get("internal_llm_provider", "openai")).strip() or "openai"
        llm_model = str(self.config.get("internal_llm_model", "gpt-4.1-mini")).strip() or "gpt-4.1-mini"
        api_key = _resolve_provider_secret(
            str(self.config.get("internal_llm_api_key", "")),
            str(self.config.get("internal_llm_api_key_env", "OPENAI_API_KEY")),
            llm_provider,
            legacy_env_name="LLM_API_KEY",
        )
        if not api_key:
            raise ValueError(
                "Missing EverMemOS internal LLM API key. Set OPENAI_API_KEY or internal_llm_api_key."
            )
        base_url = _resolve_provider_base_url(
            str(self.config.get("internal_llm_base_url", "")),
            str(self.config.get("internal_llm_base_url_env", "OPENAI_BASE_URL")),
            llm_provider,
            legacy_env_name="LLM_BASE_URL",
            default="https://api.openai.com/v1",
        )
        return {
            "provider": llm_provider,
            "model": llm_model,
            "api_key": api_key,
            "base_url": base_url,
            "temperature": float(self.config.get("internal_llm_temperature", 0.3)),
            "max_tokens": int(self.config.get("internal_llm_max_tokens", 16384)),
        }

    def _load_components(self) -> Dict[str, Any]:
        if self._components is None:
            self._components = _load_evermemos_components()
        return self._components

    def _build_answer_llm(self) -> Any:
        llm_provider_cls = self._load_components()["LLMProvider"]
        model_config = dict(self.config.get("_model_cfg", {}))
        provider_type = str(model_config.get("provider", "openai")).strip() or "openai"
        api_key = _resolve_provider_secret(
            str(model_config.get("api_key", "")),
            str(model_config.get("api_key_env", "OPENAI_API_KEY")),
            provider_type,
            legacy_env_name="LLM_API_KEY",
        )
        if not api_key:
            raise ValueError(
                "Missing API key for EverMemOS benchmark QA model. "
                "Set OPENAI_API_KEY or provide api_key/api_key_env in the model config."
            )
        base_url = _resolve_provider_base_url(
            str(model_config.get("base_url", "")),
            str(model_config.get("base_url_env", "OPENAI_BASE_URL")),
            provider_type,
            legacy_env_name="LLM_BASE_URL",
            default="https://api.openai.com/v1",
        )
        return llm_provider_cls(
            provider_type=provider_type,
            model=self._answer_model_name,
            api_key=api_key,
            base_url=base_url,
            temperature=0.0,
            max_tokens=int(model_config.get("max_new_tokens", 128) or 128),
        )

    def _build_answer_client(self) -> OpenAI:
        model_config = dict(self.config.get("_model_cfg", {}))
        provider_type = str(model_config.get("provider", "openai")).strip() or "openai"
        api_key = _resolve_provider_secret(
            str(model_config.get("api_key", "")),
            str(model_config.get("api_key_env", "OPENAI_API_KEY")),
            provider_type,
            legacy_env_name="LLM_API_KEY",
        )
        if not api_key:
            raise ValueError(
                "Missing API key for EverMemOS benchmark QA model. "
                "Set OPENAI_API_KEY or provide api_key/api_key_env in the model config."
            )
        base_url = _resolve_provider_base_url(
            str(model_config.get("base_url", "")),
            str(model_config.get("base_url_env", "OPENAI_BASE_URL")),
            provider_type,
            legacy_env_name="LLM_BASE_URL",
            default="https://api.openai.com/v1",
        )
        timeout = int(model_config.get("timeout", 90) or 90)
        return OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    def _answer_mcq(self, query: str, context: str, options: Dict[str, Any]) -> str:
        assert self._answer_client is not None
        normalized_options: List[tuple[str, str]] = []
        for raw_key, raw_value in options.items():
            option_key = str(raw_key).strip().upper()
            if not option_key:
                continue
            normalized_options.append((option_key, str(raw_value).strip()))

        normalized_options.sort(key=lambda item: item[0])
        enum_values = [key for key, _ in normalized_options] or ["A", "B", "C", "D"]
        options_block = "\n".join(
            f"{key}. {value}" for key, value in normalized_options
        ).strip()
        user_prompt = (
            "Choose the single best option using only the retrieved context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question:\n{query}\n\n"
            f"Options:\n{options_block}\n"
        )
        response = self._answer_client.chat.completions.create(
            model=self._answer_model_name,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Answer the multiple-choice question with exactly one option letter. "
                        "Do not explain your reasoning."
                    ),
                },
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "mcq_response",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "answer": {
                                "type": "string",
                                "enum": enum_values,
                            }
                        },
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=0.0,
        )
        raw = response.choices[0].message.content or ""
        for option in enum_values:
            if f'"{option}"' in raw or raw.strip() == option:
                return option
        return raw.strip()

    def _build_conversation(self, dataset: MemoryBenchmarkDataset) -> Any:
        components = self._load_components()
        conversation_cls = components["Conversation"]
        message_cls = components["Message"]
        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name", "")).strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        messages: List[Any] = []
        message_index = 0
        for session_index, session_id in enumerate(dataset.session_order()):
            session = dataset.get_session(session_id)
            session_base = _parse_session_base(session.get("date", ""), session_index)
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id:
                    continue
                round_payload = dataset.rounds.get(round_id)
                if not round_payload:
                    continue
                round_messages = _round_messages(round_payload, self._speaker_a, self._speaker_b)
                for round_message in round_messages:
                    content = round_message["content"]
                    if not content:
                        continue
                    timestamp = session_base + timedelta(seconds=message_index * 30)
                    message_index += 1
                    role = round_message["role"]
                    messages.append(
                        message_cls(
                            sender_id=round_message["sender_id"],
                            sender_name=round_message["sender_name"],
                            content=content,
                            timestamp=timestamp,
                            metadata={
                                "session_id": session_id,
                                "round_id": round_id,
                                "role": role,
                            },
                        )
                    )
                    self._debug_rows.append(
                        {
                            "type": "stored_memory",
                            "session_id": session_id,
                            "round_id": round_id,
                            "role": role,
                            "sender_id": round_message["sender_id"],
                            "sender_name": round_message["sender_name"],
                            "timestamp": timestamp.isoformat(),
                            "text": content,
                        }
                    )

        return conversation_cls(
            conversation_id="0",
            messages=messages,
            metadata={"speaker_a": self._speaker_a, "speaker_b": self._speaker_b},
        )

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._adapter is not None and self._dataset_key == dataset_id:
            return

        self._debug_rows = []
        adapter_cls = self._load_components()["EverMemOSAdapter"]
        adapter_config = {
            "llm": self._build_internal_llm_config(),
            "search": {
                "mode": str(self.config.get("search_mode", "agentic")).strip() or "agentic",
                "lightweight_search_mode": str(
                    self.config.get("lightweight_search_mode", "bm25_only")
                ).strip()
                or "bm25_only",
                "use_hybrid_search": bool(self.config.get("use_hybrid_search", True)),
                "hybrid_emb_candidates": int(self.config.get("hybrid_emb_candidates", 50)),
                "hybrid_bm25_candidates": int(self.config.get("hybrid_bm25_candidates", 50)),
                "hybrid_rrf_k": int(self.config.get("hybrid_rrf_k", 40)),
            },
        }
        for optional_key in ("response_top_k", "use_reranker", "use_multi_query"):
            if optional_key in self.config:
                adapter_config[optional_key] = self.config[optional_key]

        runtime_dir = self._runtime_dir(dataset)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            runtime_dir / "signature.json",
            {
                "runtime_signature": self._runtime_signature(dataset),
                "signature_payload": self._runtime_signature_payload(dataset),
            },
        )
        self._adapter = adapter_cls(adapter_config, output_dir=runtime_dir)
        self._answer_llm = self._build_answer_llm()
        self._answer_client = self._build_answer_client()
        self._conversation = self._build_conversation(dataset)
        self._index_metadata = _run_async(self._adapter.add([self._conversation], output_dir=runtime_dir))
        self._dataset_key = dataset_id
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "num_memories": len(self._conversation.messages),
                "debug_dir": str(self._debug_dir(dataset)),
                "runtime_dir": str(runtime_dir),
                "runtime_signature": self._runtime_signature(dataset),
                "internal_llm_model": adapter_config["llm"]["model"],
                "answer_model": str(self.config.get("answer_model", "gpt-4.1-nano")).strip()
                or "gpt-4.1-nano",
            }
        )
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._adapter is not None
        assert self._answer_llm is not None
        assert self._conversation is not None
        assert self._index_metadata is not None
        stage4 = self._load_components()["stage4_response"]

        raw_question = _raw_question_text(qa, question)
        query = _question_with_image_caption(qa, raw_question)
        search_result = _run_async(
            self._adapter.search(
                query=query,
                conversation_id=self._conversation.conversation_id,
                index=self._index_metadata,
                conversation=self._conversation,
            )
        )
        formatted_context = str(
            search_result.retrieval_metadata.get("formatted_context", "")
        ).strip()
        if isinstance(qa.get("options"), dict):
            prediction = self._answer_mcq(query, formatted_context, qa["options"])
        else:
            prediction = _run_async(
                stage4.locomo_response(
                    llm_provider=self._answer_llm,
                    context=formatted_context,
                    question=query,
                    experiment_config=self._adapter._convert_config_to_experiment_config(),
                )
            )

        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "raw_question": raw_question,
                "recall_query": query,
                "retrieved_memories": search_result.results,
                "retrieved_context": formatted_context,
                "prediction": prediction,
            }
        )
        self._flush_debug(dataset)
        return prediction

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
