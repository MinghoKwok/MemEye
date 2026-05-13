import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI

from .common import SCRIPT_DIR, write_json
from .dataset import MemoryBenchmarkDataset, round_image_captions, validate_text_only_captions
from .methods import HistoryMethod


_MEMORYOS_ROOT = (
    Path(__file__).resolve().parent / "memoryos" / "MemoryOS" / "memoryos-pypi"
).resolve()
_LOCAL_ENV_PATH = (SCRIPT_DIR / ".env.local").resolve()


def _load_memoryos_class() -> Any:
    module_name = "_memoryos_upstream"
    if module_name in sys.modules:
        module = sys.modules[module_name]
        return getattr(module, "Memoryos")

    if str(_MEMORYOS_ROOT) not in sys.path:
        sys.path.insert(0, str(_MEMORYOS_ROOT))

    init_path = _MEMORYOS_ROOT / "__init__.py"
    spec = importlib.util.spec_from_file_location(
        module_name,
        init_path,
        submodule_search_locations=[str(_MEMORYOS_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load MemoryOS package from {init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Official MemoryOS dependencies are not fully installed. Missing module: "
            f"{exc.name}. Install benchmark/memoryos/MemoryOS/"
            "memoryos-pypi/requirements.txt in the active environment."
        ) from exc
    return getattr(module, "Memoryos")


def _read_local_env() -> Dict[str, str]:
    if not _LOCAL_ENV_PATH.exists():
        return {}
    values: Dict[str, str] = {}
    for raw_line in _LOCAL_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


def _resolve_secret(
    primary: Dict[str, Any],
    secondary: Dict[str, Any],
    *,
    raw_key: str,
    env_key: str,
    default_env: str,
) -> str:
    raw = str(primary.get(raw_key, "")).strip() or str(secondary.get(raw_key, "")).strip()
    if raw:
        return raw
    env_name = str(primary.get(env_key, "")).strip() or str(secondary.get(env_key, "")).strip() or default_env
    if env_name:
        env_value = str(os.getenv(env_name, "")).strip()
        if env_value:
            return env_value
        return _read_local_env().get(env_name, "").strip()
    return ""


def _question_caption_text(qa: Dict[str, Any]) -> str:
    raw = qa.get("question_image_caption")
    if raw is None:
        raw = qa.get("image_caption")
    if isinstance(raw, list):
        return " ".join(str(item).strip() for item in raw if str(item).strip())
    return str(raw or "").strip()


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    query = question.strip()
    has_question_image = bool(qa.get("question_image") or qa.get("question_images"))
    caption_text = _question_caption_text(qa)
    if not has_question_image or not caption_text:
        return query
    return f"{query}\nquestion's image:\nimage_caption: {caption_text}"


def _round_user_with_captions(round_payload: Dict[str, Any]) -> str:
    user_text = str(round_payload.get("user", "")).strip()
    captions = round_image_captions(round_payload)
    if not captions:
        return user_text

    lines: List[str] = [user_text] if user_text else []
    for caption in captions:
        lines.extend(
            [
                "image:",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def _stored_memory_text(user_input: str, assistant_response: str) -> str:
    parts: List[str] = []
    if user_input:
        parts.append(f"User: {user_input}")
    if assistant_response:
        parts.append(f"Assistant: {assistant_response}")
    return "\n".join(parts).strip()


def _load_json_object(raw: str, fallback_key: str) -> str:
    try:
        payload = json.loads(raw)
        if isinstance(payload, dict):
            value = payload.get(fallback_key, "")
            if isinstance(value, str):
                return value.strip()
    except Exception:
        pass
    return raw.strip()


def _format_retrieval_context(retrieval_results: Dict[str, Any]) -> str:
    pages = retrieval_results.get("retrieved_pages", []) or []
    user_knowledge = retrieval_results.get("retrieved_user_knowledge", []) or []
    assistant_knowledge = retrieval_results.get("retrieved_assistant_knowledge", []) or []

    sections: List[str] = []
    if pages:
        page_lines: List[str] = []
        for idx, page in enumerate(pages, start=1):
            page_lines.append(
                "\n".join(
                    [
                        f"[Retrieved Memory {idx}]",
                        f"User: {str(page.get('user_input', '')).strip()}",
                        f"Assistant: {str(page.get('agent_response', '')).strip()}",
                        f"Time: {str(page.get('timestamp', '')).strip()}",
                        f"Conversation chain overview: {str(page.get('meta_info', 'N/A')).strip()}",
                    ]
                ).strip()
            )
        sections.append("\n\n".join(page_lines))

    if user_knowledge:
        knowledge_lines = [
            f"- {str(entry.get('knowledge', '')).strip()} (Recorded: {str(entry.get('timestamp', '')).strip()})"
            for entry in user_knowledge
            if str(entry.get("knowledge", "")).strip()
        ]
        if knowledge_lines:
            sections.append("Relevant User Knowledge:\n" + "\n".join(knowledge_lines))

    if assistant_knowledge:
        knowledge_lines = [
            f"- {str(entry.get('knowledge', '')).strip()} (Recorded: {str(entry.get('timestamp', '')).strip()})"
            for entry in assistant_knowledge
            if str(entry.get("knowledge", "")).strip()
        ]
        if knowledge_lines:
            sections.append("Relevant Assistant Knowledge:\n" + "\n".join(knowledge_lines))

    return "\n\n".join(section for section in sections if section).strip()


class MemoryOSAgent:
    def __init__(
        self,
        method_config: Dict[str, Any],
        model_config: Dict[str, Any],
        *,
        data_storage_path: Path,
    ) -> None:
        memoryos_cls = _load_memoryos_class()

        self.core_model = (
            str(method_config.get("llm_model", "")).strip()
            or str(model_config.get("model", "")).strip()
            or "gpt-4.1-nano"
        )
        self.core_api_key = _resolve_secret(
            method_config,
            model_config,
            raw_key="llm_api_key",
            env_key="llm_api_key_env",
            default_env="OPENAI_API_KEY",
        ) or _resolve_secret(
            model_config,
            method_config,
            raw_key="api_key",
            env_key="api_key_env",
            default_env="OPENAI_API_KEY",
        )
        self.core_base_url = (
            str(method_config.get("llm_base_url", "")).strip()
            or str(model_config.get("base_url", "")).strip()
            or "https://api.openai.com/v1"
        )
        if not self.core_api_key:
            raise ValueError(
                "MemoryOS requires an OpenAI-compatible API key. "
                "Set it in .env.local or the active environment."
            )

        self.answer_provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        self.answer_model = str(model_config.get("model", "")).strip() or self.core_model
        self.answer_api_key = _resolve_secret(
            model_config,
            method_config,
            raw_key="api_key",
            env_key="api_key_env",
            default_env="OPENAI_API_KEY",
        )
        self.answer_base_url = (
            str(model_config.get("base_url", "")).strip()
            or str(method_config.get("answer_base_url", "")).strip()
            or self.core_base_url
        )
        self.answer_timeout = int(model_config.get("timeout", method_config.get("llm_timeout", 90)) or 90)

        if self.answer_provider not in ("openai_api", "gemini_api"):
            raise ValueError(
                f"Unsupported answer model provider for MemoryOS benchmark QA: {self.answer_provider}. "
                "Use an OpenAI API or Gemini API model config for final answer generation."
            )
        if not self.answer_api_key:
            raise ValueError("API key not found for MemoryOS benchmark answer model.")

        self.answer_client = OpenAI(
            api_key=self.answer_api_key,
            base_url=self.answer_base_url,
            timeout=self.answer_timeout,
        )
        self.memoryos = memoryos_cls(
            user_id=str(method_config.get("user_id", "memorybench_user")).strip() or "memorybench_user",
            assistant_id=str(method_config.get("assistant_id", "memorybench_assistant")).strip() or "memorybench_assistant",
            openai_api_key=self.core_api_key,
            openai_base_url=self.core_base_url,
            data_storage_path=str(data_storage_path),
            llm_model=self.core_model,
            short_term_capacity=int(method_config.get("short_term_capacity", 10) or 10),
            mid_term_capacity=int(method_config.get("mid_term_capacity", 2000) or 2000),
            long_term_knowledge_capacity=int(method_config.get("long_term_knowledge_capacity", 100) or 100),
            retrieval_queue_capacity=int(method_config.get("retrieval_queue_capacity", 10) or 10),
            mid_term_heat_threshold=float(method_config.get("mid_term_heat_threshold", 5.0) or 5.0),
            mid_term_similarity_threshold=float(method_config.get("mid_term_similarity_threshold", 0.6) or 0.6),
            embedding_model_name=(
                str(method_config.get("embedding_model", "all-MiniLM-L6-v2")).strip() or "all-MiniLM-L6-v2"
            ),
        )

    def add_memory(self, user_input: str, agent_response: str, timestamp: Optional[str]) -> None:
        self.memoryos.add_memory(user_input=user_input, agent_response=agent_response, timestamp=timestamp)

    def answer_question(self, query: str) -> Tuple[str, str, Dict[str, Any]]:
        retrieval_results = self.memoryos.retriever.retrieve_context(
            user_query=query,
            user_id=self.memoryos.user_id,
        )
        context = _format_retrieval_context(retrieval_results)
        user_prompt = (
            "Use the retrieved MemoryOS context to answer the benchmark question. "
            "Answer briefly and use exact words from the context whenever possible.\n\n"
            f"Context:\n{context or 'No relevant memories retrieved.'}\n\n"
            f"Question:\n{query}\n\n"
            "Return only the short answer."
        )
        response = self.answer_client.chat.completions.create(
            model=self.answer_model,
            messages=[
                {"role": "system", "content": "You must respond with a JSON object."},
                {"role": "user", "content": user_prompt},
            ],
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            temperature=0.0,
        )
        raw_response = response.choices[0].message.content
        return _load_json_object(raw_response, "answer"), user_prompt, retrieval_results


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


class MemoryOSMethod(HistoryMethod):
    name = "memoryos"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._agent: Optional[MemoryOSAgent] = None
        self._debug_rows: List[Dict[str, Any]] = []
        self._runtime_signature_payload_cache: Dict[int, Dict[str, Any]] = {}
        self._runtime_signature_cache: Dict[int, str] = {}

    def _safe_task_name(self, dataset: MemoryBenchmarkDataset) -> str:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        return task_name.lower().replace(" ", "_").replace("/", "_")

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (
            SCRIPT_DIR
            / "output"
            / self._safe_task_name(dataset)
            / "memoryos"
            / self._runtime_signature(dataset)
        ).resolve()

    def _runtime_data_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (
            SCRIPT_DIR
            / "runs"
            / self._safe_task_name(dataset)
            / "memoryos_data"
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
            "code_version": 1,
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

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._agent is not None and self._dataset_key == dataset_id:
            return

        self.runtime_info.update(validate_text_only_captions(dataset.rounds))
        self._debug_rows = []
        model_config = dict(self.config.get("_model_cfg", {}))
        runtime_data_dir = self._runtime_data_dir(dataset)
        runtime_data_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            runtime_data_dir / "signature.json",
            {
                "runtime_signature": self._runtime_signature(dataset),
                "signature_payload": self._runtime_signature_payload(dataset),
            },
        )
        self._agent = MemoryOSAgent(self.config, model_config, data_storage_path=runtime_data_dir)

        stored_count = 0
        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            timestamp = str(session.get("date", "")).strip() or None
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                user_input = _round_user_with_captions(round_payload)
                agent_response = str(round_payload.get("assistant", "")).strip()
                if not user_input and not agent_response:
                    continue
                assert self._agent is not None
                self._agent.add_memory(user_input=user_input, agent_response=agent_response, timestamp=timestamp)
                stored_count += 1
                self._debug_rows.append(
                    {
                        "type": "stored_memory",
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp or "",
                        "text": _stored_memory_text(user_input, agent_response),
                    }
                )

        self._dataset_key = dataset_id
        self.runtime_info["num_memories"] = stored_count
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        self.runtime_info["data_storage_path"] = str(runtime_data_dir)
        self.runtime_info["runtime_signature"] = self._runtime_signature(dataset)
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._agent is not None
        query = _question_with_image_caption(qa, question)
        answer, answer_prompt, retrieval_results = self._agent.answer_question(query)
        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": query,
                "retrieved_memories": retrieval_results,
                "answer_prompt": answer_prompt,
                "prediction": answer,
            }
        )
        self._flush_debug(dataset)
        return answer

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
