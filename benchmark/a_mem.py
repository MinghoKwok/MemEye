import json
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

from openai import APIConnectionError, APITimeoutError, InternalServerError, OpenAI, RateLimitError

from .common import REPO_ROOT, write_json
from .dataset import MemoryBenchmarkDataset
from .methods import HistoryMethod


_A_MEM_ROOT = (Path(__file__).resolve().parent / "a-mem" / "A-mem").resolve()
if str(_A_MEM_ROOT) not in sys.path:
    sys.path.insert(0, str(_A_MEM_ROOT))

if TYPE_CHECKING:
    from memory_layer import AgenticMemorySystem, LLMController  # type: ignore


def _load_a_mem_classes() -> Tuple[Any, Any]:
    try:
        from memory_layer import AgenticMemorySystem, LLMController  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "A-MEM dependencies are not fully installed. Missing module: "
            f"{exc.name}. Install benchmark/a-mem/A-mem/requirements.txt in the active environment."
        ) from exc
    return AgenticMemorySystem, LLMController


def _resolve_openai_api_key(method_config: Dict[str, Any]) -> Optional[str]:
    raw = str(method_config.get("llm_api_key", "")).strip()
    if raw:
        return raw
    env_name = str(method_config.get("llm_api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    return os.getenv(env_name)


def _resolve_openai_base_url(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> Optional[str]:
    raw = str(method_config.get("llm_base_url", "")).strip()
    if raw:
        return raw
    raw = str(model_config.get("base_url", "")).strip()
    return raw or None


def _resolve_openai_timeout(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> int:
    try:
        return int(method_config.get("llm_timeout", model_config.get("timeout", 90)) or 90)
    except Exception:
        return 90


def _resolve_model_api_key(model_config: Dict[str, Any]) -> Optional[str]:
    raw = str(model_config.get("api_key", "")).strip()
    if raw:
        return raw
    env_name = str(model_config.get("api_key_env", "OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
    return os.getenv(env_name)


def _normalize_backend(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    backend = str(method_config.get("backend", "")).strip().lower()
    if backend:
        return backend

    provider = str(model_config.get("provider", "")).strip().lower()
    if provider == "openai_api":
        return "openai"
    if provider in {"qwen_local", "sglang"}:
        return "sglang"
    return provider or "openai"


def _resolve_model_name(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    explicit = str(method_config.get("llm_model", "")).strip()
    if explicit:
        return explicit
    return str(model_config.get("model", "")).strip() or str(model_config.get("name", "gpt-4o-mini")).strip()


def _resolve_sglang_host(method_config: Dict[str, Any]) -> str:
    return str(method_config.get("sglang_host", "http://localhost")).strip() or "http://localhost"


def _resolve_sglang_port(method_config: Dict[str, Any]) -> int:
    try:
        return int(method_config.get("sglang_port", 30000))
    except Exception:
        return 30000


def _dialogue_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    parts: List[str] = []
    user_text = str(round_payload.get("user", "")).strip()
    assistant_text = str(round_payload.get("assistant", "")).strip()
    if user_text:
        parts.append(f"{speaker_a}: {user_text}")
    if assistant_text:
        parts.append(f"{speaker_b}: {assistant_text}")
    return "\n".join(parts).strip()


def _round_image_blocks(raw_dialogue: Dict[str, Any]) -> List[Tuple[str, str]]:
    input_images = raw_dialogue.get("input_image", []) or []
    captions = raw_dialogue.get("image_caption", []) or []
    image_ids = raw_dialogue.get("image_id", []) or []
    blocks: List[Tuple[str, str]] = []
    for idx, _ in enumerate(input_images):
        image_id = ""
        if idx < len(image_ids):
            image_id = str(image_ids[idx]).strip()
        caption = ""
        if idx < len(captions):
            caption = str(captions[idx]).strip()
        blocks.append((image_id, caption))
    return blocks


def build_a_mem_note_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    raw_dialogue = round_payload.get("raw", {}) or {}
    text = _dialogue_text(round_payload, speaker_a, speaker_b)
    lines: List[str] = [text] if text else []
    for image_id, caption in _round_image_blocks(raw_dialogue):
        lines.extend(
            [
                "image:",
                f"image_id: {image_id}",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    query = question.strip()
    question_image = str(qa.get("question_image", "")).strip()
    question_caption = qa.get("image_caption")
    if not question_image or not question_caption:
        return query

    if isinstance(question_caption, list):
        caption_text = " ".join(str(item).strip() for item in question_caption if str(item).strip())
    else:
        caption_text = str(question_caption).strip()
    if not caption_text:
        return query
    return f"{query}\nquestion's image:\nimage_caption: {caption_text}"


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


class AMemAgent:
    def __init__(self, method_config: Dict[str, Any], model_config: Dict[str, Any]) -> None:
        agentic_memory_system_cls, llm_controller_cls = _load_a_mem_classes()
        embedding_model = str(method_config.get("embedding_model", "all-MiniLM-L6-v2")).strip() or "all-MiniLM-L6-v2"
        backend = _normalize_backend(method_config, model_config)
        model_name = _resolve_model_name(method_config, model_config)
        api_key = _resolve_openai_api_key(method_config) if backend == "openai" else None
        llm_base_url = _resolve_openai_base_url(method_config, model_config)
        llm_timeout = _resolve_openai_timeout(method_config, model_config)
        self.answer_model = str(model_config.get("model", "")).strip() or "gpt-4.1-nano"
        self.answer_provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        self.answer_api_key = _resolve_model_api_key(model_config)
        self.answer_base_url = str(model_config.get("base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1"
        self.answer_timeout = int(model_config.get("timeout", 90) or 90)
        self.retrieve_k = max(1, int(method_config.get("retrieve_k", 10)))
        self.temperature_c5 = float(method_config.get("temperature_c5", 0.5))
        self.memory_system = agentic_memory_system_cls(
            model_name=embedding_model,
            llm_backend=backend,
            llm_model=model_name,
            api_key=api_key,
            api_base=llm_base_url,
            api_timeout=llm_timeout,
            sglang_host=_resolve_sglang_host(method_config),
            sglang_port=_resolve_sglang_port(method_config),
        )
        self.retriever_llm = llm_controller_cls(
            backend=backend,
            model=model_name,
            api_key=api_key,
            api_base=llm_base_url,
            api_timeout=llm_timeout,
            sglang_host=_resolve_sglang_host(method_config),
            sglang_port=_resolve_sglang_port(method_config),
        )
        self.answer_client: Optional[OpenAI] = None
        if self.answer_provider in ("openai_api", "gemini_api"):
            if not self.answer_api_key:
                raise ValueError("API key not found for benchmark answer model.")
            self.answer_client = OpenAI(api_key=self.answer_api_key, base_url=self.answer_base_url, timeout=self.answer_timeout)
        self._answer_retryable_errors = (APIConnectionError, APITimeoutError, RateLimitError, InternalServerError)

    def _create_answer_with_retry(self, **kwargs: Any) -> Any:
        if self.answer_client is None:
            raise ValueError("OpenAI answer client is not initialized.")
        max_retries = 4
        for attempt in range(max_retries):
            try:
                return self.answer_client.chat.completions.create(**kwargs)
            except self._answer_retryable_errors:
                if attempt == max_retries - 1:
                    raise
                time.sleep(1.5 * (2 ** attempt))

    def add_memory(self, content: str, timestamp: Optional[str]) -> None:
        self.memory_system.add_note(content, time=timestamp)

    def retrieve_memory(self, query: str) -> str:
        return self.memory_system.find_related_memories_raw(query, k=self.retrieve_k)

    def generate_query(self, question: str) -> str:
        prompt = f"""Given the following question, generate several keywords, using 'cosmos' as the separator.

Question: {question}

Format your response as a JSON object with a "keywords" field containing the selected text.

Example response format:
{{"keywords": "keyword1, keyword2, keyword3"}}"""
        response = self.retriever_llm.llm.get_completion(
            prompt,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "response",
                    "schema": {
                        "type": "object",
                        "properties": {"keywords": {"type": "string"}},
                        "required": ["keywords"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
        )
        return _load_json_object(response, "keywords")

    def answer_question(self, question: str) -> Tuple[str, str, str]:
        keywords = self.generate_query(question)
        context = self.retrieve_memory(keywords)
        user_prompt = f"""Based on the context: {context}, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {question} Short answer:
"""
        if self.answer_client is None:
            raise ValueError(
                f"Unsupported answer model provider for A-MEM benchmark QA: {self.answer_provider}. "
                "Use an OpenAI API model config for final answer generation."
            )
        response = self._create_answer_with_retry(
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
        return _load_json_object(raw_response, "answer"), user_prompt, context


class AMemMethod(HistoryMethod):
    name = "a_mem"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._agent: Optional[AMemAgent] = None
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._debug_rows: List[Dict[str, Any]] = []

    def _ensure_caption_preprocessed(self, dataset: MemoryBenchmarkDataset) -> None:
        if not bool(self.config.get("caption_preprocessed", True)):
            return
        missing_rounds: List[str] = []
        for round_id, payload in dataset.rounds.items():
            raw = payload.get("raw", {}) or {}
            images = raw.get("input_image", []) or []
            if not images:
                continue
            captions = raw.get("image_caption", []) or []
            if len(captions) != len(images):
                missing_rounds.append(round_id)
        if missing_rounds:
            raise ValueError(
                "A-MEM text-only adaptation requires Mem-Gallery-style image captions. "
                "Run the caption preprocess first. Missing/invalid image_caption for rounds: "
                + ", ".join(missing_rounds[:10])
                + ("..." if len(missing_rounds) > 10 else "")
            )

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        safe_task = task_name.lower().replace(" ", "_").replace("/", "_")
        return (REPO_ROOT / "output" / safe_task / "a_mem").resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._agent is not None and self._dataset_key == dataset_id:
            return

        self._ensure_caption_preprocessed(dataset)
        self._debug_rows = []
        model_config = dict(self.config.get("_model_cfg", {}))
        self._agent = AMemAgent(self.config, model_config)

        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name", "")).strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                note_text = build_a_mem_note_text(round_payload, self._speaker_a, self._speaker_b)
                if not note_text:
                    continue
                timestamp = str(session.get("date", "")).strip() or None
                assert self._agent is not None
                self._agent.add_memory(note_text, timestamp)
                self._debug_rows.append(
                    {
                        "type": "stored_memory",
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp or "",
                        "text": note_text,
                    }
                )

        self._dataset_key = dataset_id
        self.runtime_info["num_memories"] = len(self._debug_rows)
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._agent is not None
        query = _question_with_image_caption(qa, question)
        answer, answer_prompt, retrieved = self._agent.answer_question(query)
        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": query,
                "retrieved_memories": retrieved,
                "answer_prompt": answer_prompt,
                "prediction": answer,
            }
        )
        self._flush_debug(dataset)
        return answer

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
