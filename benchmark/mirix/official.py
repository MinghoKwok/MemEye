from __future__ import annotations

import json
import os
import socket
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
import hashlib
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

from ..common import SCRIPT_DIR, load_json, write_json
from ..dataset import MemoryBenchmarkDataset
from ..methods import HistoryMethod


_MIRIX_UPSTREAM_ROOT = (Path(__file__).resolve().parent / "upstream").resolve()
_PROMPT_DIR = (SCRIPT_DIR / "benchmark" / "prompt").resolve()
_OFFICIAL_PERSONA = (
    "Is a helpful assistant that answers questions with extreme conciseness.\n"
    "Is persistent and tries to find the answerr using different queries and different "
    "search methods. Never uses unnecessary words or repeats the question in the answer. "
    "Always provides the shortest answer possible and tries to utter the fewest words possible."
)
_LOCAL_ENV_PATH = (SCRIPT_DIR / ".env.local").resolve()


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


def _resolve_secret(config: Dict[str, Any], raw_key: str, env_key: str, default_env: str = "") -> str:
    raw = str(config.get(raw_key, "")).strip()
    if raw:
        return raw
    env_name = str(config.get(env_key, default_env)).strip() or default_env
    if env_name:
        return str(os.getenv(env_name, "")).strip()
    return ""


def _resolve_model_name(method_config: Dict[str, Any], model_config: Dict[str, Any]) -> str:
    explicit = str(method_config.get("official_model_name", "")).strip()
    if explicit:
        return explicit
    return str(model_config.get("model", "")).strip() or "gpt-4o-mini"


def _redact_secrets(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: Dict[str, Any] = {}
        for key, item in value.items():
            lowered = str(key).lower()
            is_sensitive = lowered in {
                "api_key",
                "openai_api_key",
                "gemini_api_key",
                "access_token",
                "refresh_token",
                "id_token",
                "secret",
                "client_secret",
                "password",
            } or lowered.endswith(("_api_key", "_token", "_secret", "_password"))
            if is_sensitive:
                redacted[str(key)] = "***"
            else:
                redacted[str(key)] = _redact_secrets(item)
        return redacted
    if isinstance(value, list):
        return [_redact_secrets(item) for item in value]
    return value


def _build_runtime_env(
    runtime_home: Path,
    method_config: Dict[str, Any],
    model_config: Dict[str, Any],
    runtime_model_name: str,
) -> Dict[str, str]:
    local_env = _read_local_env()
    mirix_dir = runtime_home / ".mirix"
    images_dir = mirix_dir / "images"
    env = {
        "HOME": str(runtime_home),
        "MIRIX_DIR": str(mirix_dir),
        "MIRIX_IMAGES_DIR": str(images_dir),
    }

    provider = str(model_config.get("provider", "")).strip().lower()
    runtime_model = str(runtime_model_name).strip().lower()

    openai_key = ""
    if provider in {"", "openai_api"} or runtime_model.startswith(("gpt-", "o4-", "o3-", "o1-")) or not runtime_model.startswith("gemini-"):
        openai_key = _resolve_secret(model_config, "api_key", "api_key_env", "OPENAI_API_KEY") or _resolve_secret(
            method_config, "api_key", "api_key_env", "OPENAI_API_KEY"
        )
        if not openai_key:
            openai_key = local_env.get("OPENAI_API_KEY", "").strip()
    if openai_key:
        env["OPENAI_API_KEY"] = openai_key

    openai_base = str(model_config.get("base_url", "")).strip() or str(method_config.get("base_url", "")).strip()
    if openai_base:
        env["OPENAI_API_BASE"] = openai_base

    gemini_key = ""
    if provider == "gemini_api" or runtime_model.startswith("gemini-"):
        gemini_key = _resolve_secret(model_config, "api_key", "api_key_env", "GEMINI_API_KEY") or _resolve_secret(
            method_config, "gemini_api_key", "gemini_api_key_env", "GEMINI_API_KEY"
        )
        if not gemini_key:
            gemini_key = local_env.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        env.setdefault("GEMINI_API_KEY", gemini_key)

    # MIRIX embedding (text-embedding-3-small) always needs a real OpenAI key,
    # even when the LLM is routed through OpenRouter or another provider.
    # Prefer explicit MIRIX_EMBEDDING_API_KEY from method config or env.
    embedding_key = (
        str(method_config.get("embedding_api_key", "")).strip()
        or os.environ.get("MIRIX_EMBEDDING_API_KEY", "").strip()
        or local_env.get("OPENAI_API_KEY", "").strip()
    )
    if embedding_key:
        env["MIRIX_EMBEDDING_API_KEY"] = embedding_key

    return env


@contextmanager
def _temporary_env(updates: Dict[str, str]) -> Generator[None, None, None]:
    previous: Dict[str, Optional[str]] = {}
    try:
        for key, value in updates.items():
            previous[key] = os.environ.get(key)
            os.environ[key] = value
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _load_official_agent_wrapper(runtime_env: Dict[str, str]) -> Any:
    if str(_MIRIX_UPSTREAM_ROOT) not in sys.path:
        sys.path.insert(0, str(_MIRIX_UPSTREAM_ROOT))
    try:
        with _temporary_env(runtime_env):
            from mirix.agent import AgentWrapper  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Official MIRIX dependencies are not fully installed. Missing module: "
            f"{exc.name}. Install benchmark/mirix/upstream/requirements.txt "
            "in the active environment."
        ) from exc
    return AgentWrapper


def _preflight_runtime_access(runtime_env: Dict[str, str], runtime_model_name: str) -> None:
    model = str(runtime_model_name).strip().lower()
    if model.startswith("gemini-"):
        if not runtime_env.get("GEMINI_API_KEY", "").strip():
            raise RuntimeError(
                "MIRIX preflight failed: GEMINI_API_KEY is not available. "
                "Set it in .env.local or the active environment."
            )
        try:
            socket.getaddrinfo("generativelanguage.googleapis.com", 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise RuntimeError(
                "MIRIX preflight failed: Gemini network/API unavailable. "
                "The runtime cannot resolve generativelanguage.googleapis.com. "
                "Run this benchmark from a network-enabled environment."
            ) from exc

    if model.startswith(("gpt-", "o4-", "o3-", "o1-")):
        if not runtime_env.get("OPENAI_API_KEY", "").strip():
            raise RuntimeError(
                "MIRIX preflight failed: OPENAI_API_KEY is not available. "
                "Set it in .env.local or the active environment."
            )
        try:
            socket.getaddrinfo("api.openai.com", 443, type=socket.SOCK_STREAM)
        except OSError as exc:
            raise RuntimeError(
                "MIRIX preflight failed: OpenAI network/API unavailable. "
                "The runtime cannot resolve api.openai.com. "
                "Run this benchmark from a network-enabled environment."
            ) from exc


def _safe_task_name(dataset: MemoryBenchmarkDataset) -> str:
    task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
    return task_name.lower().replace(" ", "_").replace("/", "_")


def _format_timestamp(raw_date: str, offset_seconds: int) -> str:
    base = str(raw_date or "").strip()
    if not base:
        return ""
    try:
        dt = datetime.fromisoformat(base)
    except ValueError:
        try:
            dt = datetime.strptime(base, "%Y-%m-%d")
        except ValueError:
            return base
    return (dt + timedelta(seconds=offset_seconds)).strftime("%Y-%m-%d %H:%M:%S")


def _round_user_message(round_payload: Dict[str, Any]) -> Optional[str]:
    text = str(round_payload.get("user", "")).strip()
    return text or None


def _round_assistant_message(round_payload: Dict[str, Any]) -> Optional[str]:
    text = str(round_payload.get("assistant", "")).strip()
    return text or None


def _load_shared_answer_prompt(qa: Dict[str, Any]) -> str:
    is_mcq = isinstance(qa.get("options"), dict)
    prompt_path = _PROMPT_DIR / ("sys_prompt_mcq.txt" if is_mcq else "sys_prompt_open.txt")
    if not prompt_path.exists():
        prompt_path = _PROMPT_DIR / "sys_prompt.txt"
    return prompt_path.read_text(encoding="utf-8").strip()


def _benchmark_persona(qa: Dict[str, Any]) -> str:
    shared_prompt = _load_shared_answer_prompt(qa)
    return f"{_OFFICIAL_PERSONA}\n\n{shared_prompt}".strip()


class _StrictMCQFormatter:
    def __init__(self, method_config: Dict[str, Any]) -> None:
        model_config = dict(method_config.get("_model_cfg", {}))
        self.provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        if self.provider not in {"openai_api", "gemini_api"}:
            raise ValueError(
                "Mirix strict MCQ formatting currently requires an OpenAI-compatible or Gemini benchmark model config."
            )

        self.model_name = _resolve_model_name(method_config, model_config)
        self.timeout = int(model_config.get("timeout", method_config.get("final_answer_timeout", 90)) or 90)
        if self.provider == "openai_api":
            try:
                from openai import OpenAI
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "OpenAI client is required for Mirix strict MCQ formatting."
                ) from exc

            self.api_key = _resolve_secret(model_config, "api_key", "api_key_env", "OPENAI_API_KEY") or _resolve_secret(
                method_config, "api_key", "api_key_env", "OPENAI_API_KEY"
            )
            if not self.api_key:
                self.api_key = _read_local_env().get("OPENAI_API_KEY", "").strip()
            if not self.api_key:
                raise ValueError("OpenAI API key not found for Mirix strict MCQ formatting.")

            self.base_url = (
                str(model_config.get("base_url", "")).strip()
                or str(method_config.get("final_answer_base_url", "")).strip()
                or "https://api.openai.com/v1"
            )
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)
        else:
            try:
                from google import genai
                from google.genai import types as genai_types
            except ModuleNotFoundError as exc:
                raise ModuleNotFoundError(
                    "Google GenAI client is required for Mirix Gemini strict MCQ formatting."
                ) from exc

            self.api_key = _resolve_secret(model_config, "api_key", "api_key_env", "GEMINI_API_KEY") or _resolve_secret(
                method_config, "gemini_api_key", "gemini_api_key_env", "GEMINI_API_KEY"
            )
            if not self.api_key:
                self.api_key = _read_local_env().get("GEMINI_API_KEY", "").strip()
            if not self.api_key:
                raise ValueError("GEMINI_API_KEY not found for Mirix strict MCQ formatting.")

            self.base_url = (
                str(model_config.get("base_url", "")).strip()
                or str(method_config.get("final_answer_base_url", "")).strip()
                or "https://generativelanguage.googleapis.com/v1beta"
            )
            self._gemini_types = genai_types
            self.client = genai.Client(
                api_key=self.api_key,
                http_options=genai_types.HttpOptions(base_url=self.base_url, timeout=self.timeout),
            )

    def format_answer(self, question: str, qa: Dict[str, Any], draft_answer: str) -> Tuple[str, str]:
        options = qa.get("options")
        if not isinstance(options, dict):
            return draft_answer, ""

        valid_options = sorted(str(key).strip() for key in options.keys() if str(key).strip())
        option_lines = [f"{key}. {options[key]}" for key in valid_options]
        user_prompt = (
            "You are producing the final benchmark answer.\n"
            "Use the benchmark multiple-choice question and the Mirix draft answer below.\n"
            f"Select exactly one option letter from: {', '.join(valid_options)}.\n"
            "Return the best final option even if the draft answer is verbose.\n\n"
            f"Benchmark question:\n{question}\n\n"
            f"Options:\n" + "\n".join(option_lines) + "\n\n"
            f"Mirix draft answer:\n{draft_answer or '[empty]'}"
        )
        if self.provider == "openai_api":
            response = self.client.chat.completions.create(
                model=self.model_name,
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
                            "properties": {"answer": {"type": "string", "enum": valid_options}},
                            "required": ["answer"],
                            "additionalProperties": False,
                        },
                        "strict": True,
                    },
                },
                temperature=0.0,
            )
            raw_response = response.choices[0].message.content
        else:
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=user_prompt,
                config=self._gemini_types.GenerateContentConfig(
                    system_instruction="You must respond with a JSON object.",
                    response_mime_type="application/json",
                    response_schema={
                        "type": "object",
                        "properties": {"answer": {"type": "string", "enum": valid_options}},
                        "required": ["answer"],
                        "additionalProperties": False,
                    },
                    temperature=0.0,
                ),
            )
            raw_response = str(getattr(response, "text", "") or "").strip()
        payload = json.loads(raw_response)
        answer = str(payload.get("answer", "")).strip()
        if answer not in valid_options:
            raise ValueError(f"Invalid Mirix MCQ formatter answer: {answer!r}")
        return answer, user_prompt


class _OfficialMIRIXAgent:
    def __init__(
        self,
        method_config: Dict[str, Any],
        model_config: Dict[str, Any],
        runtime_home: Path,
        config_path: Path,
    ) -> None:
        runtime_model_name = _resolve_model_name(method_config, model_config)
        self.runtime_env = _build_runtime_env(runtime_home, method_config, model_config, runtime_model_name)
        _preflight_runtime_access(self.runtime_env, runtime_model_name)
        agent_wrapper_cls = _load_official_agent_wrapper(self.runtime_env)
        with _temporary_env(self.runtime_env):
            self.agent = agent_wrapper_cls(str(config_path))
            self.agent.update_core_memory_persona(_OFFICIAL_PERSONA)

    def memorize(
        self,
        *,
        message: Optional[str],
        image_uris: Optional[List[str]],
        timestamp: str,
    ) -> None:
        with _temporary_env(self.runtime_env):
            self.agent.send_message(
                message=message,
                image_uris=image_uris or None,
                memorizing=True,
                force_absorb_content=message is not None,
                delete_after_upload=False,
                async_upload=True,
                specific_timestamps=[timestamp] if timestamp else None,
            )

    def answer(
        self,
        question: str,
        *,
        system_prompt: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        with _temporary_env(self.runtime_env):
            self.agent.update_core_memory_persona(system_prompt)
            response = self.agent.send_message(
                message=question,
                image_uris=question_images or None,
                memorizing=False,
                delete_after_upload=False,
            )
        return str(response or "").strip()


class OfficialMIRIXMethod(HistoryMethod):
    name = "mirix"
    fixed_modality = "multimodal"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._agent: Optional[_OfficialMIRIXAgent] = None
        model_config = dict(self.config.get("_model_cfg", {}))
        provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        self._mcq_formatter: Optional[_StrictMCQFormatter] = (
            _StrictMCQFormatter(self.config) if provider in {"openai_api", "gemini_api"} else None
        )
        self._debug_rows: List[Dict[str, Any]] = []
        self._runtime_home: Optional[Path] = None
        self._runtime_signature_payload_cache: Dict[int, Dict[str, Any]] = {}
        self._runtime_signature_cache: Dict[int, str] = {}

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        path = (
            SCRIPT_DIR
            / "output"
            / _safe_task_name(dataset)
            / "mirix"
            / self._runtime_signature(dataset)
        ).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _runtime_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        path = (
            SCRIPT_DIR
            / "runs"
            / _safe_task_name(dataset)
            / self.name
            / self._runtime_signature(dataset)
        ).resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _runtime_home_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        path = (self._runtime_dir(dataset) / "runtime_home").resolve()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _write_runtime_signature(self, dataset: MemoryBenchmarkDataset) -> None:
        write_json(
            self._runtime_dir(dataset) / "signature.json",
            {
                "runtime_signature": self._runtime_signature(dataset),
                "signature_payload": self._runtime_signature_payload(dataset),
            },
        )
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
        eval_cfg = dict(self.config.get("_eval_cfg", {}))
        payload = {
            "task_name": str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem,
            "dialog_json_path": str(dialog_path),
            "dialog_sha256": dialog_sha256,
            "method_name": self.name,
            "method_config": method_config,
            "model_config": model_cfg,
            "eval_mode": str(eval_cfg.get("mode", "")).strip(),
            "code_version": 3,
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
        write_json(
            self._debug_dir(dataset) / "debug_trace.json",
            {
                "dataset_path": str(dataset.dialog_json_path),
                "runtime_signature": self._runtime_signature(dataset),
                "runtime_home": str(self._runtime_home) if self._runtime_home else "",
                "rows": self._debug_rows,
            },
        )

    def _resume_state_path(self, dataset: MemoryBenchmarkDataset) -> Path:
        return self._debug_dir(dataset) / "resume_state.json"

    def _load_resume_state(
        self,
        dataset: MemoryBenchmarkDataset,
        model_name: str,
        runtime_signature: str,
    ) -> Optional[Dict[str, Any]]:
        state_path = self._resume_state_path(dataset)
        if not state_path.exists():
            return None
        try:
            state = load_json(state_path)
        except Exception:
            return None
        runtime_home = Path(str(state.get("runtime_home", ""))).resolve()
        if (
            str(state.get("dataset_path", "")) != str(dataset.dialog_json_path)
            or str(state.get("model_name", "")) != model_name
            or str(state.get("runtime_signature", "")) != runtime_signature
            or not runtime_home.exists()
        ):
            return None
        return state

    def _save_resume_state(self, dataset: MemoryBenchmarkDataset, state: Dict[str, Any]) -> None:
        write_json(self._resume_state_path(dataset), state)

    def _load_existing_debug_rows(self, dataset: MemoryBenchmarkDataset, runtime_home: Path) -> List[Dict[str, Any]]:
        trace_path = self._debug_dir(dataset) / "debug_trace.json"
        if not trace_path.exists():
            return []
        try:
            payload = load_json(trace_path)
        except Exception:
            return []
        if str(payload.get("runtime_signature", "")) != str(self._runtime_signature(dataset)):
            return []
        if str(payload.get("runtime_home", "")) != str(runtime_home):
            return []
        rows = payload.get("rows", [])
        return rows if isinstance(rows, list) else []

    def _write_official_config(self, model_name: str, runtime_home: Path) -> Path:
        config_path = runtime_home / "mirix_agent.yaml"
        config_path.write_text(
            f"agent_name: mirix\nmodel_name: {model_name}\n",
            encoding="utf-8",
        )
        return config_path

    def _memorize_with_context(
        self,
        *,
        session_id: str,
        round_id: str,
        phase: str,
        message: Optional[str],
        image_uris: Optional[List[str]],
        timestamp: str,
    ) -> None:
        assert self._agent is not None
        try:
            self._agent.memorize(
                message=message,
                image_uris=image_uris,
                timestamp=timestamp,
            )
        except Exception as exc:
            raise RuntimeError(
                "MIRIX ingestion failed "
                f"(session={session_id}, round={round_id}, phase={phase}, timestamp={timestamp})"
            ) from exc

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._agent is not None and self._dataset_key == dataset_id:
            return

        model_config = dict(self.config.get("_model_cfg", {}))
        model_name = _resolve_model_name(self.config, model_config)
        runtime_signature = self._runtime_signature(dataset)
        resume_state = self._load_resume_state(dataset, model_name, runtime_signature)
        if resume_state is not None:
            self._runtime_home = Path(str(resume_state["runtime_home"])).resolve()
        else:
            self._runtime_home = self._runtime_home_dir(dataset)
        self._debug_rows = self._load_existing_debug_rows(dataset, self._runtime_home)
        config_path = self._write_official_config(model_name, self._runtime_home)
        self._agent = _OfficialMIRIXAgent(self.config, model_config, self._runtime_home, config_path)

        state = resume_state or {
            "dataset_path": str(dataset.dialog_json_path),
            "model_name": model_name,
            "runtime_signature": runtime_signature,
            "runtime_home": str(self._runtime_home),
            "next_session_idx": 0,
            "next_dialogue_idx": 0,
            "next_phase": "user",
            "turn_offset": 0,
            "ingest_count": 0,
            "completed_ingestion": False,
        }
        self.runtime_info = {
            "official_mirix": True,
            "official_model_name": model_name,
            "ingested_turns": int(state.get("ingest_count", 0)),
            "runtime_signature": runtime_signature,
            "runtime_dir": str(self._runtime_dir(dataset)),
            "runtime_home": str(self._runtime_home),
            "debug_dir": str(self._debug_dir(dataset)),
            "resumed": resume_state is not None,
        }
        self._write_runtime_signature(dataset)
        write_json(
            self._debug_dir(dataset) / "signature.json",
            {
                "runtime_signature": runtime_signature,
                "signature_payload": self._runtime_signature_payload(dataset),
            },
        )
        session_ids = dataset.session_order()
        for session_idx in range(int(state.get("next_session_idx", 0)), len(session_ids)):
            session_id = session_ids[session_idx]
            session = dataset.get_session(session_id)
            session_date = str(session.get("date", "")).strip()
            start_dialogue_idx = int(state.get("next_dialogue_idx", 0)) if session_idx == int(state.get("next_session_idx", 0)) else 0
            for dialogue_idx in range(start_dialogue_idx, len(session.get("dialogues", []))):
                dialogue = session.get("dialogues", [])[dialogue_idx]
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    state.update(
                        {
                            "next_session_idx": session_idx,
                            "next_dialogue_idx": dialogue_idx + 1,
                            "next_phase": "user",
                        }
                    )
                    self._save_resume_state(dataset, state)
                    continue
                round_payload = dataset.rounds[round_id]
                current_phase = str(state.get("next_phase", "user"))
                timestamp = _format_timestamp(session_date, int(state.get("turn_offset", 0)))
                user_message = _round_user_message(round_payload)
                round_images = list(round_payload.get("images", []) or [])
                if current_phase == "user" and (user_message is not None or round_images):
                    self._memorize_with_context(
                        session_id=session_id,
                        round_id=round_id,
                        phase="user",
                        message=user_message,
                        image_uris=round_images,
                        timestamp=timestamp,
                    )
                    self._debug_rows.append(
                        {
                            "type": "memorize_user",
                            "session_id": session_id,
                            "round_id": round_id,
                            "timestamp": timestamp,
                            "message": user_message or "",
                            "images": round_images,
                        }
                    )
                    state["ingest_count"] = int(state.get("ingest_count", 0)) + 1
                    state["turn_offset"] = int(state.get("turn_offset", 0)) + 1
                    state["next_phase"] = "assistant"
                    self.runtime_info["ingested_turns"] = int(state["ingest_count"])
                    self._flush_debug(dataset)
                    self._save_resume_state(dataset, state)
                else:
                    state["next_phase"] = "assistant"

                assistant_message = _round_assistant_message(round_payload)
                if assistant_message is not None and str(state.get("next_phase", "assistant")) == "assistant":
                    assistant_timestamp = _format_timestamp(session_date, int(state.get("turn_offset", 0)))
                    self._memorize_with_context(
                        session_id=session_id,
                        round_id=round_id,
                        phase="assistant",
                        message=assistant_message,
                        image_uris=None,
                        timestamp=assistant_timestamp,
                    )
                    self._debug_rows.append(
                        {
                            "type": "memorize_assistant",
                            "session_id": session_id,
                            "round_id": round_id,
                            "timestamp": assistant_timestamp,
                            "message": assistant_message,
                            "images": [],
                        }
                    )
                    state["ingest_count"] = int(state.get("ingest_count", 0)) + 1
                    state["turn_offset"] = int(state.get("turn_offset", 0)) + 1
                    self.runtime_info["ingested_turns"] = int(state["ingest_count"])
                    self._flush_debug(dataset)
                state.update(
                    {
                        "next_session_idx": session_idx,
                        "next_dialogue_idx": dialogue_idx + 1,
                        "next_phase": "user",
                    }
                )
                self._save_resume_state(dataset, state)

        self._dataset_key = dataset_id
        state["completed_ingestion"] = True
        self.runtime_info.update(
            {
                "ingested_turns": int(state.get("ingest_count", 0)),
                "completed_ingestion": True,
            }
        )
        self._save_resume_state(dataset, state)
        self._flush_debug(dataset)

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._agent is not None
        benchmark_persona = _benchmark_persona(qa)
        draft_prediction = self._agent.answer(
            question,
            system_prompt=benchmark_persona,
            question_images=question_images,
        )
        prediction = draft_prediction
        formatter_prompt = ""
        if isinstance(qa.get("options"), dict):
            if self._mcq_formatter is None:
                raise ValueError(
                    "Mirix MCQ final-answer formatting requires an OpenAI-compatible benchmark model config."
                )
            prediction, formatter_prompt = self._mcq_formatter.format_answer(question, qa, draft_prediction)
        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "question_images": list(question_images or []),
                "answer_prompt": benchmark_persona,
                "draft_prediction": draft_prediction,
                "final_answer_prompt": formatter_prompt,
                "prediction": prediction,
            }
        )
        self._flush_debug(dataset)
        return prediction

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
