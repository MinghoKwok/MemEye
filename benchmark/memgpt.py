import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from .common import SCRIPT_DIR, write_json
from .dataset import MemoryBenchmarkDataset, round_image_captions, validate_text_only_captions
from .methods import HistoryMethod


_MEMGPT_ROOT = (Path(__file__).resolve().parent / "memgpt" / "upstream").resolve()
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
    if not env_name:
        return ""
    value = str(os.getenv(env_name, "")).strip()
    if value:
        return value
    return _read_local_env().get(env_name, "").strip()


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


def _load_json_object(raw_response: str, field: str) -> str:
    try:
        payload = json.loads(raw_response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse JSON response for '{field}': {raw_response}") from exc
    value = payload.get(field)
    if not isinstance(value, str):
        raise ValueError(f"JSON response missing string field '{field}': {raw_response}")
    return value.strip()


def _load_openai_client_class() -> Any:
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError(
            "The benchmark-side strict MemGPT formatter requires the modern openai package "
            "in the active memorybench environment."
        ) from exc
    return OpenAI


class _CaptureInterface:
    def __init__(self) -> None:
        self.events: List[Dict[str, str]] = []

    async def internal_monologue(self, msg: str) -> None:
        self.events.append({"kind": "internal_monologue", "message": str(msg or "")})

    async def assistant_message(self, msg: str) -> None:
        self.events.append({"kind": "assistant_message", "message": str(msg or "")})

    async def function_message(self, msg: str) -> None:
        self.events.append({"kind": "function_message", "message": str(msg or "")})

    async def user_message(self, msg: str, raw: bool = False, debug: bool = False) -> None:
        self.events.append({"kind": "user_message", "message": str(msg or "")})

    async def system_message(self, msg: str) -> None:
        self.events.append({"kind": "system_message", "message": str(msg or "")})

    async def memory_message(self, msg: str) -> None:
        self.events.append({"kind": "memory_message", "message": str(msg or "")})


class _MemGPTModules:
    def __init__(self, *, runtime_home: Path, api_key: str, base_url: str) -> None:
        self.runtime_home = runtime_home
        self.api_key = api_key
        self.base_url = base_url

        if str(_MEMGPT_ROOT) not in sys.path:
            sys.path.insert(0, str(_MEMGPT_ROOT))

        os.environ["MEMGPT_CONFIG_PATH"] = str(runtime_home / "config")
        os.environ["OPENAI_API_KEY"] = api_key
        if base_url:
            os.environ["OPENAI_API_BASE"] = base_url

        self._install_questionary_shim()

        try:
            import openai
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Official MemGPT requires the legacy openai package in the active environment. "
                f"Missing module: {exc.name}."
            ) from exc

        self._install_openai_compat(openai)

        openai.api_key = api_key
        if base_url:
            openai.api_base = base_url

        try:
            import memgpt.agent as agent_module
            import memgpt.config as config_module
            import memgpt.constants as constants_module
            import memgpt.interface as interface_module
            import memgpt.openai_tools as openai_tools_module
            import memgpt.presets as presets_module
            import memgpt.system as system_module
            import memgpt.utils as utils_module
            from memgpt.config import AgentConfig, MemGPTConfig
            from memgpt.humans import humans as humans_module
            from memgpt.personas import personas as personas_module
            from memgpt.persistence_manager import LocalStateManager
        except Exception as exc:
            raise RuntimeError(
                "Bundled MemGPT could not be imported with the active environment. "
                "This adapter requires the upstream-style dependency stack for MemGPT, "
                "especially a compatible trio of openai, llama-index, and langchain packages. "
                f"Underlying import error: {exc}"
            ) from exc

        runtime_home_str = str(runtime_home)
        for module in (
            constants_module,
            config_module,
            utils_module,
            agent_module,
        ):
            if hasattr(module, "MEMGPT_DIR"):
                setattr(module, "MEMGPT_DIR", runtime_home_str)

        MemGPTConfig.config_path = str(runtime_home / "config")
        runtime_home.mkdir(parents=True, exist_ok=True)
        self._ensure_config(MemGPTConfig)

        openai_tools_module.HOST = base_url or os.getenv("OPENAI_API_BASE")
        if base_url:
            openai_tools_module.openai.api_base = base_url
        openai_tools_module.openai.api_key = api_key

        interface_module.STRIP_UI = True

        self.agent_module = agent_module
        self.config_module = config_module
        self.constants_module = constants_module
        self.interface_module = interface_module
        self.openai_tools_module = openai_tools_module
        self.presets_module = presets_module
        self.system_module = system_module
        self.utils_module = utils_module
        self.AgentConfig = AgentConfig
        self.MemGPTConfig = MemGPTConfig
        self.LocalStateManager = LocalStateManager
        self.personas_module = personas_module
        self.humans_module = humans_module

    def _install_openai_compat(self, openai: Any) -> None:
        if "openai.openai_object" not in sys.modules:
            openai_object_module = ModuleType("openai.openai_object")

            class OpenAIObject(dict):
                pass

            openai_object_module.OpenAIObject = OpenAIObject
            sys.modules["openai.openai_object"] = openai_object_module

        if hasattr(openai, "ChatCompletion") and hasattr(openai, "Embedding"):
            if not hasattr(openai, "error") and hasattr(openai, "RateLimitError"):
                openai.error = SimpleNamespace(RateLimitError=openai.RateLimitError)
            return

        if not hasattr(openai, "OpenAI") or not hasattr(openai, "AsyncOpenAI"):
            raise RuntimeError(
                "Bundled MemGPT expects either the legacy OpenAI client or the v1+ client with OpenAI/AsyncOpenAI."
            )

        if not hasattr(openai, "error") and hasattr(openai, "RateLimitError"):
            openai.error = SimpleNamespace(RateLimitError=openai.RateLimitError)

        def _client() -> Any:
            return openai.OpenAI(api_key=self.api_key, base_url=self.base_url)

        def _aclient() -> Any:
            return openai.AsyncOpenAI(api_key=self.api_key, base_url=self.base_url)

        def _normalize_content(content: Any) -> Optional[str]:
            if content is None or isinstance(content, str):
                return content
            if isinstance(content, list):
                parts: List[str] = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    else:
                        text = getattr(item, "text", None)
                        if text:
                            parts.append(str(text))
                return "".join(parts) if parts else None
            return str(content)

        def _convert_functions(kwargs: Dict[str, Any]) -> Dict[str, Any]:
            payload = dict(kwargs)
            functions = payload.pop("functions", None)
            function_call = payload.pop("function_call", None)
            if functions is not None:
                payload["tools"] = [{"type": "function", "function": fn} for fn in functions]
            if function_call and function_call != "auto":
                if isinstance(function_call, str):
                    payload["tool_choice"] = {"type": "function", "function": {"name": function_call}}
                else:
                    payload["tool_choice"] = function_call
            return payload

        class _CompatMessage(dict):
            def __getattr__(self, name: str) -> Any:
                try:
                    return self[name]
                except KeyError as exc:
                    raise AttributeError(name) from exc

            def copy(self) -> Dict[str, Any]:
                return dict(self)

        class _CompatChoice:
            def __init__(self, *, finish_reason: str, message: _CompatMessage) -> None:
                self.finish_reason = finish_reason
                self.message = message

        class _CompatResponse:
            def __init__(self, *, choices: List[_CompatChoice], usage: Dict[str, Any]) -> None:
                self.choices = choices
                self.usage = usage

            def __getitem__(self, key: str) -> Any:
                return getattr(self, key)

        def _convert_response(response: Any) -> _CompatResponse:
            choices: List[_CompatChoice] = []
            for choice in response.choices:
                message = choice.message
                function_call = None
                tool_calls = getattr(message, "tool_calls", None) or []
                if tool_calls:
                    tool_call = tool_calls[0]
                    function = getattr(tool_call, "function", None)
                    if function is not None:
                        function_call = {
                            "name": getattr(function, "name", ""),
                            "arguments": getattr(function, "arguments", "") or "",
                        }
                compat_message = _CompatMessage(
                    role=getattr(message, "role", "assistant"),
                    content=_normalize_content(getattr(message, "content", None)),
                )
                if function_call is not None:
                    compat_message["function_call"] = function_call
                choices.append(
                    _CompatChoice(
                        finish_reason=getattr(choice, "finish_reason", "stop") or "stop",
                        message=compat_message,
                    )
                )
            usage_obj = getattr(response, "usage", None)
            usage = {
                "total_tokens": getattr(usage_obj, "total_tokens", 0) if usage_obj is not None else 0,
            }
            return _CompatResponse(choices=choices, usage=usage)

        class _ChatCompletion:
            @staticmethod
            def create(**kwargs: Any) -> _CompatResponse:
                payload = _convert_functions(kwargs)
                response = _client().chat.completions.create(**payload)
                return _convert_response(response)

            @staticmethod
            async def acreate(**kwargs: Any) -> _CompatResponse:
                payload = _convert_functions(kwargs)
                response = await _aclient().chat.completions.create(**payload)
                return _convert_response(response)

        class _Embedding:
            @staticmethod
            def create(**kwargs: Any) -> Dict[str, Any]:
                response = _client().embeddings.create(**kwargs)
                return {
                    "data": [{"embedding": item.embedding} for item in response.data],
                }

            @staticmethod
            async def acreate(**kwargs: Any) -> Dict[str, Any]:
                response = await _aclient().embeddings.create(**kwargs)
                return {
                    "data": [{"embedding": item.embedding} for item in response.data],
                }

        openai.ChatCompletion = _ChatCompletion
        openai.Embedding = _Embedding

    def _install_questionary_shim(self) -> None:
        try:
            import questionary  # type: ignore  # noqa: F401
            return
        except ModuleNotFoundError:
            pass

        class _Prompt:
            def __init__(self, value: Any = None) -> None:
                self._value = value

            def ask(self) -> Any:
                return self._value

            async def ask_async(self) -> Any:
                return self._value

        class _Choice:
            def __init__(self, title: str, value: Any = None) -> None:
                self.title = title
                self.value = title if value is None else value

        shim = SimpleNamespace(
            Choice=_Choice,
            select=lambda *args, **kwargs: _Prompt(kwargs.get("default")),
            confirm=lambda *args, **kwargs: _Prompt(kwargs.get("default", False)),
            text=lambda *args, **kwargs: _Prompt(kwargs.get("default", "")),
            print=lambda *args, **kwargs: None,
        )
        sys.modules["questionary"] = shim

    def _ensure_config(self, config_cls: Any) -> None:
        config = config_cls.load()
        config.model_endpoint = "openai"
        config.archival_storage_type = "local"
        config.openai_key = self.api_key
        config.save()


class MemGPTAgent:
    def __init__(
        self,
        method_config: Dict[str, Any],
        model_config: Dict[str, Any],
        *,
        runtime_home: Path,
    ) -> None:
        provider = str(model_config.get("provider", "")).strip().lower()
        if provider != "openai_api":
            raise ValueError(
                f"Unsupported benchmark model provider for MemGPT: {provider or '<missing>'}. "
                "MemGPT adapter only supports OpenAI-compatible model configs."
            )

        self.model_name = str(model_config.get("model", "")).strip() or "gpt-4"
        self.api_key = _resolve_secret(
            model_config,
            method_config,
            raw_key="api_key",
            env_key="api_key_env",
            default_env="OPENAI_API_KEY",
        )
        if not self.api_key:
            raise ValueError("OpenAI API key not found for MemGPT benchmark adapter.")
        self.base_url = (
            str(model_config.get("base_url", "")).strip()
            or str(method_config.get("base_url", "")).strip()
            or "https://api.openai.com/v1"
        )
        self.runtime_home = runtime_home
        self.modules = _MemGPTModules(runtime_home=runtime_home, api_key=self.api_key, base_url=self.base_url)

        self.persona_name = str(method_config.get("persona", "memgpt_doc")).strip() or "memgpt_doc"
        self.human_name = str(method_config.get("human", "basic")).strip() or "basic"
        self.agent_name = str(method_config.get("agent_name", "memorybench_memgpt")).strip() or "memorybench_memgpt"
        self.base_state_ready = False

    def _build_agent(self) -> Tuple[Any, _CaptureInterface]:
        modules = self.modules
        interface = _CaptureInterface()
        agent_config = modules.AgentConfig(
            name=self.agent_name,
            persona=self.persona_name,
            human=self.human_name,
            model=self.model_name,
            preset=modules.presets_module.DEFAULT_PRESET,
        )
        persistence_manager = modules.LocalStateManager(agent_config)
        agent = modules.presets_module.use_preset(
            agent_config.preset,
            agent_config,
            agent_config.model,
            modules.personas_module.get_persona_text(agent_config.persona),
            modules.humans_module.get_human_text(agent_config.human),
            interface,
            persistence_manager,
        )
        return agent, interface

    def prepare_base_state(self, rows: List[Dict[str, Any]]) -> None:
        agent, _ = self._build_agent()
        print(f"[MemGPT] Preparing base state with {len(rows)} archival memories...", flush=True)
        for idx, row in enumerate(rows, start=1):
            text = str(row.get("text", "")).strip()
            if text:
                agent.persistence_manager.archival_memory.insert(text)
            if idx == 1 or idx % 10 == 0 or idx == len(rows):
                print(f"[MemGPT] Inserted {idx}/{len(rows)} archival memories", flush=True)
        print("[MemGPT] Saving base state checkpoint...", flush=True)
        agent.save()
        print("[MemGPT] Base state checkpoint saved", flush=True)
        self.base_state_ready = True

    def _load_fresh_agent(self) -> Tuple[Any, _CaptureInterface]:
        if not self.base_state_ready:
            raise RuntimeError("MemGPT base state has not been prepared.")
        interface = _CaptureInterface()
        agent_config = self.modules.AgentConfig.load(self.agent_name)
        agent = self.modules.agent_module.AgentAsync.load_agent(interface, agent_config)
        return agent, interface

    async def _run_question_async(self, query: str) -> Dict[str, Any]:
        agent, interface = self._load_fresh_agent()
        system = self.modules.system_module
        constants = self.modules.constants_module
        user_message = system.package_user_message(query)
        started_at = len(agent.messages)
        while True:
            new_messages, heartbeat_request, function_failed, token_warning = await agent.step(
                user_message,
                first_message=False,
                skip_verify=True,
            )
            if token_warning:
                user_message = system.get_token_limit_warning()
                continue
            if function_failed:
                user_message = system.get_heartbeat(constants.FUNC_FAILED_HEARTBEAT_MESSAGE)
                continue
            if heartbeat_request:
                user_message = system.get_heartbeat(constants.REQ_HEARTBEAT_MESSAGE)
                continue
            break

        post_messages = agent.messages[started_at:]
        final_prediction = ""
        for event in reversed(interface.events):
            if event.get("kind") == "assistant_message" and event.get("message", "").strip():
                final_prediction = event["message"].strip()
                break

        return {
            "prediction": final_prediction,
            "events": interface.events,
            "messages": post_messages,
            "all_messages": agent.messages,
        }

    def answer_question(self, query: str) -> Dict[str, Any]:
        import asyncio

        return asyncio.run(self._run_question_async(query))


class _MemGPTBackendMethod(HistoryMethod):
    name = "memgpt"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._agent: Optional[MemGPTAgent] = None
        self._debug_rows: List[Dict[str, Any]] = []

    def _safe_task_name(self, dataset: MemoryBenchmarkDataset) -> str:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        return task_name.lower().replace(" ", "_").replace("/", "_")

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (SCRIPT_DIR / "output" / self._safe_task_name(dataset) / "memgpt").resolve()

    def _runtime_home(self, dataset: MemoryBenchmarkDataset) -> Path:
        return (SCRIPT_DIR / "runs" / self._safe_task_name(dataset) / "memgpt_home").resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _build_archive_rows(self, dataset: MemoryBenchmarkDataset) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            timestamp = str(session.get("date", "")).strip() or ""
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                user_input = _round_user_with_captions(round_payload)
                assistant_response = str(round_payload.get("assistant", "")).strip()
                text = _stored_memory_text(user_input, assistant_response)
                if not text:
                    continue
                rows.append(
                    {
                        "round_id": round_id,
                        "session_id": round_payload.get("session_id", ""),
                        "timestamp": timestamp,
                        "text": text,
                    }
                )
        return rows

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._agent is not None and self._dataset_key == dataset_id:
            return

        self.runtime_info.clear()
        self.runtime_info.update(validate_text_only_captions(dataset.rounds))
        self._debug_rows = []
        model_config = dict(self.config.get("_model_cfg", {}))
        runtime_home = self._runtime_home(dataset)
        runtime_home.mkdir(parents=True, exist_ok=True)

        archive_rows = self._build_archive_rows(dataset)
        agent = MemGPTAgent(self.config, model_config, runtime_home=runtime_home)
        agent.prepare_base_state(archive_rows)

        for row in archive_rows:
            self._debug_rows.append(
                {
                    "type": "stored_memory",
                    "round_id": row["round_id"],
                    "session_id": row["session_id"],
                    "timestamp": row["timestamp"],
                    "text": row["text"],
                }
            )

        self._agent = agent
        self._dataset_key = dataset_id
        self.runtime_info["num_memories"] = len(archive_rows)
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        self.runtime_info["memgpt_home"] = str(runtime_home)
        self.runtime_info["model_name"] = agent.model_name
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._agent is not None
        query = _question_with_image_caption(qa, question)
        result = self._agent.answer_question(query)
        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": query,
                "memgpt_events": result["events"],
                "memgpt_messages": result["messages"],
                "prediction": result["prediction"],
            }
        )
        self._flush_debug(dataset)
        return result["prediction"]

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class _StrictAnswerFormatter:
    def __init__(self, method_config: Dict[str, Any]) -> None:
        model_config = dict(method_config.get("_model_cfg", {}))
        self.provider = str(model_config.get("provider", "")).strip().lower() or "openai_api"
        if self.provider != "openai_api":
            raise ValueError(
                f"Unsupported benchmark model provider for MemGPT final answer formatting: {self.provider}. "
                "Use an OpenAI API model config for the strict final answer step."
            )

        self.model_name = str(model_config.get("model", "")).strip() or str(model_config.get("name", "")).strip()
        if not self.model_name:
            raise ValueError("Missing benchmark model name for MemGPT final answer formatting.")

        self.api_key = _resolve_secret(
            model_config,
            method_config,
            raw_key="api_key",
            env_key="api_key_env",
            default_env="OPENAI_API_KEY",
        )
        if not self.api_key:
            raise ValueError("OpenAI API key not found for MemGPT strict final answer formatting.")

        self.base_url = (
            str(model_config.get("base_url", "")).strip()
            or str(method_config.get("final_answer_base_url", "")).strip()
            or "https://api.openai.com/v1"
        )
        self.timeout = int(model_config.get("timeout", method_config.get("final_answer_timeout", 90)) or 90)
        self.client = _load_openai_client_class()(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    def _mcq_prompt(self, question: str, draft_answer: str, valid_options: List[str]) -> str:
        valid = ", ".join(valid_options)
        return (
            "You are producing the final benchmark answer.\n"
            "Use the benchmark question and the MemGPT draft answer below.\n"
            f"Select exactly one option letter from: {valid}.\n"
            "Return the best final option even if the draft answer is verbose.\n\n"
            f"Benchmark question:\n{question}\n\n"
            f"MemGPT draft answer:\n{draft_answer or '[empty]'}"
        )

    def _open_prompt(self, question: str, draft_answer: str) -> str:
        return (
            "You are producing the final benchmark answer.\n"
            "Use the benchmark question and the MemGPT draft answer below.\n"
            "Return only a short answer phrase with no explanation.\n\n"
            f"Benchmark question:\n{question}\n\n"
            f"MemGPT draft answer:\n{draft_answer or '[empty]'}"
        )

    def format_answer(self, question: str, qa: Dict[str, Any], draft_answer: str) -> Tuple[str, str]:
        options = qa.get("options")
        messages = [{"role": "system", "content": "You must respond with a JSON object."}]

        if options and isinstance(options, dict):
            valid_options = sorted(str(key).strip() for key in options.keys() if str(key).strip())
            user_prompt = self._mcq_prompt(question, draft_answer, valid_options)
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages + [{"role": "user", "content": user_prompt}],
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
            return _load_json_object(raw_response, "answer"), user_prompt

        user_prompt = self._open_prompt(question, draft_answer)
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages + [{"role": "user", "content": user_prompt}],
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
        return _load_json_object(raw_response, "answer"), user_prompt


class MemGPTMethod(HistoryMethod):
    name = "memgpt"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._worker: Optional[subprocess.Popen[str]] = None
        self._worker_log: List[str] = []
        self._answer_formatter = _StrictAnswerFormatter(self.config)

    def _spawn_worker(self, dataset: MemoryBenchmarkDataset) -> None:
        model_config = dict(self.config.get("_model_cfg", {}))
        env_name = str(self.config.get("conda_env", "")).strip()
        if env_name:
            cmd = [
                "conda",
                "run",
                "--no-capture-output",
                "-n",
                env_name,
                "python",
                "-m",
                "benchmark.memgpt",
                "--worker",
            ]
        else:
            cmd = [
                sys.executable,
                "-m",
                "benchmark.memgpt",
                "--worker",
            ]
        self._worker = subprocess.Popen(
            cmd,
            cwd=str(SCRIPT_DIR),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        init_payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "image_root": str(dataset.image_root) if dataset.image_root else "",
            "method_config": self.config,
            "model_config": model_config,
        }
        response = self._send(init_payload)
        self.runtime_info.update(response.get("runtime_info", {}))

    def _readline(self, stream: Any) -> str:
        line = stream.readline()
        if line:
            return line
        stderr = ""
        if self._worker is not None and self._worker.stderr is not None:
            stderr = self._worker.stderr.read().strip()
        raise RuntimeError(f"MemGPT worker exited unexpectedly. {stderr}".strip())

    def _send(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        if self._worker is None or self._worker.stdin is None or self._worker.stdout is None:
            raise RuntimeError("MemGPT worker is not running.")
        self._worker.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
        self._worker.stdin.flush()
        while True:
            line = self._readline(self._worker.stdout).strip()
            if not line:
                continue
            try:
                response = json.loads(line)
                break
            except json.JSONDecodeError:
                self._worker_log.append(line)
                continue
        if response.get("status") == "error":
            raise RuntimeError(str(response.get("error", "Unknown MemGPT worker error")))
        return response

    def _ensure_worker(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._worker is not None and self._dataset_key == dataset_id:
            return
        self.close()
        self._spawn_worker(dataset)
        self._dataset_key = dataset_id

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_worker(dataset)
        response = self._send(
            {
                "question": question,
                "qa": qa,
            }
        )
        if "runtime_info" in response:
            self.runtime_info.update(response["runtime_info"])
        raw_prediction = str(response.get("prediction", "")).strip()
        final_prediction, final_prompt = self._answer_formatter.format_answer(question, qa, raw_prediction)
        self.runtime_info.update(
            {
                "final_answer_formatter": "openai_json_schema",
                "final_answer_model": self._answer_formatter.model_name,
                "last_memgpt_raw_prediction": raw_prediction,
                "last_final_answer_prompt": final_prompt,
            }
        )
        return final_prediction

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []

    def close(self) -> None:
        if self._worker is None:
            return
        try:
            if self._worker.stdin is not None:
                self._worker.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                self._worker.stdin.flush()
        except Exception:
            pass
        try:
            self._worker.terminate()
            self._worker.wait(timeout=5)
        except Exception:
            try:
                self._worker.kill()
            except Exception:
                pass
        self._worker = None

    def __del__(self) -> None:
        self.close()


def _reply(payload: Dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def _worker_main() -> None:
    method: Optional[_MemGPTBackendMethod] = None
    dataset: Optional[MemoryBenchmarkDataset] = None

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            if payload.get("command") == "shutdown":
                _reply({"status": "ok"})
                return

            if method is None:
                dataset_path = Path(payload["dataset_path"])
                image_root_raw = payload.get("image_root") or None
                image_root = Path(image_root_raw) if image_root_raw else None
                dataset = MemoryBenchmarkDataset(dataset_path, image_root)
                method_cfg = dict(payload.get("method_config", {}))
                method_cfg["_model_cfg"] = dict(payload.get("model_config", {}))
                method = _MemGPTBackendMethod(method_cfg)
                method._ensure_initialized(dataset)
                _reply({"status": "ready", "runtime_info": dict(method.runtime_info)})
                continue

            assert dataset is not None
            question = str(payload.get("question", "")).strip()
            qa = payload.get("qa", {}) or {}
            prediction = method.answer(dataset, qa, question)
            _reply(
                {
                    "status": "ok",
                    "prediction": prediction,
                    "runtime_info": dict(method.runtime_info),
                }
            )
        except Exception as exc:
            _reply({"status": "error", "error": str(exc)})
            return


if __name__ == "__main__":
    if "--worker" in sys.argv[1:]:
        _worker_main()
