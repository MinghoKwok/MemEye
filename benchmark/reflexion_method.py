import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import REPO_ROOT, write_json
from .dataset import MemoryBenchmarkDataset
from .methods import HistoryMethod
from .runner import instantiate_router, load_sys_prompt
from .reflexion.memengine.config.Config import MemoryConfig
from .reflexion.memengine.memory.RFMemory import RFMemory


def _resolve_api_key(raw: str, env_name: str) -> Optional[str]:
    value = str(raw).strip()
    if value:
        return value
    env = str(env_name).strip() or "OPENAI_API_KEY"
    return os.getenv(env)


def _dialogue_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    parts: List[str] = []
    user_text = str(round_payload.get("user", "")).strip()
    assistant_text = str(round_payload.get("assistant", "")).strip()
    if user_text:
        parts.append(f"{speaker_a}: {user_text}")
    if assistant_text:
        parts.append(f"{speaker_b}: {assistant_text}")
    return "\n".join(parts).strip()


def _round_captions(round_payload: Dict[str, Any]) -> List[str]:
    raw = round_payload.get("raw", {}) or {}
    captions = raw.get("image_caption", []) or []
    return [str(caption).strip() for caption in captions if str(caption).strip()]


def build_reflexion_note_text(round_payload: Dict[str, Any], speaker_a: str, speaker_b: str) -> str:
    text = _dialogue_text(round_payload, speaker_a, speaker_b)
    lines: List[str] = [text] if text else []
    for caption in _round_captions(round_payload):
        lines.extend(
            [
                "image:",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def _question_with_image_caption(qa: Dict[str, Any], question: str) -> str:
    query = question.strip()
    question_image = qa.get("question_image") or qa.get("question_images")
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


def _build_answer_history(recalled_context: str) -> List[Dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "text": f"Retrieved memory context:\n{recalled_context}".strip(),
            "images": [],
        }
    ]


def _default_reflection_example() -> str:
    return (
        "Question: Which mug matched the earlier desk mug?\n"
        "Retrieved context: The desk mug was white with a thin blue rim. "
        "The meeting-room mug had no blue rim.\n"
        "Model answer: The meeting-room mug matched.\n"
        "Ground truth: They did not match because the meeting-room mug lacked the blue rim.\n"
        "Correct: no\n"
        "Updated insight: Distinguishing visual details like rim color should outweigh looser object matches."
    )


def _build_rfmemory_config(
    method_config: Dict[str, Any],
    *,
    llm_model: str,
    api_key: str,
    base_url: str,
) -> Dict[str, Any]:
    truncation_words = max(1, int(method_config.get("context_word_limit", 6000)))
    temperature = float(method_config.get("reflection_temperature", 0.0))
    example = str(method_config.get("reflection_example", "")).strip() or _default_reflection_example()
    return {
        "global_config": {
            "usable_gpu": "",
        },
        "storage": {},
        "store": {},
        "recall": {
            "empty_memory": str(method_config.get("empty_memory", "No stored memory yet.")),
            "truncation": {
                "method": "LMTruncation",
                "mode": "word",
                "number": truncation_words,
                "path": "",
            },
            "utilization": {
                "method": "ConcateUtilization",
                "prefix": "",
                "suffix": "",
                "list_config": {
                    "index": True,
                    "sep": "\n\n",
                },
                "dict_config": {
                    "key_format": "[%s]",
                    "key_value_sep": "\n",
                    "item_sep": "\n\n",
                },
            },
        },
        "optimize": {
            "reflector": {
                "example": example,
                "LLM_config": {
                    "method": "APILLM",
                    "name": llm_model,
                    "api_key": api_key,
                    "base_url": base_url,
                    "temperature": temperature,
                },
                "prompt": {
                    "input_variables": ["previous_insight", "new_trial", "example"],
                    "template": (
                        "You are updating a compact memory insight for a benchmark agent.\n"
                        "Keep the insight concise, factual, and useful for future QA.\n\n"
                        "Example style:\n{example}\n\n"
                        "Previous insight:\n{previous_insight}\n\n"
                        "New trial:\n{new_trial}\n\n"
                        "Write the updated global insight only."
                    ),
                },
            },
        },
        "display": {
            "method": "ScreenDisplay",
            "prefix": "===== Reflexion %s =====",
            "suffix": "",
            "key_format": "%s",
            "key_value_sep": "\n",
            "item_sep": "\n\n",
        },
    }


class ReflexionMethod(HistoryMethod):
    name = "reflexion"
    fixed_modality = "text_only"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config=config)
        self._dataset_key: Optional[int] = None
        self._memory: Optional[RFMemory] = None
        self._speaker_a: str = "user"
        self._speaker_b: str = "assistant"
        self._routers: Dict[str, Any] = {}
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
                continue
            if any(not str(caption).strip() for caption in captions):
                missing_rounds.append(round_id)
        if missing_rounds:
            raise ValueError(
                "Reflexion text-only adaptation requires dataset image_caption fields. "
                "Missing/invalid image_caption for rounds: "
                + ", ".join(missing_rounds[:10])
                + ("..." if len(missing_rounds) > 10 else "")
            )

    def _debug_dir(self, dataset: MemoryBenchmarkDataset) -> Path:
        task_name = str(dataset.data.get("task_name", "")).strip() or dataset.dialog_json_path.stem
        safe_task = task_name.lower().replace(" ", "_").replace("/", "_")
        return (REPO_ROOT / "output" / safe_task / "reflexion").resolve()

    def _flush_debug(self, dataset: MemoryBenchmarkDataset) -> None:
        if not self._debug_rows:
            return
        payload = {
            "dataset_path": str(dataset.dialog_json_path),
            "rows": self._debug_rows,
        }
        write_json(self._debug_dir(dataset) / "debug_trace.json", payload)

    def _resolve_reflection_llm(self) -> Dict[str, str]:
        model_cfg = dict(self.config.get("_model_cfg", {}))
        llm_model = str(self.config.get("llm_model", "")).strip()
        if llm_model:
            api_key = _resolve_api_key(
                str(self.config.get("llm_api_key", "")),
                str(self.config.get("llm_api_key_env", "OPENAI_API_KEY")),
            )
            if not api_key:
                raise ValueError("Reflexion reflection LLM API key not found.")
            base_url = str(self.config.get("llm_base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1"
            return {
                "model": llm_model,
                "api_key": api_key,
                "base_url": base_url,
            }

        provider = str(model_cfg.get("provider", "")).strip().lower()
        if provider != "openai_api":
            raise ValueError(
                "Reflexion reflection currently requires an OpenAI-compatible API model. "
                "Set llm_model / llm_api_key / llm_base_url in config/methods/reflexion.yaml "
                "or run with an openai_api benchmark model."
            )
        api_key = _resolve_api_key(
            str(model_cfg.get("api_key", "")),
            str(model_cfg.get("api_key_env", "OPENAI_API_KEY")),
        )
        if not api_key:
            raise ValueError("OpenAI-compatible API key not found for Reflexion reflection.")
        return {
            "model": str(model_cfg.get("model", "")).strip(),
            "api_key": api_key,
            "base_url": str(model_cfg.get("base_url", "https://api.openai.com/v1")).strip() or "https://api.openai.com/v1",
        }

    def _get_answer_router(self, qa: Dict[str, Any]) -> Any:
        mode = "mcq" if isinstance(qa.get("options"), dict) else "open"
        router = self._routers.get(mode)
        if router is not None:
            return router
        method_cfg = dict(self.config)
        method_cfg["modality"] = "text_only"
        router = instantiate_router(
            dict(self.config.get("_model_cfg", {})),
            system_prompt=load_sys_prompt(mode, method_cfg),
        )
        self._routers[mode] = router
        return router

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._memory is not None and self._dataset_key == dataset_id:
            return

        self._ensure_caption_preprocessed(dataset)
        self._debug_rows = []
        self._routers = {}

        llm_cfg = self._resolve_reflection_llm()
        memory_config = MemoryConfig(
            _build_rfmemory_config(
                self.config,
                llm_model=llm_cfg["model"],
                api_key=llm_cfg["api_key"],
                base_url=llm_cfg["base_url"],
            )
        )
        self._memory = RFMemory(memory_config)

        character_profile = dataset.data.get("character_profile", {}) or {}
        speaker_name = str(character_profile.get("name", "")).strip()
        self._speaker_a = f"user ({speaker_name})" if speaker_name else "user"
        self._speaker_b = "assistant"

        stored_count = 0
        for session_id in dataset.session_order():
            session = dataset.get_session(session_id)
            timestamp = str(session.get("date", "")).strip() or None
            for dialogue in session.get("dialogues", []):
                round_id = str(dialogue.get("round", "")).strip()
                if not round_id or round_id not in dataset.rounds:
                    continue
                round_payload = dataset.rounds[round_id]
                note_text = build_reflexion_note_text(round_payload, self._speaker_a, self._speaker_b)
                if not note_text:
                    continue
                observation = {
                    "text": note_text,
                    "timestamp": timestamp or "",
                    "dialogue_id": round_id,
                    "session_id": session_id,
                }
                self._memory.store(observation)
                stored_count += 1
                self._debug_rows.append(
                    {
                        "type": "stored_memory",
                        "round_id": round_id,
                        "session_id": session_id,
                        "timestamp": timestamp or "",
                        "text": note_text,
                    }
                )

        self._dataset_key = dataset_id
        self.runtime_info["num_memories"] = stored_count
        self.runtime_info["debug_dir"] = str(self._debug_dir(dataset))
        self._flush_debug(dataset)

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._memory is not None

        recall_query = _question_with_image_caption(qa, question)
        recalled_context = str(self._memory.recall({"text": recall_query})).strip()
        router = self._get_answer_router(qa)
        answer_history = _build_answer_history(recalled_context)
        prediction = router.answer(answer_history, recall_query, question_images=[])

        gt = str(qa.get("answer", "")).strip()
        if isinstance(qa.get("options"), dict):
            is_correct = prediction.strip().upper() == gt.upper()
        else:
            is_correct = prediction.strip().lower() == gt.lower()
        prior_insight = str(self._memory.insight.get("global_insight", "")).strip()
        new_trial = (
            f"Question:\n{question}\n\n"
            f"Retrieved context:\n{recalled_context}\n\n"
            f"Model answer:\n{prediction}\n\n"
            f"Ground truth:\n{gt}\n\n"
            f"Correct: {'yes' if is_correct else 'no'}"
        )
        self._memory.optimize(new_trial=new_trial)
        updated_insight = str(self._memory.insight.get("global_insight", "")).strip()

        self._debug_rows.append(
            {
                "type": "qa",
                "question_id": qa.get("question_id", ""),
                "question": question,
                "recall_query": recall_query,
                "retrieved_context": recalled_context,
                "prediction": prediction,
                "ground_truth": gt,
                "correct": is_correct,
                "insight_before": prior_insight,
                "trial_text": new_trial,
                "insight_after": updated_insight,
            }
        )
        self._flush_debug(dataset)
        return prediction

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []
