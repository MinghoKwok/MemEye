from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from .dataset import MemoryBenchmarkDataset, history_from_round_ids, validate_text_only_captions
from .retrieval import select_round_ids_for_qa


# ---------------------------------------------------------------------------
# Token estimation helpers for context-window truncation
# ---------------------------------------------------------------------------

# Rough chars-per-token ratio (conservative for English + mixed content)
_CHARS_PER_TOKEN = 4

# Estimated token cost per image, following Mem-Gallery (predefined token cost)
_IMAGE_TOKEN_COST = 765


def _normalize_modality(config: Dict[str, Any], method_name: str) -> str:
    raw = str(config.get("modality", "")).strip().lower()
    if raw in {"text_only", "multimodal", "no_visual"}:
        return raw
    if method_name in {"semantic_rag_multimodal", "full_context_multimodal"}:
        return "multimodal"
    if method_name in {"full_context_text_only", "semantic_rag_text_only"}:
        return "text_only"
    return "text_only"


def _estimate_turn_tokens(turn: Dict[str, Any]) -> int:
    """Estimate token count for a single history turn (text + images)."""
    text = str(turn.get("text", ""))
    text_tokens = max(1, len(text) // _CHARS_PER_TOKEN)
    image_tokens = len(turn.get("images", []) or []) * _IMAGE_TOKEN_COST
    return text_tokens + image_tokens


def _truncate_history(history: List[Dict[str, Any]], max_tokens: int) -> List[Dict[str, Any]]:
    """Truncate history from the front (keep most recent turns) to fit within max_tokens.

    This follows the standard Full Memory approach in Mem-Gallery:
    include all memory and truncate according to the context token limit.
    Truncation removes the oldest turns first (FIFO-style).
    """
    if max_tokens <= 0:
        return history

    # Walk backwards (most recent first) and accumulate token budget
    cumulative = 0
    cutoff_idx = len(history)
    for i in range(len(history) - 1, -1, -1):
        cumulative += _estimate_turn_tokens(history[i])
        if cumulative > max_tokens:
            cutoff_idx = i + 1
            break
    else:
        # Everything fits
        return history

    return history[cutoff_idx:]


class HistoryMethod(ABC):
    name = "base"
    fixed_modality: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        self.config = config or {}
        self.runtime_info: Dict[str, Any] = {}
        self.modality = self.fixed_modality or _normalize_modality(self.config, self.name)

    @abstractmethod
    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        raise NotImplementedError


class _MemGalleryHistoryMethod(HistoryMethod):
    history_source = "history"

    def _validate_modality_inputs(self, dataset: MemoryBenchmarkDataset) -> None:
        if self.modality != "text_only":
            return
        self.runtime_info.update(validate_text_only_captions(dataset.rounds))

    def _update_history_runtime(
        self,
        history: List[Dict[str, Any]],
        *,
        history_before_truncation: Optional[int] = None,
    ) -> None:
        existing = dict(self.runtime_info)
        self.runtime_info.clear()
        self.runtime_info.update(existing)
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": self.history_source,
                "captions_loaded": self.modality == "text_only",
                "images_loaded": self.modality == "multimodal",
                "history_turns_after_truncation": len(history),
            }
        )
        if history_before_truncation is not None:
            self.runtime_info["history_turns_before_truncation"] = history_before_truncation


class _MemGalleryFullContextMethod(_MemGalleryHistoryMethod):
    history_source = "full_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Mem-Gallery-style full-context memory, split into text and multimodal variants."""
        self._validate_modality_inputs(dataset)
        history: List[Dict[str, Any]] = []
        for sid in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    dataset.get_session(sid),
                    dataset.rounds,
                    modality=self.modality,
                )
            )

        max_tokens = int(self.config.get("context_token_limit", 128_000))
        # Reserve tokens for system prompt (~500) + question (~200) + answer generation
        reserved = int(self.config.get("reserved_tokens", 1_000))
        truncated = _truncate_history(history, max_tokens - reserved)
        self._update_history_runtime(truncated, history_before_truncation=len(history))
        return truncated


class FullContextTextMethod(_MemGalleryFullContextMethod):
    """Local equivalent of Mem-Gallery FUMemory (text + captions, no images)."""

    name = "full_context_text_only"
    fixed_modality = "text_only"


class FullContextMultimodalMethod(_MemGalleryFullContextMethod):
    """Local equivalent of Mem-Gallery MMFUMemory."""

    name = "full_context_multimodal"
    fixed_modality = "multimodal"


class FullContextNoVisualMethod(_MemGalleryFullContextMethod):
    """Ablation: full dialogue text, no images, no captions. Tests context leakage."""

    name = "full_context_no_visual"
    fixed_modality = "no_visual"


class QuestionOnlyMethod(HistoryMethod):
    """Ablation: zero history, question + options only. Tests MCQ guessability."""

    name = "question_only"
    fixed_modality = "no_visual"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class TargetSessionContextMethod(_MemGalleryHistoryMethod):
    name = "target_session_context"
    history_source = "target_session_context"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_modality_inputs(dataset)
        history: List[Dict[str, Any]] = []
        target_sessions = set(qa.get("session_id", []))
        for sid in dataset.session_order():
            if sid not in target_sessions:
                continue
            history.extend(
                history_from_round_ids(
                    dataset.get_session(sid),
                    dataset.rounds,
                    modality=self.modality,
                )
            )
        self._update_history_runtime(history)
        return history


class ClueOnlyContextMethod(_MemGalleryHistoryMethod):
    """Oracle retrieval: only include the exact rounds listed in QA clue field."""

    name = "clue_only_context"
    history_source = "clue_only"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_modality_inputs(dataset)
        clue_round_ids = set()
        for clue in qa.get("clue", []):
            # clue format: "S1:4" or "D28:1" — the round_id is the full string
            clue_round_ids.add(clue)
        history: List[Dict[str, Any]] = []
        for sid in dataset.session_order():
            session = dataset.get_session(sid)
            for dialogue in session.get("dialogues", []):
                rid = dialogue.get("round", "")
                if rid in clue_round_ids:
                    round_payload = dataset.rounds.get(rid, {})
                    if round_payload:
                        history.extend(
                            history_from_round_ids(
                                {"dialogues": [dialogue]},
                                dataset.rounds,
                                modality=self.modality,
                            )
                        )
        self._update_history_runtime(history)
        return history


class _RetrievalHistoryMethod(_MemGalleryHistoryMethod):
    history_source = "retrieval"

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        self._validate_modality_inputs(dataset)
        selected_round_ids = select_round_ids_for_qa(dataset, qa, self.config, runtime_info=self.runtime_info)
        if not selected_round_ids:
            return []

        history: List[Dict[str, Any]] = []
        allowed_round_ids = set(selected_round_ids)
        for sid in dataset.session_order():
            history.extend(
                history_from_round_ids(
                    dataset.get_session(sid),
                    dataset.rounds,
                    allowed_round_ids,
                    modality=self.modality,
                )
            )
        self.runtime_info.update(
            {
                "method_modality": self.modality,
                "history_source": self.history_source,
                "captions_loaded": self.modality == "text_only",
                "images_loaded": self.modality == "multimodal",
                "history_turns_after_truncation": len(history),
            }
        )
        return history


class SemanticRAGTextMethod(_RetrievalHistoryMethod):
    name = "semantic_rag_text_only"
    fixed_modality = "text_only"


class SemanticRAGMultimodalMethod(_RetrievalHistoryMethod):
    name = "semantic_rag_multimodal"
    fixed_modality = "multimodal"



class M2AAgentMethod(HistoryMethod):
    name = "m2a"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return

        from .m2a import M2ASystem

        self._system = M2ASystem(self.config)
        sessions = dataset.session_order()
        print(f"[M2A] Building memory from {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id
        print(f"[M2A] Memory ready: {self._system.num_memories} semantic memories stored.")

    def answer(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any], question: str) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        return self._system.answer_question(question)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


class MMAAgentMethod(HistoryMethod):
    """Confidence-aware multimodal memory agent (adapted from MMA)."""

    name = "mma"

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        self._system: Optional[Any] = None
        self._dataset_key: Optional[int] = None

    def _ensure_initialized(self, dataset: MemoryBenchmarkDataset) -> None:
        dataset_id = id(dataset)
        if self._system is not None and self._dataset_key == dataset_id:
            return

        from .mma import MMASystem

        self._system = MMASystem(self.config)
        sessions = dataset.session_order()
        print(f"[MMA] Building memory from {len(sessions)} session(s)...")
        self._system.process_all_sessions(dataset)
        self._dataset_key = dataset_id

    def answer(
        self,
        dataset: MemoryBenchmarkDataset,
        qa: Dict[str, Any],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        self._ensure_initialized(dataset)
        assert self._system is not None
        qa_images = question_images
        if qa_images is None:
            qa_images = qa.get("question_images") or qa.get("question_image") or None
        return self._system.answer_question(question, image_paths=qa_images)

    def build_history(self, dataset: MemoryBenchmarkDataset, qa: Dict[str, Any]) -> List[Dict[str, Any]]:
        return []


def get_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    config = config or {}
    registry = {
        FullContextTextMethod.name: FullContextTextMethod,
        FullContextMultimodalMethod.name: FullContextMultimodalMethod,
        FullContextNoVisualMethod.name: FullContextNoVisualMethod,
        QuestionOnlyMethod.name: QuestionOnlyMethod,
        TargetSessionContextMethod.name: TargetSessionContextMethod,
        ClueOnlyContextMethod.name: ClueOnlyContextMethod,
        SemanticRAGTextMethod.name: SemanticRAGTextMethod,
        SemanticRAGMultimodalMethod.name: SemanticRAGMultimodalMethod,
        M2AAgentMethod.name: M2AAgentMethod,
        MMAAgentMethod.name: MMAAgentMethod,
    }
    cls = registry.get(method_name)
    if cls is None:
        if method_name == "a_mem":
            from .a_mem import AMemMethod

            return AMemMethod(config=config)
        if method_name == "memgpt":
            from .memgpt import MemGPTMethod

            return MemGPTMethod(config=config)
        if method_name == "gen_agents":
            from .gen_agents import GAMethod

            return GAMethod(config=config)
        if method_name == "evermemos":
            from .evermemos import EverMemOSMethod

            return EverMemOSMethod(config=config)
        if method_name == "reflexion":
            from .reflexion_method import ReflexionMethod

            return ReflexionMethod(config=config)
        if method_name == "simplemem":
            from .simplemem import SimpleMemMethod

            return SimpleMemMethod(config=config)
        if method_name == "memoryos":
            from .memoryos import MemoryOSMethod

            return MemoryOSMethod(config=config)
        from .mirix import get_mirix_method

        return get_mirix_method(method_name, config=config)
    return cls(config=config)
