import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .dataset import MemoryBenchmarkDataset, build_round_retrieval_text


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by",
    "for", "from", "had", "has", "have", "he", "her", "his", "i", "in",
    "into", "is", "it", "its", "me", "my", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "was",
    "we", "were", "what", "when", "where", "which", "who", "why", "with",
    "you", "your",
}

_RETRIEVER_CACHE: Dict[Tuple[Any, ...], "_BaseRetriever"] = {}


def _tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def _normalize_modality(config: Dict[str, Any]) -> str:
    raw = str(config.get("modality", "")).strip().lower()
    if raw in {"text_only", "multimodal"}:
        return raw

    method_name = str(config.get("name", "")).strip().lower()
    if method_name == "semantic_rag_multimodal":
        return "multimodal"
    return "text_only"


def _idf(documents: List[List[str]]) -> Dict[str, float]:
    num_docs = len(documents)
    doc_freq: Counter = Counter()
    for doc in documents:
        for token in set(doc):
            doc_freq[token] += 1
    return {
        token: math.log((1 + num_docs) / (1 + freq)) + 1.0
        for token, freq in doc_freq.items()
    }


def _tfidf_vector(tokens: List[str], idf: Dict[str, float]) -> Dict[str, float]:
    counts = Counter(tokens)
    total = sum(counts.values()) or 1
    return {token: (count / total) * idf.get(token, 0.0) for token, count in counts.items()}


def _cosine_similarity(left: Dict[str, float], right: Dict[str, float]) -> float:
    if not left or not right:
        return 0.0
    dot = sum(value * right.get(token, 0.0) for token, value in left.items())
    left_norm = math.sqrt(sum(v * v for v in left.values()))
    right_norm = math.sqrt(sum(v * v for v in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _dense_cosine(left: List[float], right: List[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(v * v for v in left))
    right_norm = math.sqrt(sum(v * v for v in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _keyword_overlap(query_tokens: List[str], doc_tokens: List[str]) -> float:
    if not query_tokens:
        return 0.0
    return len(set(query_tokens) & set(doc_tokens)) / len(set(query_tokens))


def _normalize_backend(config: Dict[str, Any]) -> str:
    backend = str(config.get("retrieval_backend", "")).strip().lower()
    if backend:
        return backend
    method_name = str(config.get("name", "")).strip().lower()
    if method_name in {"semantic_rag_text_only", "semantic_rag_multimodal"}:
        return "dense_text"
    return "legacy_sparse"


def _normalize_corpus(config: Dict[str, Any]) -> str:
    corpus = str(config.get("retrieval_corpus", "round_text")).strip().lower()
    corpus = corpus or "round_text"
    if corpus != "round_text":
        raise ValueError(
            "Only retrieval_corpus=round_text is supported for the current benchmark methods. "
            f"Unsupported retrieval_corpus: {corpus}"
        )
    return corpus


def _dataset_round_order(dataset: MemoryBenchmarkDataset) -> List[str]:
    ordered: List[str] = []
    for session_id in dataset.session_order():
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id:
                ordered.append(round_id)
    return ordered


def _resolve_notes_path(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Path:
    configured = str(config.get("retrieval_notes_json", "")).strip()
    if configured:
        path = Path(configured)
        if not path.is_absolute():
            path = (dataset.dialog_json_path.parent / path).resolve()
        return path
    return dataset.dialog_json_path.with_name(f"{dataset.dialog_json_path.stem}_notes.json")


def _load_note_texts(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Tuple[Dict[str, str], Path]:
    notes_path = _resolve_notes_path(dataset, config)
    if not notes_path.exists():
        raise FileNotFoundError(
            f"retrieval_corpus=notes requires a notes file at '{notes_path}'."
        )
    with notes_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    raw_entries: List[Dict[str, Any]] = []
    if isinstance(payload, dict):
        notes_value = payload.get("notes", payload.get("entries", []))
        if isinstance(notes_value, dict):
            raw_entries = [
                {"round_id": round_id, "text": text}
                for round_id, text in notes_value.items()
            ]
        elif isinstance(notes_value, list):
            raw_entries = [entry for entry in notes_value if isinstance(entry, dict)]
    elif isinstance(payload, list):
        raw_entries = [entry for entry in payload if isinstance(entry, dict)]

    note_text_by_round: Dict[str, str] = {}
    for entry in raw_entries:
        round_id = str(entry.get("round_id", "")).strip()
        text = str(entry.get("text", "")).strip()
        if not round_id or not text or round_id not in dataset.rounds:
            continue
        note_text_by_round[round_id] = text
    return note_text_by_round, notes_path


def _build_corpus_rows(
    dataset: MemoryBenchmarkDataset,
    config: Dict[str, Any],
) -> Tuple[List[Tuple[str, str]], Dict[str, Any]]:
    corpus = _normalize_corpus(config)
    modality = _normalize_modality(config)

    rows: List[Tuple[str, str]] = []
    for round_id in _dataset_round_order(dataset):
        round_payload = dataset.rounds.get(round_id, {})
        text = build_round_retrieval_text(round_payload, modality=modality)
        if text:
            rows.append((round_id, text))
    return rows, {
        "retrieval_corpus": "round_text",
        "corpus_entry_count": len(rows),
        "modality": modality,
    }


def _expand_with_neighbors(
    dataset: MemoryBenchmarkDataset,
    seed_round_ids: List[str],
    session_ids: List[str],
    window: int,
) -> List[str]:
    if window <= 0:
        return seed_round_ids

    selected = set(seed_round_ids)
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        ordered_round_ids = [
            dialogue.get("round", "") for dialogue in session.get("dialogues", [])
        ]
        index_by_round_id = {rid: idx for idx, rid in enumerate(ordered_round_ids)}
        for round_id in list(seed_round_ids):
            if round_id not in index_by_round_id:
                continue
            idx = index_by_round_id[round_id]
            start = max(0, idx - window)
            end = min(len(ordered_round_ids), idx + window + 1)
            selected.update(ordered_round_ids[start:end])

    ordered: List[str] = []
    for session_id in session_ids:
        session = dataset.get_session(session_id)
        for dialogue in session.get("dialogues", []):
            round_id = dialogue.get("round", "")
            if round_id in selected:
                ordered.append(round_id)
    return ordered


class _BaseRetriever:
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        self.dataset = dataset
        self.config = config
        self.top_k = int(config.get("top_k", 10))
        self.neighbor_window = int(config.get("neighbor_window", 1))
        self.session_ids = dataset.session_order()
        self.corpus_rows, self.corpus_meta = _build_corpus_rows(dataset, config)

    def _build_debug_info(
        self,
        qa: Dict[str, Any],
        seed_round_ids: List[str],
        selected_round_ids: List[str],
        top_candidates: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        clue_rounds = list(qa.get("clue", []) or [])
        debug: Dict[str, Any] = {
            "retrieval_backend": self.config.get("retrieval_backend", "legacy_sparse"),
            "retrieval_corpus": self.corpus_meta.get("retrieval_corpus", "round_text"),
            "method_modality": self.corpus_meta.get("modality", _normalize_modality(self.config)),
            "top_k": self.top_k,
            "neighbor_window": self.neighbor_window,
            "seed_round_ids": seed_round_ids,
            "selected_round_ids": selected_round_ids,
            "top_candidates": top_candidates,
            "clue_hit_count": sum(1 for rid in selected_round_ids if rid in clue_rounds),
            "corpus_entry_count": self.corpus_meta.get("corpus_entry_count", 0),
        }
        notes_path = self.corpus_meta.get("retrieval_notes_json")
        if notes_path:
            debug["retrieval_notes_json"] = notes_path
        return debug

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        raise NotImplementedError


class _SparseRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        super().__init__(dataset, config)
        self.lexical_weight = float(config.get("lexical_weight", 0.35))
        self.semantic_weight = float(config.get("semantic_weight", 0.65))
        self.candidate_rows: List[Tuple[str, List[str]]] = []
        for round_id, text in self.corpus_rows:
            tokens = _tokenize(text)
            if tokens:
                self.candidate_rows.append((round_id, tokens))

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        query_text = str(qa.get("question", "")).strip()
        query_tokens = _tokenize(query_text)
        if not query_tokens or not self.candidate_rows:
            return [], self._build_debug_info(qa, [], [], [])

        documents = [tokens for _, tokens in self.candidate_rows]
        idf = _idf(documents)
        query_vector = _tfidf_vector(query_tokens, idf)

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for round_id, tokens in self.candidate_rows:
            doc_vector = _tfidf_vector(tokens, idf)
            lexical_score = _keyword_overlap(query_tokens, tokens)
            semantic_score = _cosine_similarity(query_vector, doc_vector)
            score = (self.lexical_weight * lexical_score) + (self.semantic_weight * semantic_score)
            if score <= 0:
                continue
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "lexical_score": lexical_score,
                        "semantic_score": semantic_score,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        if not scored:
            return [], self._build_debug_info(qa, [], [], [])

        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        return selected_round_ids, self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[:5]],
        )


class _DenseTextRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        super().__init__(dataset, config)
        from .embeddings import TextEmbedder

        self.text_embedding_model = str(config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
        self.text_embedder = TextEmbedder(self.text_embedding_model)
        self.round_texts: List[Tuple[str, str]] = list(self.corpus_rows)
        texts = [text for _, text in self.round_texts]
        self.round_vectors = self.text_embedder.embed_batch(texts) if texts else []

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        query_text = str(qa.get("question", "")).strip()
        if not query_text or not self.round_texts:
            return [], self._build_debug_info(qa, [], [], [])

        query_vec = self.text_embedder.embed_query(query_text)
        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        for (round_id, _), round_vec in zip(self.round_texts, self.round_vectors):
            score = _dense_cosine(query_vec, round_vec)
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "text_dense_score": score,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        debug = self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[:5]],
        )
        debug["text_embedding_model"] = self.text_embedding_model
        debug["caption_text_included"] = self.corpus_meta.get("modality") == "text_only"
        debug["image_embeddings_built"] = False
        return selected_round_ids, debug


class _DenseMultimodalRetriever(_BaseRetriever):
    def __init__(self, dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> None:
        super().__init__(dataset, config)
        from .embeddings import TextEmbedder, get_multimodal_embedder

        self.text_embedding_model = str(config.get("text_embedding_model", TextEmbedder.DEFAULT_MODEL))
        self.text_dense_weight = float(config.get("text_dense_weight", 1.0))
        self.image_dense_weight = float(config.get("image_dense_weight", 0.0))
        self.text_embedder = TextEmbedder(self.text_embedding_model)
        self.mm_model = str(config.get("multimodal_embedding_model", "siglip2-base-patch16-384"))
        self.mm_embedder = get_multimodal_embedder(self.mm_model)
        if self.mm_embedder is None:
            raise RuntimeError(
                "semantic_rag dense_multimodal requires a working multimodal embedder; "
                "neither vLLM SigLIP2 nor local CLIP is available."
            )

        self.round_rows: List[Tuple[str, Optional[List[float]], List[Tuple[str, List[float]]]]] = []
        text_batch_round_ids: List[str] = []
        text_batch_texts: List[str] = []
        image_jobs: List[Tuple[str, str]] = []
        for round_id, corpus_text in self.corpus_rows:
            round_payload = dataset.rounds.get(round_id, {})
            images = list(round_payload.get("images", []) or [])
            if not corpus_text and not images:
                continue
            self.round_rows.append((round_id, None, []))
            if corpus_text:
                text_batch_round_ids.append(round_id)
                text_batch_texts.append(corpus_text)
            if images:
                for image_path in images:
                    image_jobs.append((round_id, image_path))

        text_vectors_by_round: Dict[str, List[float]] = {}
        if text_batch_texts:
            for round_id, vec in zip(text_batch_round_ids, self.text_embedder.embed_batch(text_batch_texts)):
                text_vectors_by_round[round_id] = vec

        image_vectors_by_round: Dict[str, List[Tuple[str, List[float]]]] = {}
        for round_id, image_path in image_jobs:
            image_vectors_by_round.setdefault(round_id, []).append(
                (image_path, self.mm_embedder.embed_image(image_path))
            )

        self.round_rows = [
            (round_id, text_vectors_by_round.get(round_id), image_vectors_by_round.get(round_id, []))
            for round_id, _, _ in self.round_rows
        ]

    def select(self, qa: Dict[str, Any]) -> Tuple[List[str], Dict[str, Any]]:
        query_text = str(qa.get("question", "")).strip()
        if not query_text or not self.round_rows:
            return [], self._build_debug_info(qa, [], [], [])

        text_query_vec: Optional[List[float]] = None
        image_query_vec: Optional[List[float]] = None
        if self.text_dense_weight > 0:
            text_query_vec = self.text_embedder.embed_query(query_text)
        if self.image_dense_weight > 0:
            image_query_vec = self.mm_embedder.embed_text(query_text)

        scored: List[Tuple[float, str, Dict[str, Any]]] = []
        total_indexed_images = 0
        for round_id, text_vec, image_items in self.round_rows:
            text_score = _dense_cosine(text_query_vec or [], text_vec or [])
            total_indexed_images += len(image_items)
            best_image_path = ""
            image_score = 0.0
            for image_path, image_vec in image_items:
                score = _dense_cosine(image_query_vec or [], image_vec or [])
                if score >= image_score:
                    image_score = score
                    best_image_path = image_path
            score = (self.text_dense_weight * text_score) + (self.image_dense_weight * image_score)
            scored.append(
                (
                    score,
                    round_id,
                    {
                        "round_id": round_id,
                        "score": score,
                        "text_dense_score": text_score,
                        "image_dense_score": image_score,
                        "indexed_image_count": len(image_items),
                        "best_image_path": best_image_path,
                    },
                )
            )
        scored.sort(key=lambda item: (-item[0], item[1]))
        seed_round_ids = [round_id for _, round_id, _ in scored[: max(1, self.top_k)]]
        selected_round_ids = _expand_with_neighbors(
            self.dataset, seed_round_ids, self.session_ids, self.neighbor_window
        )
        debug = self._build_debug_info(
            qa,
            seed_round_ids,
            selected_round_ids,
            [row for _, _, row in scored[:5]],
        )
        debug["text_embedding_model"] = self.text_embedding_model
        debug["multimodal_embedding_model"] = self.mm_model
        debug["text_dense_weight"] = self.text_dense_weight
        debug["image_dense_weight"] = self.image_dense_weight
        debug["caption_text_included"] = False
        debug["image_embeddings_built"] = total_indexed_images > 0
        debug["multi_image_indexing_enabled"] = True
        debug["indexed_image_count"] = total_indexed_images
        return selected_round_ids, debug


def _cache_key(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> Tuple[Any, ...]:
    backend = _normalize_backend(config)
    corpus = _normalize_corpus(config)
    relevant_keys = [
        "name",
        "top_k",
        "neighbor_window",
        "lexical_weight",
        "semantic_weight",
        "text_embedding_model",
        "multimodal_embedding_model",
        "text_dense_weight",
        "image_dense_weight",
        "retrieval_backend",
        "retrieval_corpus",
        "retrieval_notes_json",
    ]
    dialog_mtime = dataset.dialog_json_path.stat().st_mtime_ns if dataset.dialog_json_path.exists() else None
    notes_path = None
    notes_mtime = None
    dataset_key = (
        str(dataset.dialog_json_path),
        str(dataset.image_root),
        dialog_mtime,
        str(notes_path) if notes_path is not None else "",
        notes_mtime,
    )
    return (dataset_key, backend, corpus) + tuple((key, config.get(key)) for key in relevant_keys)


def _get_retriever(dataset: MemoryBenchmarkDataset, config: Dict[str, Any]) -> _BaseRetriever:
    key = _cache_key(dataset, config)
    retriever = _RETRIEVER_CACHE.get(key)
    if retriever is not None:
        return retriever

    backend = _normalize_backend(config)
    if backend == "dense_text":
        retriever = _DenseTextRetriever(dataset, config)
    elif backend == "dense_multimodal":
        retriever = _DenseMultimodalRetriever(dataset, config)
    else:
        retriever = _SparseRetriever(dataset, config)
    _RETRIEVER_CACHE[key] = retriever
    return retriever


def clear_retriever_cache() -> None:
    """Release all cached retrievers and their embedding vectors."""
    _RETRIEVER_CACHE.clear()


def select_round_ids_for_qa(
    dataset: MemoryBenchmarkDataset,
    qa: Dict[str, Any],
    config: Dict[str, Any],
    runtime_info: Optional[Dict[str, Any]] = None,
) -> List[str]:
    retriever = _get_retriever(dataset, config)
    selected_round_ids, debug = retriever.select(qa)
    if runtime_info is not None:
        runtime_info.clear()
        runtime_info.update(debug)
    return selected_round_ids
