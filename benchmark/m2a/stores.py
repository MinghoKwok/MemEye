"""
In-memory equivalents of RawMessageStore (SQLite) and SemanticStore (Milvus).
Faithful to official M2A architecture; replaces Milvus with in-memory numpy +
custom BM25 to avoid heavy infra dependency while preserving identical algorithms.
"""
from __future__ import annotations

import json
import math
import re
import sqlite3
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# RawMessageStore  (faithful to official agent/stores/raw.py)
# ---------------------------------------------------------------------------

@dataclass
class RawMessage:
    msg_id: int
    timestamp: str
    role: str
    text: str
    image_path: Optional[str] = None


class RawMessageStore:
    """
    Append-only SQLite message store. Messages are never modified or deleted.
    Faithful to official M2A RawMessageStore schema and interface.
    """

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                msg_id     INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp  TEXT,
                role       TEXT,
                text       TEXT,
                image_path TEXT
            )
            """
        )
        self._conn.commit()

    def append(
        self,
        timestamp: str,
        role: str,
        text: str,
        image_path: Optional[str] = None,
    ) -> int:
        cur = self._conn.execute(
            "INSERT INTO messages (timestamp, role, text, image_path) VALUES (?, ?, ?, ?)",
            (timestamp, role, text, image_path or ""),
        )
        self._conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def fetch_by_ids(self, id_ranges: List[List[int]]) -> List[RawMessage]:
        """Fetch messages by [[start, end], ...] ID ranges."""
        messages: List[RawMessage] = []
        for start, end in id_ranges:
            rows = self._conn.execute(
                "SELECT msg_id, timestamp, role, text, image_path "
                "FROM messages WHERE msg_id BETWEEN ? AND ? ORDER BY msg_id",
                (start, end),
            ).fetchall()
            for row in rows:
                messages.append(
                    RawMessage(
                        msg_id=row[0],
                        timestamp=row[1],
                        role=row[2],
                        text=row[3],
                        image_path=row[4] or None,
                    )
                )
        return messages

    def fetch_by_timerange(self, start_date: str, end_date: str) -> List[RawMessage]:
        rows = self._conn.execute(
            "SELECT msg_id, timestamp, role, text, image_path "
            "FROM messages WHERE timestamp >= ? AND timestamp <= ? ORDER BY msg_id",
            (start_date, end_date),
        ).fetchall()
        return [
            RawMessage(msg_id=r[0], timestamp=r[1], role=r[2], text=r[3], image_path=r[4] or None)
            for r in rows
        ]

    def get_latest_n(self, n: int) -> List[RawMessage]:
        rows = self._conn.execute(
            "SELECT msg_id, timestamp, role, text, image_path "
            "FROM messages ORDER BY msg_id DESC LIMIT ?",
            (n,),
        ).fetchall()
        return [
            RawMessage(msg_id=r[0], timestamp=r[1], role=r[2], text=r[3], image_path=r[4] or None)
            for r in reversed(rows)
        ]


# ---------------------------------------------------------------------------
# BM25  (replaces Milvus built-in BM25 function / text_sparse collection)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> List[str]:
    return _TOKEN_RE.findall(text.lower())


class _BM25Index:
    """
    BM25 scorer. Replaces Milvus's built-in FunctionType.BM25.
    Identical scoring formula: Robertson et al. (1994).
    k1=1.5, b=0.75 — same defaults as Elasticsearch / Milvus defaults.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._ids: List[int] = []
        self._corpus_tokens: List[List[str]] = []
        self._idf: Dict[str, float] = {}
        self._doc_lens: List[int] = []
        self._avgdl: float = 1.0

    def fit(self, ids: List[int], texts: List[str]) -> None:
        self._ids = list(ids)
        self._corpus_tokens = [_tokenize(t) for t in texts]
        self._doc_lens = [len(tok) for tok in self._corpus_tokens]
        total = sum(self._doc_lens)
        self._avgdl = total / len(self._doc_lens) if self._doc_lens else 1.0

        N = len(self._corpus_tokens)
        df: Counter = Counter()
        for tok_list in self._corpus_tokens:
            for tok in set(tok_list):
                df[tok] += 1
        self._idf = {
            tok: math.log((N - freq + 0.5) / (freq + 0.5) + 1)
            for tok, freq in df.items()
        }

    def get_top_k(self, query: str, k: int = 5) -> List[Tuple[int, float]]:
        """Returns [(memory_id, score), ...] sorted descending."""
        if not self._corpus_tokens:
            return []
        q_toks = _tokenize(query)
        if not q_toks:
            return []
        scored: List[Tuple[int, float]] = []
        for idx, doc_toks in enumerate(self._corpus_tokens):
            tf_map: Counter = Counter(doc_toks)
            dl = self._doc_lens[idx]
            score = 0.0
            for tok in q_toks:
                tf = tf_map.get(tok, 0)
                idf = self._idf.get(tok, 0.0)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self._avgdl)
                if denom:
                    score += idf * (tf * (self.k1 + 1.0)) / denom
            scored.append((self._ids[idx], score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [(mid, s) for mid, s in scored[:k] if s > 0.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def _rrf(
    retrieved: List[List[Tuple[int, Any]]],
    k: int = 60,
) -> List[Tuple[int, float]]:
    """
    Reciprocal Rank Fusion. Faithful replica of official SemanticStore._rrf().
    retrieved: list of ranked lists, each element (memory_id, score).
    Returns: [(memory_id, rrf_score), ...] sorted descending.
    """
    rrf_map: Dict[int, float] = {}
    for path in retrieved:
        if not path:
            continue
        for rank, (mem_id, _) in enumerate(path, start=1):
            rrf_map[mem_id] = rrf_map.get(mem_id, 0.0) + 1.0 / (k + rank)
    return sorted(rrf_map.items(), key=lambda x: x[1], reverse=True)


# ---------------------------------------------------------------------------
# SemanticMemory dataclass  (faithful to official)
# ---------------------------------------------------------------------------

@dataclass
class SemanticMemory:
    """
    High-level semantic observation (upper memory layer).
    Faithful to official M2A SemanticMemory dataclass.
    """
    memory_id: Optional[int] = None
    text: Optional[str] = None
    image_caption: Optional[str] = None
    image_path: Optional[str] = None
    evidence_ids: List[List[int]] = field(default_factory=list)  # [[start, end], ...]


# ---------------------------------------------------------------------------
# SemanticStore  (in-memory Milvus replacement)
# ---------------------------------------------------------------------------

class SemanticStore:
    """
    In-memory semantic memory store.

    Mirrors official Milvus schema (three logical collections):
      - memory      : SemanticMemory objects
      - text_embs   : dense text vectors (MiniLM) + BM25 sparse index
      - image_embs  : dense image vectors (SigLIP2)

    Two-level RRF fusion (faithful to official hybrid_search):
      Level-1: dense text + BM25  → text_res  (replicates Milvus RRFRanker)
      Level-2: text_res + image_res → final    (replicates official _rrf)
    """

    def __init__(self, text_embedder: Any, multimodal_embedder: Any) -> None:
        self._text_embedder = text_embedder
        self._multimodal_embedder = multimodal_embedder

        # memory collection
        self._memories: Dict[int, SemanticMemory] = {}
        self._next_id: int = 1

        # text_embs collection
        self._text_dense: Dict[int, List[float]] = {}
        self._bm25 = _BM25Index()
        self._bm25_dirty = True

        # image_embs collection
        self._image_dense: Dict[int, List[float]] = {}

    # ---- helpers ----

    @staticmethod
    def _embed_text_for_index(text: Optional[str], caption: Optional[str]) -> str:
        """Combine text + caption for embedding — faithful to _get_memory_embed_text."""
        parts = []
        if text:
            parts.append(text.strip())
        if caption:
            parts.append(caption.strip())
        return " ".join(parts)

    def _rebuild_bm25(self) -> None:
        ids = list(self._memories.keys())
        texts = [
            self._embed_text_for_index(m.text, m.image_caption)
            for m in self._memories.values()
        ]
        self._bm25.fit(ids, texts)
        self._bm25_dirty = False

    # ---- CRUD ----

    def add(self, memory: SemanticMemory) -> int:
        """Add a new semantic memory. Returns assigned memory_id."""
        mem_id = self._next_id
        self._next_id += 1
        memory.memory_id = mem_id
        self._memories[mem_id] = memory

        # Text dense embedding
        embed_text = self._embed_text_for_index(memory.text, memory.image_caption)
        if embed_text and self._text_embedder is not None:
            try:
                vec = self._text_embedder.embed_query(embed_text)
                self._text_dense[mem_id] = vec
            except Exception:
                pass

        # Image dense embedding
        if memory.image_path and self._multimodal_embedder is not None:
            try:
                vec = self._multimodal_embedder.embed_image(memory.image_path)
                self._image_dense[mem_id] = vec
            except Exception:
                pass

        self._bm25_dirty = True
        return mem_id

    def delete(self, memory_id: int) -> None:
        self._memories.pop(memory_id, None)
        self._text_dense.pop(memory_id, None)
        self._image_dense.pop(memory_id, None)
        self._bm25_dirty = True

    def get_by_id(self, memory_id: int) -> Optional[SemanticMemory]:
        return self._memories.get(memory_id)

    # ---- search ----

    def hybrid_search(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: int = 8,
    ) -> List[SemanticMemory]:
        """
        Tri-path hybrid search with two-level RRF. Faithful to official hybrid_search().

        Level-1 (text path, replicates Milvus hybrid_search with RRFRanker):
          - path-1: dense text cosine similarity (MiniLM)
          - path-2: BM25 sparse retrieval
          → RRF fusion → text_res

        Level-2 (cross-modal, replicates official _rrf([text_res, image_res])):
          - text_res + image_res (SigLIP2 image similarity, only when image query)
          → RRF fusion → final
        """
        if not self._memories:
            return []

        INNER_LIMIT = 5  # matches official limit=5 per path

        # ---- Level-1: text path ----
        text_res: List[Tuple[int, float]] = []
        if query_text:
            # path-1: dense text
            dense_ranked: List[Tuple[int, float]] = []
            if self._text_dense and self._text_embedder is not None:
                try:
                    q_vec = self._text_embedder.embed_query(query_text)
                    sims = [
                        (mid, _cosine(q_vec, vec))
                        for mid, vec in self._text_dense.items()
                    ]
                    sims.sort(key=lambda x: x[1], reverse=True)
                    dense_ranked = sims[:INNER_LIMIT]
                except Exception:
                    pass

            # path-2: BM25
            if self._bm25_dirty:
                self._rebuild_bm25()
            bm25_ranked = self._bm25.get_top_k(query_text, k=INNER_LIMIT)

            # Level-1 RRF (replicates Milvus RRFRanker for dense+BM25)
            text_res = _rrf([dense_ranked, bm25_ranked])

        # ---- image path ----
        # Faithful to official M2A: supports both image→image (when query
        # image provided) and text→image cross-modal retrieval (project
        # text query into visual space via embed_text).
        image_res: List[Tuple[int, float]] = []
        if self._image_dense and self._multimodal_embedder is not None:
            try:
                if query_image_path:
                    # image→image similarity
                    q_vec = self._multimodal_embedder.embed_image(query_image_path)
                elif query_text:
                    # text→image cross-modal retrieval
                    q_vec = self._multimodal_embedder.embed_text(query_text)
                else:
                    q_vec = None

                if q_vec is not None:
                    sims = [
                        (mid, _cosine(q_vec, vec))
                        for mid, vec in self._image_dense.items()
                    ]
                    sims.sort(key=lambda x: x[1], reverse=True)
                    image_res = sims[:3]  # matches official limit=3
            except Exception:
                pass

        # ---- Level-2 RRF (replicates official _rrf([text_res, image_res])) ----
        final = _rrf([text_res, image_res])

        return [
            self._memories[mid]
            for mid, _ in final[:top_k]
            if mid in self._memories
        ]

    def __len__(self) -> int:
        return len(self._memories)
