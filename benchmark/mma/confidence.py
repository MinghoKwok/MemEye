"""
Lightweight confidence scoring adapted from MMA (AIGeeksGroup/MMA).

Three-component weighted score:
  confidence = w_source * source_score + w_time * time_score + w_consensus * consensus_score

Source credibility: user > knowledge > web > model
Temporal decay: exponential half-life (default 30 days)
Conflict-aware consensus: signed cosine similarity with linked memories
"""

from __future__ import annotations

import datetime as dt
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class MemoryEntry:
    """A single memory item with metadata for confidence scoring."""

    id: str
    text: str
    source: str = "model"  # user | knowledge | web | system | model
    created_at: Optional[dt.datetime] = None
    embedding: Optional[List[float]] = None
    image_embedding: Optional[List[float]] = None  # multimodal (CLIP/SigLIP) embedding
    image_paths: Optional[List[str]] = None
    round_id: str = ""
    session_id: str = ""
    confidence: float = 0.0
    links: Dict[str, Any] = field(default_factory=dict)


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return dot / (na * nb)


class ConfidenceScorer:
    """Lightweight confidence scorer following MMA's three-component design."""

    SOURCE_SCORES = {
        "user": 1.0,
        "knowledge": 0.9,
        "web": 0.7,
        "system": 0.6,
        "model": 0.6,
        "import": 0.5,
    }

    def __init__(
        self,
        w_source: float = 0.45,
        w_time: float = 0.40,
        w_consensus: float = 0.15,
        half_life_days: float = 30.0,
        link_top_k: int = 5,
    ):
        self.w_source = w_source
        self.w_time = w_time
        self.w_consensus = w_consensus
        self.half_life_days = half_life_days
        self.link_top_k = link_top_k

    def source_score(self, entry: MemoryEntry) -> float:
        src = entry.source.lower()
        for k, v in self.SOURCE_SCORES.items():
            if k in src:
                return v
        return 0.6

    def time_score(self, entry: MemoryEntry, now: Optional[dt.datetime] = None) -> float:
        if entry.created_at is None:
            return 0.5
        now = now or dt.datetime.now(dt.timezone.utc)
        created = entry.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        age_days = max(0.0, (now - created).total_seconds() / 86400.0)
        return 0.5 ** (age_days / max(self.half_life_days, 1e-3))

    def consensus_score(
        self, entry: MemoryEntry, all_entries: List[MemoryEntry]
    ) -> float:
        """Compute conflict-aware consensus from linked neighbors."""
        neighbors = entry.links.get("neighbors", [])
        if not neighbors:
            return 0.0

        id_to_entry = {e.id: e for e in all_entries}
        total_w = 0.0
        acc = 0.0
        for n in neighbors[: self.link_top_k]:
            neighbor = id_to_entry.get(n.get("target_id", ""))
            if neighbor is None:
                continue
            w = float(n.get("weight", 0.0))
            total_w += w
            support = 1.0
            if entry.embedding and neighbor.embedding:
                support = _cosine(entry.embedding, neighbor.embedding)
            acc += w * neighbor.confidence * support

        if total_w < 1e-9:
            return 0.0
        return max(-1.0, min(1.0, acc / total_w))

    def compute(
        self, entry: MemoryEntry, all_entries: Optional[List[MemoryEntry]] = None
    ) -> float:
        s = self.source_score(entry)
        t = self.time_score(entry)
        c = self.consensus_score(entry, all_entries or []) if all_entries else 0.0
        total_w = self.w_source + self.w_time + self.w_consensus
        if total_w < 1e-9:
            return 0.0
        score = (self.w_source * s + self.w_time * t + self.w_consensus * c) / total_w
        return max(0.0, min(1.0, score))

    def generate_links(
        self,
        entry: MemoryEntry,
        candidates: List[MemoryEntry],
        emb_weight: float = 0.7,
        text_weight: float = 0.3,
        min_weight: float = 0.1,
    ) -> None:
        """Compute and store similarity links for an entry."""
        from rapidfuzz import fuzz

        scored = []
        for c in candidates:
            if c.id == entry.id:
                continue
            emb_sim = _cosine(entry.embedding or [], c.embedding or [])
            txt_sim = fuzz.ratio(entry.text, c.text) / 100.0
            weight = emb_weight * emb_sim + text_weight * txt_sim
            if weight >= min_weight:
                scored.append((c.id, weight))

        scored.sort(key=lambda x: -x[1])
        entry.links = {
            "neighbors": [
                {"target_id": tid, "weight": round(w, 6), "type": "similarity"}
                for tid, w in scored[: self.link_top_k]
            ]
        }
