"""
MMASystem: Confidence-aware multimodal memory agent for benchmark evaluation.

Adapted from MMA (AIGeeksGroup/MMA). Core innovation: confidence-weighted
memory retrieval using source credibility, temporal decay, and conflict-aware
consensus scoring.

Two-phase protocol (same as M2A):
  1. process_all_sessions(dataset) — build memory store from all dialogue
  2. answer_question(question)     — retrieve & answer using confidence-ranked memory
"""

from __future__ import annotations

import datetime as dt
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .confidence import ConfidenceScorer, MemoryEntry


@dataclass
class MMAConfig:
    # LLM settings
    llm_model: str = "gpt-4.1-nano"
    llm_api_key: str = ""
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_base_url: str = "https://api.openai.com/v1"
    llm_timeout: int = 90

    # Embedding
    text_embedding_model: str = "all-MiniLM-L6-v2"

    # Multimodal embedding (reuse M2A's CLIP/SigLIP infrastructure)
    multimodal_embedding_model: str = "google/siglip-so400m-patch14-384"
    image_text_weight: float = 0.5  # blend weight: text vs image retrieval

    # Retrieval
    retrieval_top_k: int = 10

    # Confidence scoring
    w_source: float = 0.45
    w_time: float = 0.40
    w_consensus: float = 0.15
    half_life_days: float = 30.0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MMAConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


def _load_shared_sys_prompt(question: str) -> str:
    """Load the shared MemEye system prompt, choosing open vs mcq based on question."""
    import re as _re
    prompt_dir = Path(__file__).resolve().parent.parent / "prompt"
    # Detect MCQ by option pattern in question (e.g. "A. ...")
    is_mcq = bool(_re.search(r"\n[A-Z]\.\s", question))
    fname = "sys_prompt_mcq.txt" if is_mcq else "sys_prompt_open.txt"
    path = prompt_dir / fname
    if not path.exists():
        path = prompt_dir / "sys_prompt.txt"
    return path.read_text(encoding="utf-8").strip()


def _parse_session_date(date_str: str, session_idx: int) -> dt.datetime:
    """Parse session date string into a datetime, falling back to index-based spacing.

    Tries common formats (ISO, YYYY-MM-DD). If parsing fails, spaces sessions
    one day apart so that temporal decay still produces meaningful differences.
    """
    date_str = date_str.strip()
    if not date_str:
        # No date — use index-based synthetic timestamp
        return dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=session_idx)

    # Try ISO / YYYY-MM-DD
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = dt.datetime.strptime(date_str, fmt)
            return parsed.replace(tzinfo=dt.timezone.utc)
        except ValueError:
            continue

    # Fallback: use index-based spacing
    return dt.datetime(2025, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=session_idx)


class MMASystem:
    """
    Confidence-aware multimodal memory system for benchmark evaluation.

    Memory entries are scored using three components:
      - Source credibility (user utterances > model inferences)
      - Temporal decay (recent memories weighted higher, using session dates)
      - Conflict-aware consensus (agreement with linked memories)

    Retrieval blends text embedding similarity with image embedding similarity
    for entries that carry visual content.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg_dict = config or {}
        self.cfg = MMAConfig.from_dict(cfg_dict)

        # Resolve API key
        api_key = self.cfg.llm_api_key
        if not api_key:
            api_key = os.environ.get(self.cfg.llm_api_key_env, "")

        # OpenAI client
        import openai

        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=self.cfg.llm_base_url,
            timeout=self.cfg.llm_timeout,
        )
        self._model = self.cfg.llm_model

        # Text embedder
        from ..embeddings import TextEmbedder

        self._text_embedder = TextEmbedder(self.cfg.text_embedding_model)

        # Multimodal embedder (best-effort: CLIP/SigLIP, falls back to None)
        from ..embeddings import get_multimodal_embedder

        self._mm_embedder = get_multimodal_embedder(
            vllm_model=self.cfg.multimodal_embedding_model,
        )
        if self._mm_embedder is not None:
            print(f"[MMA] Multimodal embedder: {type(self._mm_embedder).__name__}")
        else:
            print("[MMA] No multimodal embedder available — text-only retrieval")

        # Confidence scorer
        self._scorer = ConfidenceScorer(
            w_source=self.cfg.w_source,
            w_time=self.cfg.w_time,
            w_consensus=self.cfg.w_consensus,
            half_life_days=self.cfg.half_life_days,
        )

        # Memory store
        self._memories: List[MemoryEntry] = []
        self._session_dates: List[str] = []

    # ---- Phase 1: build memory from dialogue ----

    def process_all_sessions(self, dataset: Any) -> None:
        """Build memory entries from all dialogue sessions."""
        session_ids = list(dataset.session_order())
        entry_id = 0

        for session_idx, session_id in enumerate(session_ids, start=1):
            session = dataset.get_session(session_id)
            date_str = str(session.get("date", "")).strip()
            if date_str and date_str not in self._session_dates:
                self._session_dates.append(date_str)

            # Fix #1: Use session date (or index-based synthetic date) for temporal decay
            session_dt = _parse_session_date(date_str, session_idx)

            dialogues = session.get("dialogues", [])
            print(
                f"[MMA] Session {session_idx}/{len(session_ids)} "
                f"{session_id} date={date_str or 'unknown'} rounds={len(dialogues)}"
            )

            for round_idx, dialogue in enumerate(dialogues):
                round_id = dialogue.get("round", "")
                if not round_id:
                    continue

                round_data = dataset.rounds.get(round_id, {})
                user_text = str(round_data.get("user", "")).strip()
                asst_text = str(round_data.get("assistant", "")).strip()
                images: List[str] = round_data.get("images", []) or []

                # Offset each round by minutes within the session
                round_dt = session_dt + dt.timedelta(minutes=round_idx)

                # Create memory entry for user turn
                if user_text:
                    entry_id += 1
                    mem = MemoryEntry(
                        id=f"mem_{entry_id}",
                        text=user_text,
                        source="user",
                        created_at=round_dt,
                        image_paths=images if images else None,
                        round_id=round_id,
                        session_id=session_id,
                    )
                    self._memories.append(mem)

                # Create memory entry for assistant turn
                if asst_text:
                    entry_id += 1
                    mem = MemoryEntry(
                        id=f"mem_{entry_id}",
                        text=asst_text,
                        source="model",
                        created_at=round_dt,
                        round_id=round_id,
                        session_id=session_id,
                    )
                    self._memories.append(mem)

        # Compute text embeddings for all entries
        print(f"[MMA] Computing text embeddings for {len(self._memories)} memory entries...")
        texts = [m.text for m in self._memories]
        if texts:
            embeddings = self._text_embedder.embed_batch(texts)
            for mem, emb in zip(self._memories, embeddings):
                mem.embedding = emb.tolist() if hasattr(emb, "tolist") else list(emb)

        # Fix #2: Compute image embeddings for entries with images
        if self._mm_embedder is not None:
            image_entries = [m for m in self._memories if m.image_paths]
            if image_entries:
                print(f"[MMA] Computing image embeddings for {len(image_entries)} visual entries...")
                for mem in image_entries:
                    try:
                        # Embed all images and average (not just the first)
                        embs = []
                        for img_path in mem.image_paths:
                            emb = self._mm_embedder.embed_image(img_path)
                            if hasattr(emb, "tolist"):
                                emb = emb.tolist()
                            embs.append(emb)
                        if embs:
                            avg = [sum(col) / len(col) for col in zip(*embs)]
                            mem.image_embedding = avg
                    except Exception as e:
                        print(f"[MMA] Image embedding failed for {mem.round_id}: {e}")

        # Generate links between memories
        print("[MMA] Generating similarity links...")
        try:
            for mem in self._memories:
                self._scorer.generate_links(mem, self._memories)
        except ImportError:
            print("[MMA] rapidfuzz not available, skipping link generation")

        # Compute confidence scores with iterative convergence
        print("[MMA] Computing confidence scores...")
        max_iters = 3
        for iteration in range(max_iters):
            max_delta = 0.0
            for mem in self._memories:
                old = mem.confidence
                mem.confidence = self._scorer.compute(mem, self._memories)
                max_delta = max(max_delta, abs(mem.confidence - old))
            if iteration > 0 and max_delta < 1e-4:
                break
        print(f"[MMA] Confidence converged after {iteration + 1} iteration(s) (max_delta={max_delta:.6f}).")

        print(f"[MMA] Memory ready: {len(self._memories)} entries indexed.")

    # ---- Phase 2: answer questions ----

    def _retrieve(
        self,
        question: str,
        query_image_paths: Optional[List[str]] = None,
        top_k: Optional[int] = None,
    ) -> List[MemoryEntry]:
        """Retrieve top-k memories using blended text + image similarity * confidence.

        Score = (w_text * text_sim + w_image * image_sim) * confidence
        where w_text + w_image = 1.0 (configurable via image_text_weight).

        If query_image_paths are provided, the query image embedding is computed
        from those images and used for cross-modal retrieval. Otherwise the text
        query is projected into the visual space via embed_text().
        """
        top_k = top_k or self.cfg.retrieval_top_k
        if not self._memories:
            return []

        q_text_emb = self._text_embedder.embed_query(question)
        q_text = q_text_emb.tolist() if hasattr(q_text_emb, "tolist") else list(q_text_emb)

        # Build query image embedding for cross-modal retrieval
        q_img_emb = None
        if self._mm_embedder is not None:
            # Prefer actual query images if provided (average all, not just first)
            if query_image_paths:
                try:
                    embs = []
                    for qip in query_image_paths:
                        emb = self._mm_embedder.embed_image(qip)
                        embs.append(emb.tolist() if hasattr(emb, "tolist") else list(emb))
                    if embs:
                        q_img_emb = [sum(col) / len(col) for col in zip(*embs)]
                except Exception:
                    pass
            # Fallback: project question text into visual space
            if q_img_emb is None:
                try:
                    q_img_emb = self._mm_embedder.embed_text(question)
                    if hasattr(q_img_emb, "tolist"):
                        q_img_emb = q_img_emb.tolist()
                except Exception:
                    pass

        w_text = 1.0 - self.cfg.image_text_weight
        w_image = self.cfg.image_text_weight

        scored = []
        for mem in self._memories:
            if mem.embedding is None:
                continue

            # Text similarity (embeddings are L2-normalized by sentence-transformers)
            text_sim = sum(a * b for a, b in zip(q_text, mem.embedding))

            # Image similarity (if both query and memory have image embeddings)
            img_sim = 0.0
            mem_img_emb = getattr(mem, "image_embedding", None)
            if q_img_emb is not None and mem_img_emb is not None:
                img_sim = sum(a * b for a, b in zip(q_img_emb, mem_img_emb))

            # Blend: always apply w_text scaling so text-only entries
            # don't get an unfair advantage over multimodal entries.
            if mem_img_emb is not None and q_img_emb is not None:
                sim = w_text * text_sim + w_image * img_sim
            else:
                sim = w_text * text_sim

            score = sim * mem.confidence
            scored.append((score, mem))

        scored.sort(key=lambda x: -x[0])
        return [mem for _, mem in scored[:top_k]]

    def _format_context(self, entries: List[MemoryEntry]) -> str:
        """Format retrieved memories as context string with confidence indicators."""
        if not entries:
            return "(No relevant memories found.)"
        lines = []
        for i, mem in enumerate(entries, 1):
            conf_label = (
                "HIGH" if mem.confidence >= 0.7
                else "MED" if mem.confidence >= 0.4
                else "LOW"
            )
            img_note = ""
            if mem.image_paths:
                img_note = f" [+{len(mem.image_paths)} image(s)]"
            lines.append(
                f"[{i}] (conf={conf_label}, src={mem.source}, "
                f"session={mem.session_id}, round={mem.round_id}){img_note}\n"
                f"    {mem.text}"
            )
        return "\n".join(lines)

    def _build_messages(
        self, question: str, context: str, image_paths: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """Build chat messages with optional image content."""
        sys_prompt = _load_shared_sys_prompt(question)

        user_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    f"Memory context (ranked by confidence):\n{context}\n\n"
                    f"Question: {question}"
                ),
            }
        ]

        # Attach images from retrieved memories (compressed to avoid API size limits)
        if image_paths:
            from router.http_utils import encode_image_data_url

            seen = set()
            for img_path in image_paths:
                if img_path in seen:
                    continue
                seen.add(img_path)
                if len(seen) > 5:  # Limit to 5 unique images
                    break
                try:
                    data_url = encode_image_data_url(img_path)
                    user_content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "low"},
                        }
                    )
                except FileNotFoundError:
                    continue

        return [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]

    def answer_question(
        self, question: str, image_paths: Optional[List[str]] = None
    ) -> str:
        """Answer a question using confidence-weighted memory retrieval.

        Args:
            question: The question text.
            image_paths: Optional images attached to the question itself
                         (Fix #3: propagated from runner → method → here).
        """
        # Use query images for both retrieval and generation
        retrieved = self._retrieve(question, query_image_paths=image_paths)

        # Collect images: query images first, then retrieved memory images
        all_images: List[str] = []
        if image_paths:
            all_images.extend(image_paths)
        for mem in retrieved:
            if mem.image_paths:
                all_images.extend(mem.image_paths)

        context = self._format_context(retrieved)
        messages = self._build_messages(question, context, all_images or None)

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_completion_tokens=256,
                temperature=0,
            )
            return str(response.choices[0].message.content).strip()
        except Exception as e:
            print(f"[MMA] Answer generation failed: {e}")
            return "Unable to generate answer"

    @property
    def num_memories(self) -> int:
        return len(self._memories)
