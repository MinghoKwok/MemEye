"""
M2ASystem: orchestrates the full M2A pipeline for benchmark evaluation.

Two-phase protocol (faithful to official eval_wrapper.py):
  1. process_all_sessions(dataset) — chat phase: build memory from all dialogue
  2. answer_question(question, images) — question phase: query memory and answer
"""
from __future__ import annotations

import datetime
import os
from typing import Any, Dict, List, Optional

from .chat_agent import QA_SYSTEM_PROMPT, ChatAgent
from ..embeddings import TextEmbedder, get_multimodal_embedder
from .image_manager import ImageManager
from .memory_manager import MemoryManager
from .stores import RawMessageStore, SemanticStore


class M2ASystem:
    """
    Full M2A system wired for benchmark evaluation.

    Configuration keys (all optional, with sensible defaults):
      llm_model                   : str  = "gpt-4o-mini"
      llm_api_key                 : str  = "" (falls back to OPENAI_API_KEY env var)
      llm_api_key_env             : str  = "OPENAI_API_KEY"
      llm_base_url                : str  = "https://api.openai.com/v1"
      llm_timeout                 : int  = 90
      text_embedding_model        : str  = "all-MiniLM-L6-v2"
      multimodal_embedding_model  : str  = "siglip2-base-patch16-384"
      max_memory_iterations       : int  = 15
      context_window              : int  = 5
      max_query_iterations        : int  = 5

    Image embedding fallback order:
      1. Local SigLIP2 (google/siglip2-base-patch16-384 via transformers)
      2. Local CLIP (openai/clip-vit-base-patch32 via transformers)
      3. None (image retrieval disabled, text-only search)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        cfg = config or {}

        llm_model = str(cfg.get("llm_model", "gpt-4o-mini"))
        api_key = str(cfg.get("llm_api_key", ""))
        api_key_env = str(cfg.get("llm_api_key_env", "OPENAI_API_KEY"))
        base_url = str(cfg.get("llm_base_url", "https://api.openai.com/v1"))
        llm_timeout = int(cfg.get("llm_timeout", 90))

        text_model = str(cfg.get("text_embedding_model", "all-MiniLM-L6-v2"))
        mm_model = str(cfg.get("multimodal_embedding_model", "siglip2-base-patch16-384"))

        max_mem_iter = int(cfg.get("max_memory_iterations", 15))
        ctx_window = int(cfg.get("context_window", 5))
        max_q_iter = int(cfg.get("max_query_iterations", 5))

        # Resolve LLM API key
        if not api_key:
            api_key = os.environ.get(api_key_env, "")

        # OpenAI client for LLM calls
        import openai  # type: ignore
        self._llm_client = openai.OpenAI(api_key=api_key, base_url=base_url, timeout=llm_timeout)
        self._llm_model = llm_model
        self._max_q_iter = max_q_iter

        # Embedders
        self._text_embedder = TextEmbedder(text_model)
        # Try local SigLIP2, fallback to local CLIP, or None
        self._multimodal_embedder = get_multimodal_embedder(
            vllm_model=mm_model,
        )

        # Stores
        self._raw_store = RawMessageStore()
        self._semantic_store = SemanticStore(
            text_embedder=self._text_embedder,
            multimodal_embedder=self._multimodal_embedder,
        )
        self._image_manager = ImageManager()

        # MemoryManager
        self._memory_manager = MemoryManager(
            raw_store=self._raw_store,
            semantic_store=self._semantic_store,
            image_manager=self._image_manager,
            llm_client=self._llm_client,
            model=llm_model,
            max_iterations=max_mem_iter,
            context_window=ctx_window,
        )

        # ChatAgent for chat phase (update_only=True: no response generation)
        self._chat_agent = ChatAgent(
            memory_manager=self._memory_manager,
            raw_store=self._raw_store,
            image_manager=self._image_manager,
            llm_client=self._llm_client,
            model=llm_model,
            update_memory=True,
            update_only=True,
        )

        # Collect speaker names and dates for QA system prompt context
        self._speakers: List[str] = []
        self._session_dates: List[str] = []

    # ---- Phase 1: chat (build memory) ----

    def process_all_sessions(self, dataset: Any) -> None:
        """
        Chat phase: process all dialogue sessions in temporal order to build memory.
        Faithful to official eval_wrapper.chat() protocol.

        For each round in session order:
          - user turn  → store in raw store + update memory
          - assistant turn → store in raw store + update memory
        """
        session_ids = list(dataset.session_order())
        for session_idx, session_id in enumerate(session_ids, start=1):
            session = dataset.get_session(session_id)
            date = str(session.get("date", "")).strip()
            if date and date not in self._session_dates:
                self._session_dates.append(date)

            dialogues = session.get("dialogues", [])
            print(
                f"[M2A] Session {session_idx}/{len(session_ids)} "
                f"{session_id} date={date or 'unknown'} rounds={len(dialogues)}"
            )
            for round_idx, dialogue in enumerate(dialogues, start=1):
                round_id = dialogue.get("round", "")
                if not round_id:
                    continue
                round_payload = dataset.rounds.get(round_id, {})
                user_text = str(round_payload.get("user", "")).strip()
                assistant_text = str(round_payload.get("assistant", "")).strip()
                images: List[str] = round_payload.get("images", []) or []
                print(
                    f"[M2A] Round {round_idx}/{len(dialogues)} {round_id} "
                    f"user_images={len(images)} has_user={bool(user_text)} "
                    f"has_assistant={bool(assistant_text)}"
                )

                # User turn
                if user_text:
                    self._chat_agent.chat(
                        text=user_text,
                        image_paths=images if images else None,
                        timestamp=date,
                        role="user",
                    )

                # Assistant turn
                if assistant_text:
                    self._chat_agent.chat(
                        text=assistant_text,
                        timestamp=date,
                        role="assistant",
                    )

    # ---- Phase 2: question (query memory) ----

    def answer_question(
        self,
        question: str,
        image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        Question phase: create a fresh ChatAgent and query memory to answer.
        Faithful to official eval_wrapper.question() protocol.
        """
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        speakers_str = (
            ", ".join(self._speakers) if self._speakers else "user, assistant"
        )

        qa_prompt = QA_SYSTEM_PROMPT.format(
            current_datetime=current_time,
            speakers=speakers_str,
        )

        # Fresh ChatAgent per question (faithful to official: new instance, no updates)
        qa_agent = ChatAgent(
            memory_manager=self._memory_manager,
            raw_store=self._raw_store,
            image_manager=self._image_manager,
            llm_client=self._llm_client,
            model=self._llm_model,
            update_memory=False,
            update_only=False,
            max_query_iterations=self._max_q_iter,
            system_prompt=qa_prompt,
        )

        return qa_agent.chat(
            text=question,
            image_paths=image_paths,
            timestamp=current_time,
            role="user",
        )

    @property
    def num_memories(self) -> int:
        return len(self._semantic_store)
