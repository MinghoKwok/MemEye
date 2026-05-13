"""
ChatAgent: manages conversation flow with memory query / update.
Replaces LangGraph with a native tool-calling loop.
Faithful to official M2A agent/agents/chat_agent.py.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Dict, List, Optional

from .image_manager import ImageManager
from .memory_manager import MemoryManager
from .stores import RawMessageStore


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = "You are an AI assistant with access to long-term memory."

QA_SYSTEM_PROMPT = """\
You are an AI assistant answering questions based on past conversation memories.
The current date/time is {current_datetime}.
Conversation participants: {speakers}.
Use the query_memory tool to retrieve relevant past information before answering.
Answer concisely and directly based on the retrieved memories.\
"""

MAX_QUERY_ITERATIONS = 5


def _retry_wait_seconds(exc: Exception, attempt: int) -> int:
    """Bounded 429 retry delay for ChatAgent question-time calls."""
    default_wait = min((2 ** attempt) + 1, 60)
    resp = getattr(exc, "response", None)
    headers = getattr(resp, "headers", {}) or {}

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            hinted = max(1, int(float(retry_after_ms) / 1000.0 + 0.999))
            return max(default_wait, hinted)
        except (TypeError, ValueError):
            pass

    retry_after = headers.get("retry-after")
    if retry_after:
        try:
            hinted = max(1, int(float(retry_after)))
            return max(default_wait, hinted)
        except (TypeError, ValueError):
            pass

    match = re.search(r"try again in\s+([0-9.]+)\s*(ms|s)\b", str(exc), re.IGNORECASE)
    if match:
        value = float(match.group(1))
        unit = match.group(2).lower()
        hinted = max(1, int(value / 1000.0 + 0.999)) if unit == "ms" else max(1, int(value + 0.999))
        return max(default_wait, hinted)

    return default_wait


def _is_retryable_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return (
        "rate_limit" in text
        or "429" in text
        or "timed out" in text
        or "timeout" in text
        or "connection error" in text
        or "api connection" in text
    )


# ---------------------------------------------------------------------------
# Tool schemas for ChatAgent
# ---------------------------------------------------------------------------

def _tool_query_memory() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "query_memory",
            "description": (
                "Query memory only when past, persistent, or non-local context is required. "
                "Do NOT query memory if the answer can be derived from the recent conversation. "
                "Text query MUST NOT contain any image tokens. "
                "Image tokens are ONLY allowed in the image field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Text query for semantic memory search. "
                            "Must NOT contain image tokens."
                        ),
                    },
                    "image": {
                        "type": "string",
                        "description": (
                            "Image token (e.g. <image0>) to search by visual content. "
                            "ONLY allowed in this field."
                        ),
                    },
                },
                "required": [],
            },
        },
    }


def _tool_update_memory() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "update_memory",
            "description": (
                "Store important facts, personal information, temporal markers, "
                "preferences, habits, or session summaries to long-term memory. "
                "DO NOT duplicate existing memories. "
                "DEFAULT TO UPDATE: when uncertain, prefer updating over skipping. "
                "Text MUST NOT contain image tokens. "
                "Image tokens are ONLY allowed in the image field."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Content to store. Must NOT contain image tokens.",
                    },
                    "image": {
                        "type": "string",
                        "description": (
                            "Image token (e.g. <image0>) if visual content should be stored. "
                            "ONLY allowed here, not in text."
                        ),
                    },
                },
                "required": [],
            },
        },
    }


# ---------------------------------------------------------------------------
# ChatAgent
# ---------------------------------------------------------------------------

class ChatAgent:
    """
    Manages one conversation session with memory integration.

    Two modes (faithful to official ChatAgent):
      update_only=True  — chat phase: no response generation, only memory updates
      update_only=False — question phase: generate response, optionally query memory
    """

    def __init__(
        self,
        memory_manager: MemoryManager,
        raw_store: RawMessageStore,
        image_manager: ImageManager,
        llm_client: Any,
        model: str = "gpt-4o-mini",
        update_memory: bool = True,
        update_only: bool = False,
        max_chat_context_tokens: int = 8000,
        max_query_iterations: int = MAX_QUERY_ITERATIONS,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    ) -> None:
        self._mm = memory_manager
        self._raw = raw_store
        self._image_manager = image_manager
        self._client = llm_client
        self._model = model
        self._update_memory = update_memory
        self._update_only = update_only
        self._max_ctx = max_chat_context_tokens
        self._max_query_iter = max_query_iterations
        self.system_prompt = system_prompt

        # In-session conversation history for LLM context window
        self._messages: List[Dict[str, Any]] = []

    # ---- helpers ----

    def _trim_messages(self) -> List[Dict[str, Any]]:
        """Keep context window bounded. Faithful to official trim_messages()."""
        total = sum(len(str(m.get("content", ""))) // 4 for m in self._messages)
        while total > self._max_ctx and len(self._messages) > 1:
            removed = self._messages.pop(0)
            total -= len(str(removed.get("content", ""))) // 4
        return self._messages

    def _format_user_content(
        self,
        text: str,
        image_paths: Optional[List[str]],
    ) -> Any:
        if not image_paths:
            return text
        return self._image_manager.format_content(text, image_paths)

    # ---- main interface ----

    def chat(
        self,
        text: str,
        image_paths: Optional[List[str]] = None,
        timestamp: str = "",
        role: str = "user",
    ) -> str:
        """
        Process one conversation turn.

        update_only=True (chat phase):
          Stores raw message, triggers memory update, returns "".
        update_only=False (question phase):
          Optionally queries memory, generates response, returns it.
        """
        # Register images with ImageManager for token mapping
        for path in (image_paths or []):
            self._image_manager.image_to_token(path)

        # Store in raw message store
        img_path = image_paths[0] if image_paths else None
        msg_id = self._raw.append(
            timestamp=timestamp,
            role=role,
            text=text,
            image_path=img_path,
        )

        if self._update_only:
            # Chat phase: update memory from this turn, no response generation
            if self._update_memory:
                update_text = f"[{timestamp}] {role}: {text}"
                self._mm.update(update_text, image_paths=image_paths)
            return ""

        # Question phase: add to context, generate response
        user_content = self._format_user_content(text, image_paths)
        self._messages.append({"role": "user", "content": user_content})

        response_text = self._generate_response()

        self._messages.append({"role": "assistant", "content": response_text})

        if self._update_memory:
            update_text = f"[{timestamp}] {role}: {text}\nassistant: {response_text}"
            self._mm.update(update_text, image_paths=image_paths)

        return response_text

    def _generate_response(self) -> str:
        """
        Generate a response, optionally calling query_memory tool.
        Faithful to official _generate_response / handle_query pattern.
        """
        messages_with_sys = [
            {"role": "system", "content": self.system_prompt}
        ] + self._trim_messages()

        for iteration in range(self._max_query_iter):
            # Disable tools on the last iteration to force a text response
            use_tools = iteration < self._max_query_iter - 1
            kwargs: Dict[str, Any] = {
                "model": self._model,
                "messages": messages_with_sys,
            }
            if use_tools:
                kwargs["tools"] = [_tool_query_memory()]
                kwargs["tool_choice"] = "auto"

            # Retry with exponential backoff for rate limit errors
            max_retries = 12
            for attempt in range(max_retries):
                try:
                    response = self._client.chat.completions.create(**kwargs)
                    break
                except Exception as e:
                    if _is_retryable_error(e):
                        wait_time = _retry_wait_seconds(e, attempt)
                        print(f"[M2A] Retryable LLM error, waiting {wait_time}s (attempt {attempt + 1}/{max_retries})")
                        time.sleep(wait_time)
                        if attempt == max_retries - 1:
                            raise
                    else:
                        raise
            msg = response.choices[0].message

            if not msg.tool_calls:
                return msg.content or ""

            # Execute query_memory tool calls
            messages_with_sys.append(msg.model_dump(exclude_unset=True))
            for tc in msg.tool_calls:
                if tc.function.name == "query_memory":
                    args = json.loads(tc.function.arguments or "{}")
                    query_text = args.get("text", "")
                    query_image_token = args.get("image", "")
                    query_image_path = (
                        self._image_manager.token_to_image(query_image_token)
                        if query_image_token
                        else None
                    )
                    result = self._mm.query(
                        query_text,
                        query_image_paths=[query_image_path] if query_image_path else None,
                    )
                    messages_with_sys.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result or "No relevant memories found.",
                        }
                    )

        return ""
