"""
MemoryManager: LLM-driven agent for memory QUERY and UPDATE operations.
Implements the ReAct tool-calling loop natively (no LangGraph dependency).
Faithful to official M2A agent/agents/memory_manager.py.
"""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Optional

from .image_manager import ImageManager
from .stores import RawMessageStore, SemanticMemory, SemanticStore


# ---------------------------------------------------------------------------
# System prompts  (faithful to official M2A)
# ---------------------------------------------------------------------------

QUERY_SYSTEM_PROMPT = """\
You are MemoryManager handling a QUERY request from ChatAgent.
[GOAL] Retrieve relevant information and return a focused answer to the query.

[CONTEXT]
{context}

[STEPS]
1. SEARCH semantic memories (always start here). Use query_text and/or query_image \
to retrieve relevant memories.
2. Assess whether the retrieved semantic memories sufficiently answer the query.
   - If yes: return the answer directly.
   - If no: use fetch_raw_messages to get detailed conversation context using the \
evidence_ids from the memories.
3. Optionally use fetch_raw_messages_by_time if you need to look up information from \
a specific time period.
4. Return a concise, query-focused answer based on the retrieved information.

[NOTES]
- Always convert relative time references (e.g. "last week", "yesterday") to specific \
absolute dates using the current date in [CONTEXT].
- Return only information relevant to the query. Do not pad with unrelated details.
- If no relevant information is found, say so clearly.\
"""

UPDATE_SYSTEM_PROMPT = """\
You are MemoryManager processing an UPDATE request from ChatAgent.
[GOAL] Analyze ChatAgent's update suggestion and maintain the semantic memory database.

[CONTEXT]
{context}

[STEPS]
1. Understand the request: parse mentioned entities, events, and timeframes.
2. Query existing memory: check for conflicts, duplicates, or related memories that \
may need updating.
3. Plan your operations — choose ONE of: CREATE, DELETE, BOTH, or NONE.
4. Execute the planned operations.
5. Respond with an empty string when done.

[RULES]
- Always resolve relative time references to absolute dates (e.g. "yesterday" → \
"2026-02-10") using the date in [CONTEXT].
- Use specific names for entities — never use pronouns like "she", "he", "it", "they".
- Always provide evidence_ids when adding memories (raw message ID ranges that support \
the memory).
- Break complex information into atomic memories (one fact per memory entry).
- Prefer updating or deleting stale memories over creating duplicates.
- If the incoming information is already covered by existing memories, choose NONE.\
"""

MAX_ITERATIONS = 15


def _retry_wait_seconds(exc: Exception, attempt: int) -> int:
    """
    Compute a bounded wait time for rate-limit retries.

    OpenAI 429s may include a short "try again in" hint, but for long-running M2A
    memory builds we need to survive full TPM windows, not just sub-second bursts.
    """
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
# Tool JSON schemas  (faithful to official tool definitions)
# ---------------------------------------------------------------------------

def _tool_search() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "search_semantic_memories",
            "description": "Search high-level semantic memories using text or image or both.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query_text": {
                        "type": "string",
                        "description": "Text query to search semantic memories.",
                    },
                    "query_image": {
                        "type": "string",
                        "description": (
                            "Image token (e.g. <image0>) to search by visual content. "
                            "MUST NOT appear in query_text."
                        ),
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Number of top memories to return. Default: 5.",
                        "default": 5,
                    },
                },
                "required": [],
            },
        },
    }


def _tool_fetch_raw() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_raw_messages",
            "description": "Fetch raw conversation messages by ID ranges.",
            "parameters": {
                "type": "object",
                "properties": {
                    "id_ranges": {
                        "type": "string",
                        "description": (
                            "JSON array of [start, end] ID ranges, e.g. "
                            '"[[1, 5], [10, 12]]". Returns up to 20 messages.'
                        ),
                    },
                },
                "required": ["id_ranges"],
            },
        },
    }


def _tool_fetch_by_time() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "fetch_raw_messages_by_time",
            "description": "Fetch raw conversation messages within a date range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "start_date": {
                        "type": "string",
                        "description": "Start date (ISO format YYYY-MM-DD).",
                    },
                    "end_date": {
                        "type": "string",
                        "description": "End date (ISO format YYYY-MM-DD).",
                    },
                },
                "required": ["start_date", "end_date"],
            },
        },
    }


def _tool_add_memory() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "add_memory",
            "description": "Create a new semantic memory entry.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "The memory text. Must NOT contain image tokens.",
                    },
                    "image": {
                        "type": "string",
                        "description": (
                            "Image token (e.g. <image0>) if this memory has visual content. "
                            "ONLY allowed here, not in text."
                        ),
                    },
                    "image_caption": {
                        "type": "string",
                        "description": "Text description of the image (if any).",
                    },
                    "evidence_ids": {
                        "type": "string",
                        "description": (
                            "JSON array of raw message ID ranges supporting this memory, "
                            'e.g. "[[1, 3]]".'
                        ),
                    },
                },
                "required": ["text", "evidence_ids"],
            },
        },
    }


def _tool_delete_memory() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "Delete a semantic memory by its ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "memory_id": {
                        "type": "integer",
                        "description": "The memory_id to delete.",
                    },
                },
                "required": ["memory_id"],
            },
        },
    }


QUERY_TOOLS = [_tool_search(), _tool_fetch_raw(), _tool_fetch_by_time()]
ALL_TOOLS = QUERY_TOOLS + [_tool_add_memory(), _tool_delete_memory()]


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------

class MemoryManager:
    """
    LLM-driven memory manager with QUERY and UPDATE ReAct loops.
    Replaces LangGraph with a native tool-calling while-loop.
    Faithful to official M2A MemoryManager behavior and prompts.
    """

    MAX_RAW_MESSAGES = 20

    def __init__(
        self,
        raw_store: RawMessageStore,
        semantic_store: SemanticStore,
        image_manager: ImageManager,
        llm_client: Any,
        model: str = "gpt-4o-mini",
        max_iterations: int = MAX_ITERATIONS,
        context_window: int = 5,
    ) -> None:
        self._raw = raw_store
        self._semantic = semantic_store
        self._image_manager = image_manager
        self._client = llm_client
        self._model = model
        self._max_iter = max_iterations
        self._ctx_window = context_window

    # ---- context ----

    def _prepare_context(self) -> str:
        """Build context string from recent raw messages (faithful to _prepair_context)."""
        recent = self._raw.get_latest_n(self._ctx_window)
        if not recent:
            return "(no recent conversation)"
        lines = []
        for msg in recent:
            img_note = (
                f" [image: {self._image_manager.image_to_token(msg.image_path)}]"
                if msg.image_path
                else ""
            )
            lines.append(f"[{msg.timestamp}] {msg.role}: {msg.text}{img_note}")
        return "\n".join(lines)

    # ---- tool execution ----

    def _exec_search(self, args: Dict[str, Any]) -> str:
        query_text = args.get("query_text") or args.get("query") or args.get("text") or args.get("search_query") or ""
        query_image_token = args.get("query_image") or args.get("image") or args.get("image_token") or ""
        top_k = int(args.get("top_k", args.get("k", 5)))

        query_image_path: Optional[str] = None
        if query_image_token:
            query_image_path = self._image_manager.token_to_image(query_image_token)

        memories = self._semantic.hybrid_search(
            query_text=query_text or None,
            query_image_path=query_image_path,
            top_k=top_k,
        )
        if not memories:
            return "No matching memories found."

        lines = []
        for m in memories:
            img_tok = self._image_manager.image_to_token(m.image_path) if m.image_path else ""
            line = f"[memory_id={m.memory_id}] {m.text or ''}"
            if m.image_caption:
                line += f" | caption: {m.image_caption}"
            if img_tok:
                line += f" | image: {img_tok}"
            line += f" | evidence_ids: {json.dumps(m.evidence_ids)}"
            lines.append(line)
        return "\n".join(lines)

    def _exec_fetch_raw(self, args: Dict[str, Any]) -> str:
        try:
            raw = args.get("id_ranges") or args.get("ids") or args.get("message_ids") or args.get("range") or "[]"
            id_ranges = json.loads(raw) if isinstance(raw, str) else raw
            # Accept flat pair [1, 5] as [[1, 5]]
            if id_ranges and isinstance(id_ranges, list) and not isinstance(id_ranges[0], list):
                id_ranges = [id_ranges]
        except (json.JSONDecodeError, TypeError):
            return "Error: invalid id_ranges. Expected JSON like [[1, 5]]."

        messages = self._raw.fetch_by_ids(id_ranges)[: self.MAX_RAW_MESSAGES]
        if not messages:
            return "No messages found for the given ID ranges."

        lines = []
        for msg in messages:
            img_note = (
                f" [image: {self._image_manager.image_to_token(msg.image_path)}]"
                if msg.image_path
                else ""
            )
            lines.append(
                f"[id={msg.msg_id}][{msg.timestamp}] {msg.role}: {msg.text}{img_note}"
            )
        return "\n".join(lines)

    def _exec_fetch_by_time(self, args: Dict[str, Any]) -> str:
        start = args.get("start_date", "")
        end = args.get("end_date", "")
        messages = self._raw.fetch_by_timerange(start, end)[: self.MAX_RAW_MESSAGES]
        if not messages:
            return "No messages found in the given time range."
        lines = []
        for msg in messages:
            img_note = (
                f" [image: {self._image_manager.image_to_token(msg.image_path)}]"
                if msg.image_path
                else ""
            )
            lines.append(
                f"[id={msg.msg_id}][{msg.timestamp}] {msg.role}: {msg.text}{img_note}"
            )
        return "\n".join(lines)

    @staticmethod
    def _fuzzy_get(args: Dict[str, Any], canonical: str, aliases: tuple, default: str = "") -> str:
        """Get a value by canonical key, falling back to aliases for weaker models."""
        val = args.get(canonical)
        if val is not None:
            return str(val)
        for alias in aliases:
            val = args.get(alias)
            if val is not None:
                return str(val)
        # Last resort: if there's only one string-valued arg, use it
        if canonical == "text" and not default:
            str_vals = [str(v) for k, v in args.items() if isinstance(v, str) and len(str(v)) > 5]
            if len(str_vals) == 1:
                return str_vals[0]
        return default

    def _exec_add_memory(self, args: Dict[str, Any]) -> str:
        text = self._fuzzy_get(args, "text", (
            "content", "memory_text", "memory_content", "message", "user_message",
            "description", "summary", "note", "observation", "memory",
        ))
        image_token = self._fuzzy_get(args, "image", ("image_token", "img", "image_id"))
        image_caption = self._fuzzy_get(args, "image_caption", ("caption", "image_description"))
        evidence_ids_raw = self._fuzzy_get(args, "evidence_ids", (
            "evidence", "source_ids", "msg_ids", "message_ids", "ids",
        ), default="[]")

        try:
            evidence_ids = json.loads(evidence_ids_raw)
        except (json.JSONDecodeError, TypeError):
            evidence_ids = []

        image_path: Optional[str] = None
        if image_token:
            image_path = self._image_manager.token_to_image(image_token)

        mem = SemanticMemory(
            text=text,
            image_caption=image_caption or None,
            image_path=image_path,
            evidence_ids=evidence_ids,
        )
        mem_id = self._semantic.add(mem)
        return f"Memory added with id={mem_id}."

    def _exec_delete_memory(self, args: Dict[str, Any]) -> str:
        mem_id = int(args.get("memory_id") or args.get("id") or args.get("mem_id") or -1)
        self._semantic.delete(mem_id)
        return f"Memory id={mem_id} deleted."

    def _execute_tool(self, tool_name: str, args: Dict[str, Any]) -> str:
        dispatch = {
            "search_semantic_memories": self._exec_search,
            "fetch_raw_messages": self._exec_fetch_raw,
            "fetch_raw_messages_by_time": self._exec_fetch_by_time,
            "add_memory": self._exec_add_memory,
            "delete_memory": self._exec_delete_memory,
        }
        # Normalize tool name for weaker models that hallucinate names
        _name_aliases = {
            "create": "add_memory", "CREATE": "add_memory",
            "create_memory": "add_memory", "CREATE_MEMORY": "add_memory",
            "save_memory": "add_memory", "store_memory": "add_memory",
            "update_memory": "add_memory",  # treat update as add (upsert)
            "delete": "delete_memory", "DELETE": "delete_memory",
            "remove_memory": "delete_memory",
            "search": "search_semantic_memories", "SEARCH": "search_semantic_memories",
            "query_memory": "search_semantic_memories",
            "fetch_raw": "fetch_raw_messages",
            "fetch_by_time": "fetch_raw_messages_by_time",
            "NONE": None, "None": None, "noop": None, "no_op": None,
        }
        resolved = tool_name
        if tool_name not in dispatch:
            resolved = _name_aliases.get(tool_name, tool_name)
        if resolved is None:
            return "No action taken."
        fn = dispatch.get(resolved)
        if fn is None:
            return f"Unknown tool: {tool_name}"
        try:
            return fn(args)
        except Exception as exc:  # noqa: BLE001
            return f"Tool error ({tool_name}): {exc}"

    # ---- ReAct loop ----

    def _react_loop(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
    ) -> str:
        """
        Generic ReAct loop. Loops until LLM produces text (no tool calls)
        or max_iterations is reached.
        Faithful to official MemoryManager LangGraph workflow.
        parallel_tool_calls=False matches official bind_tools(parallel_tool_calls=False).
        """
        for _ in range(self._max_iter):
            # Retry with exponential backoff for rate limit errors
            max_retries = 12
            for attempt in range(max_retries):
                try:
                    response = self._client.chat.completions.create(
                        model=self._model,
                        messages=messages,
                        tools=tools,
                        tool_choice="auto",
                        parallel_tool_calls=False,
                    )
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

            # Append assistant message with tool calls
            messages.append(msg.model_dump(exclude_unset=True))

            # Execute tools and append results
            for tc in msg.tool_calls:
                raw_args = tc.function.arguments or "{}"
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    # Small models sometimes emit malformed JSON; try to
                    # salvage by decoding only the first JSON object.
                    decoder = json.JSONDecoder()
                    try:
                        args, _ = decoder.raw_decode(raw_args.strip())
                    except json.JSONDecodeError:
                        args = {}
                result = self._execute_tool(tc.function.name, args)
                if os.environ.get("M2A_DEBUG"):
                    print(f"[M2A-DBG] tool={tc.function.name} args={tc.function.arguments[:200]} result={result[:120]}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    }
                )

        return ""

    # ---- public interface ----

    def query(
        self,
        query_text: str,
        query_image_paths: Optional[List[str]] = None,
    ) -> str:
        """
        QUERY mode: retrieve relevant information from memory and return answer.
        Faithful to official handle_query flow.
        """
        context = self._prepare_context()
        sys_prompt = QUERY_SYSTEM_PROMPT.format(context=context)

        if query_image_paths:
            user_content: Any = self._image_manager.format_content(
                query_text, query_image_paths
            )
        else:
            user_content = query_text

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        return self._react_loop(messages, QUERY_TOOLS)

    def update(
        self,
        content: str,
        image_paths: Optional[List[str]] = None,
    ) -> None:
        """
        UPDATE mode: analyze content and create/delete/update semantic memories.
        Faithful to official handle_update flow.
        """
        context = self._prepare_context()
        sys_prompt = UPDATE_SYSTEM_PROMPT.format(context=context)

        if image_paths:
            user_content: Any = self._image_manager.format_content(content, image_paths)
        else:
            user_content = content

        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_content},
        ]
        self._react_loop(messages, ALL_TOOLS)
