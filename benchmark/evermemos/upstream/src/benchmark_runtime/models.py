"""Compact benchmark-owned data model layer for EverMemOS evaluation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union

from bson import ObjectId

from common_utils.datetime_utils import get_now_with_timezone, to_iso_format


class ScenarioType(str, Enum):
    SOLO = "solo"
    TEAM = "team"


class RawDataType(Enum):
    CONVERSATION = "Conversation"
    AGENTCONVERSATION = "AgentConversation"


class ParentType(str, Enum):
    MEMCELL = "memcell"
    EPISODE = "episode"


class MessageSenderRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

    @classmethod
    def from_string(cls, role_str: Optional[str]) -> Optional["MessageSenderRole"]:
        if not role_str:
            return None
        role_lower = role_str.lower()
        for role in cls:
            if role.value == role_lower:
                return role
        return None

    @classmethod
    def is_valid(cls, role_str: Optional[str]) -> bool:
        if not role_str:
            return True
        return cls.from_string(role_str) is not None


class MemoryType(str, Enum):
    PROFILE = "profile"
    EPISODIC_MEMORY = "episodic_memory"
    FORESIGHT = "foresight"
    ATOMIC_FACT = "atomic_fact"
    RAW_MESSAGE = "raw_message"
    AGENT_MEMORY = "agent_memory"
    AGENT_CASE = "agent_case"
    AGENT_SKILL = "agent_skill"

    @classmethod
    def from_string(cls, value: Optional[str]) -> Optional["MemoryType"]:
        if not value:
            return None
        normalized = str(value).strip().lower()
        for item in cls:
            if item.value == normalized:
                return item
        return None


def get_text_from_content_items(content_items: Any) -> str:
    if isinstance(content_items, str):
        return content_items
    if not isinstance(content_items, list):
        return ""
    texts: List[str] = []
    for item in content_items:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "text":
            text = item.get("text") or item.get("content", "")
            if text:
                texts.append(text)
            continue
        item_type = (item.get("type") or "file").upper()
        name = item.get("name", "")
        parsed_summary = item.get("parsed_summary")
        if name and parsed_summary:
            texts.append(f"[{item_type}: {name} | Summary: {parsed_summary}]")
        elif name:
            texts.append(f"[{item_type}: {name}]")
        elif parsed_summary:
            texts.append(f"[{item_type} | Summary: {parsed_summary}]")
        else:
            texts.append(f"[{item_type}]")
    return " ".join(texts) if texts else ""


@dataclass
class RawData:
    content: Dict[str, Any]
    data_id: str
    data_type: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def _serialize_value(self, value: Any) -> Any:
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, ObjectId):
            return str(value)
        if isinstance(value, dict):
            return {k: self._serialize_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._serialize_value(item) for item in value]
        if hasattr(value, "__dict__"):
            return self._serialize_value(value.__dict__)
        return value

    def _is_datetime_field(self, field_name: str) -> bool:
        if not isinstance(field_name, str):
            return False
        exact_fields = {
            "timestamp",
            "createtime",
            "updatetime",
            "create_time",
            "update_time",
            "sent_timestamp",
            "received_timestamp",
            "create_timestamp",
            "last_update_timestamp",
            "modify_timestamp",
            "created_at",
            "updated_at",
            "jointime",
            "leavetime",
            "lastonlinetime",
            "sync_time",
            "processed_at",
            "start_time",
            "end_time",
            "event_time",
            "build_timestamp",
            "datetime",
            "created",
            "updated",
        }
        lowered = field_name.lower()
        if field_name in exact_fields or lowered in exact_fields:
            return True
        exclusions = {
            "runtime",
            "timeout",
            "timeline",
            "timestamp_format",
            "time_zone",
            "time_limit",
            "timestamp_count",
            "timestamp_enabled",
            "time_sync",
            "playtime",
            "lifetime",
            "uptime",
            "downtime",
        }
        if field_name in exclusions or lowered in exclusions:
            return False
        if any(field_name.endswith(s) or lowered.endswith(s) for s in ("_time", "_timestamp", "_at", "_date")):
            return True
        if field_name.endswith("Time") and not field_name.endswith("runtime"):
            return True
        if field_name.endswith("Timestamp"):
            return True
        return False

    def _is_iso_datetime(self, value: str) -> bool:
        return isinstance(value, str) and len(value) >= 19 and bool(
            re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", value)
        )

    def _deserialize_value(self, value: Any, field_name: str = "") -> Any:
        if isinstance(value, str):
            if self._is_datetime_field(field_name) and self._is_iso_datetime(value):
                from common_utils.datetime_utils import from_iso_format

                try:
                    return from_iso_format(value)
                except (ValueError, ImportError):
                    return value
            return value
        if isinstance(value, dict):
            return {k: self._deserialize_value(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [self._deserialize_value(item, field_name) for item in value]
        return value

    def to_json(self) -> str:
        data = {
            "content": self._serialize_value(self.content),
            "data_id": self.data_id,
            "data_type": self.data_type,
            "metadata": self._serialize_value(self.metadata) if self.metadata else None,
        }
        return json.dumps(data, ensure_ascii=False, separators=(",", ":"))

    @classmethod
    def from_json_str(cls, json_str: str) -> "RawData":
        instance = cls(content={}, data_id="")
        data = json.loads(json_str)
        if "content" not in data or "data_id" not in data:
            raise ValueError("RawData JSON missing required fields")
        return cls(
            content=instance._deserialize_value(data["content"], "content"),
            data_id=data["data_id"],
            data_type=data.get("data_type"),
            metadata=instance._deserialize_value(data.get("metadata"), "metadata"),
        )


@dataclass
class MemCell:
    user_id_list: List[str]
    original_data: List[Dict[str, Any]]
    timestamp: datetime
    event_id: Optional[str] = None
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    sender_ids: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    _conversation_data_cache: Optional[List[Dict[str, Any]]] = field(
        default=None, init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if not self.original_data:
            raise ValueError("original_data is required")

    @staticmethod
    def _is_intermediate_agent_step(msg: Dict[str, Any]) -> bool:
        role = msg.get("role", "")
        if role == "tool":
            return True
        if role == "assistant" and msg.get("tool_calls"):
            return True
        return False

    @property
    def conversation_data(self) -> List[Dict[str, Any]]:
        if self._conversation_data_cache is not None:
            return self._conversation_data_cache
        if self.type != RawDataType.AGENTCONVERSATION:
            self._conversation_data_cache = self.original_data
        else:
            self._conversation_data_cache = [
                item
                for item in self.original_data
                if not self._is_intermediate_agent_step(
                    item.get("message", item) if isinstance(item, dict) else item
                )
            ]
        return self._conversation_data_cache

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "user_id_list": self.user_id_list,
            "original_data": self.original_data,
            "timestamp": to_iso_format(self.timestamp),
            "group_id": self.group_id,
            "participants": self.participants,
            "sender_ids": self.sender_ids,
            "type": self.type.value if self.type else None,
        }


@dataclass
class BaseMemory:
    memory_type: Union[MemoryType, str]
    user_id: str
    timestamp: datetime
    ori_event_id_list: Optional[List[str]] = None
    group_id: Optional[str] = None
    participants: Optional[List[str]] = None
    sender_ids: Optional[List[str]] = None
    type: Optional[RawDataType] = None
    keywords: Optional[List[str]] = None
    linked_entities: Optional[List[str]] = None
    user_name: Optional[str] = None
    extend: Optional[Dict[str, Any]] = None
    vector_model: Optional[str] = None
    vector: Optional[List[float]] = None
    id: Optional[str] = None
    score: Optional[float] = None
    original_data: Optional[List[Dict[str, Any]]] = None

    def _format_timestamp(self) -> Optional[str]:
        if not self.timestamp:
            return None
        if isinstance(self.timestamp, str):
            return self.timestamp or None
        try:
            return to_iso_format(self.timestamp)
        except Exception:
            return str(self.timestamp) if self.timestamp else None

    def to_dict(self) -> Dict[str, Any]:
        memory_type = self.memory_type.value if isinstance(self.memory_type, Enum) else self.memory_type
        return {
            "id": self.id,
            "memory_type": memory_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "timestamp": self._format_timestamp(),
            "group_id": self.group_id,
            "participants": self.participants,
            "sender_ids": self.sender_ids,
            "type": self.type.value if self.type else None,
            "keywords": self.keywords,
            "linked_entities": self.linked_entities,
            "score": self.score,
            "original_data": self.original_data,
            "extend": self.extend,
        }


@dataclass
class EpisodeMemory(BaseMemory):
    id: Optional[str] = None
    subject: Optional[str] = None
    summary: Optional[str] = None
    episode: Optional[str] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class AtomicFact(BaseMemory):
    time: Optional[str] = None
    atomic_fact: Optional[Union[str, List[str]]] = None
    fact_embeddings: Optional[List[List[float]]] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AtomicFact":
        return cls(
            memory_type=MemoryType.from_string(data.get("memory_type")) or MemoryType.ATOMIC_FACT,
            user_id=data.get("user_id", ""),
            timestamp=data.get("timestamp"),
            time=data.get("time", ""),
            atomic_fact=data.get("atomic_fact", []),
            fact_embeddings=data.get("fact_embeddings"),
            parent_type=data.get("parent_type"),
            parent_id=data.get("parent_id"),
        )


@dataclass
class Foresight(BaseMemory):
    foresight: Optional[str] = None
    evidence: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration_days: Optional[int] = None
    parent_type: Optional[str] = None
    parent_id: Optional[str] = None


@dataclass
class ProfileMemory(BaseMemory):
    explicit_info: List[Dict[str, Any]] = field(default_factory=list)
    implicit_traits: List[Dict[str, Any]] = field(default_factory=list)
    last_updated: Optional[datetime] = None
    processed_episode_ids: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.memory_type = MemoryType.PROFILE
        if self.last_updated is None:
            self.last_updated = datetime.now().astimezone()

    @classmethod
    def from_dict(
        cls, data: Dict[str, Any], user_id: str = "", group_id: str = ""
    ) -> "ProfileMemory":
        last_updated = data.get("last_updated")
        if isinstance(last_updated, str):
            last_updated = datetime.fromisoformat(last_updated)
        return cls(
            memory_type=MemoryType.PROFILE,
            user_id=user_id or data.get("user_id", ""),
            group_id=group_id or data.get("group_id", ""),
            timestamp=datetime.now().astimezone(),
            ori_event_id_list=data.get("ori_event_id_list", []),
            explicit_info=data.get("explicit_info", []),
            implicit_traits=data.get("implicit_traits", []),
            last_updated=last_updated,
            processed_episode_ids=data.get("processed_episode_ids", []),
        )

    def total_items(self) -> int:
        return len(self.explicit_info) + len(self.implicit_traits)

    def get_all_source_ids(self) -> set:
        ids = set()
        for item in self.explicit_info + self.implicit_traits:
            for source in item.get("sources", []):
                source = str(source)
                if "|" in source:
                    source = source.rsplit("|", 1)[-1].strip()
                if source:
                    ids.add(source)
        return ids

    def to_readable_document(self) -> str:
        lines = [
            "=" * 50,
            "User Profile Document",
            f"Last Updated: {self.last_updated.strftime('%Y-%m-%d %H:%M') if self.last_updated else 'N/A'}",
            f"Total {self.total_items()} items (Explicit: {len(self.explicit_info)}, Implicit: {len(self.implicit_traits)})",
            "=" * 50,
        ]
        if self.explicit_info:
            lines.append("\n[Explicit Info]")
            categories: Dict[str, list] = {}
            for info in self.explicit_info:
                categories.setdefault(info.get("category", ""), []).append(info)
            for cat, infos in categories.items():
                lines.append(f"  [{cat}]")
                for info in infos:
                    desc = info.get("description", "")
                    evidence = info.get("evidence", "")
                    lines.append(f"    - {desc}" + (f" (evidence: {evidence})" if evidence else ""))
        if self.implicit_traits:
            lines.append("\n[Implicit Traits]")
            for idx, trait in enumerate(self.implicit_traits, 1):
                lines.append(f"  {idx}. {trait.get('trait', '')}")
                lines.append(f"     {trait.get('description', '')}")
                lines.append(f"     - basis: {trait.get('basis', '')}")
                evidence = trait.get("evidence", "")
                if evidence:
                    lines.append(f"     - evidence: {evidence}")
        return "\n".join(lines)

    def to_readable_profile(self) -> str:
        lines: List[str] = []
        if self.explicit_info:
            lines.append("[Explicit Info]")
            categories: Dict[str, list] = {}
            for info in self.explicit_info:
                categories.setdefault(info.get("category", ""), []).append(info)
            for cat, infos in categories.items():
                lines.append(f"  {cat}:")
                for info in infos:
                    lines.append(f"    - {info.get('description', '')}")
        if self.implicit_traits:
            if lines:
                lines.append("")
            lines.append("[Implicit Traits]")
            for trait in self.implicit_traits:
                lines.append(f"  - {trait.get('trait', '')}: {trait.get('description', '')}")
        return "\n".join(lines) if lines else "No profile data yet."


RetrieveMemoryModel = Union[EpisodeMemory, AtomicFact, Foresight]

