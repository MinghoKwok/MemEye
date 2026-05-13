from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import load_json


def resolve_image_path(
    raw_image_path: str, dialog_json_path: Path, image_root: Optional[Path]
) -> str:
    cleaned = raw_image_path.replace("file://", "")
    source_path = Path(cleaned)
    candidates: List[Path] = []

    if source_path.is_absolute():
        candidates.append(source_path)
    else:
        candidates.append((dialog_json_path.parent / source_path).resolve())
        candidates.append((dialog_json_path.parent.parent / source_path).resolve())

        normalized = cleaned
        for prefix in ("../image/", "./image/", "image/", "data/image/"):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]
                break

        if image_root is not None:
            candidates.append((image_root / normalized).resolve())
            candidates.append((image_root / source_path.name).resolve())

        default_image_root = (dialog_json_path.parent.parent / "image").resolve()
        candidates.append((default_image_root / normalized).resolve())

    deduped: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(candidate)

    for candidate in deduped:
        if candidate.exists():
            return str(candidate)

    raise FileNotFoundError(
        "Could not resolve image path "
        f"'{raw_image_path}' from dialog file '{dialog_json_path}'. "
        f"Tried: {[str(p) for p in deduped]}"
    )


def build_rounds(
    dialog_data: Dict[str, Any], dialog_json_path: Path, image_root: Optional[Path]
) -> Dict[str, Dict[str, Any]]:
    rounds: Dict[str, Dict[str, Any]] = {}
    for sess in dialog_data.get("multi_session_dialogues", []):
        sid = sess.get("session_id", "")
        for d in sess.get("dialogues", []):
            rid = d.get("round", "")
            images: List[str] = []
            for rel in d.get("input_image", []) or []:
                images.append(resolve_image_path(rel, dialog_json_path, image_root))
            rounds[rid] = {
                "session_id": sid,
                "round_id": rid,
                "user": d.get("user", ""),
                "assistant": d.get("assistant", ""),
                "images": images,
                "raw": d,
            }
    return rounds


def _string_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    return [str(value).strip() for value in values if str(value).strip()]


def round_image_captions(round_payload: Dict[str, Any]) -> List[str]:
    raw = round_payload.get("raw", {}) or {}
    return _string_list(raw.get("image_caption", []) or [])


def round_image_ids(round_payload: Dict[str, Any]) -> List[str]:
    raw = round_payload.get("raw", {}) or {}
    if not isinstance(raw.get("image_id", []), list):
        return []
    image_ids = raw.get("image_id", []) or []
    return [str(value).strip() for value in image_ids]


def build_caption_text(round_payload: Dict[str, Any]) -> str:
    captions = round_image_captions(round_payload)
    if not captions:
        return ""

    image_ids = round_image_ids(round_payload)
    lines: List[str] = []
    for idx, caption in enumerate(captions):
        image_id = image_ids[idx] if idx < len(image_ids) else ""
        lines.extend(
            [
                "image:",
                f"image_id: {image_id}",
                f"image_caption: {caption}",
            ]
        )
    return "\n".join(lines).strip()


def build_round_retrieval_text(round_payload: Dict[str, Any], modality: str) -> str:
    parts = [
        str(round_payload.get("user", "")).strip(),
        str(round_payload.get("assistant", "")).strip(),
    ]
    if modality == "text_only":
        caption_text = " ".join(round_image_captions(round_payload))
        if caption_text:
            parts.append(caption_text)
    return " ".join(part for part in parts if part)


def validate_text_only_captions(rounds: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    invalid_rounds: List[str] = []
    validated_rounds = 0
    for round_id, payload in rounds.items():
        raw = payload.get("raw", {}) or {}
        images = raw.get("input_image", []) or []
        if not images:
            continue
        validated_rounds += 1
        captions = round_image_captions(payload)
        if len(captions) != len(images):
            invalid_rounds.append(round_id)
            continue
        if any(not caption for caption in captions):
            invalid_rounds.append(round_id)

    if invalid_rounds:
        raise ValueError(
            "Text-only methods require Mem-Gallery-style image captions. "
            "Add valid image_caption entries for image-bearing rounds before running. "
            "Missing/invalid image_caption for rounds: "
            + ", ".join(invalid_rounds[:10])
            + ("..." if len(invalid_rounds) > 10 else "")
        )

    return {
        "caption_validation_passed": True,
        "caption_validation_round_count": validated_rounds,
    }


def session_order(dialog_data: Dict[str, Any]) -> List[str]:
    return [s.get("session_id", "") for s in dialog_data.get("multi_session_dialogues", [])]


def get_qas(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    for key in ("human-annotated QAs", "human_annotated_qas", "qas"):
        value = data.get(key)
        if isinstance(value, list):
            return value
    return []


def history_from_round_ids(
    session: Dict[str, Any],
    rounds: Dict[str, Dict[str, Any]],
    allowed_round_ids: Optional[set[str]] = None,
    modality: str = "multimodal",
) -> List[Dict[str, Any]]:
    history: List[Dict[str, Any]] = []
    for d in session.get("dialogues", []):
        rid = d.get("round", "")
        if allowed_round_ids is not None and rid not in allowed_round_ids:
            continue
        r = rounds.get(rid, {})
        user_text = r.get("user", "")
        assistant_text = r.get("assistant", "")
        images = list(r.get("images", []) or [])
        if modality == "text_only":
            caption_text = build_caption_text(r)
            if caption_text:
                user_text = f"{user_text}\n{caption_text}".strip() if user_text else caption_text
            images = []
        elif modality == "no_visual":
            # No images, no captions — pure dialogue text for leakage testing
            images = []
        elif modality == "multimodal":
            images = images
        else:
            raise ValueError(f"Unsupported history modality: {modality}")

        if user_text or images:
            history.append({"role": "user", "text": user_text, "images": images, "round_id": rid})
        if assistant_text:
            history.append({"role": "assistant", "text": assistant_text, "images": [], "round_id": rid})
    return history


class MemoryBenchmarkDataset:
    def __init__(self, dialog_json_path: Path, image_root: Optional[Path] = None) -> None:
        self.dialog_json_path = dialog_json_path
        self.image_root = image_root
        self.data = load_json(dialog_json_path)
        self.rounds = build_rounds(self.data, dialog_json_path, image_root)
        self.qas = get_qas(self.data)
        self.sessions = self.data.get("multi_session_dialogues", [])

    def session_order(self) -> List[str]:
        return session_order(self.data)

    def get_session(self, session_id: str) -> Dict[str, Any]:
        return next(s for s in self.sessions if s.get("session_id") == session_id)

    def iter_qas(self, limit: int = 0) -> List[Dict[str, Any]]:
        if limit > 0:
            return self.qas[:limit]
        return self.qas

    def resolve_question_images(self, qa: Dict[str, Any]) -> List[str]:
        """Resolve QA-level image paths (face lookups, etc.) to absolute paths.

        Tasks that don't use question-level images return an empty list, so
        this is a no-op for legacy QAs. Accepts both ``question_image`` (str
        or list) and ``question_images`` (list) for forward-compat.
        """
        raw: List[str] = []
        single = qa.get("question_image")
        if isinstance(single, str) and single.strip():
            raw.append(single.strip())
        elif isinstance(single, list):
            raw.extend([str(p).strip() for p in single if str(p).strip()])
        plural = qa.get("question_images")
        if isinstance(plural, list):
            raw.extend([str(p).strip() for p in plural if str(p).strip()])
        if not raw:
            return []
        resolved: List[str] = []
        for path in raw:
            resolved.append(resolve_image_path(path, self.dialog_json_path, self.image_root))
        return resolved


# Backward-compatible alias for older imports.
PittAdsDataset = MemoryBenchmarkDataset
