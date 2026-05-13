"""
ImageManager: bidirectional mapping between image paths and <imageN> tokens.
Faithful to official M2A agent/stores/image_manager.py.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


class ImageManager:
    """
    Maps image file paths / URLs to integer-indexed tokens (<imageN>).
    Faithful to official M2A ImageManager interface.
    """

    def __init__(self) -> None:
        self._path_to_idx: Dict[str, int] = {}
        self._idx_to_path: Dict[int, str] = {}

    def image_to_token(self, image_path: str) -> str:
        """Register image if new, return its <imageN> token."""
        if image_path not in self._path_to_idx:
            idx = len(self._path_to_idx)
            self._path_to_idx[image_path] = idx
            self._idx_to_path[idx] = image_path
        return f"<image{self._path_to_idx[image_path]}>"

    def token_to_image(self, token: str) -> Optional[str]:
        """Resolve <imageN> token back to image path."""
        m = re.match(r"<image(\d+)>", token)
        if not m:
            return None
        return self._idx_to_path.get(int(m.group(1)))

    def format_content(
        self,
        text: str,
        image_paths: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Format text + images into OpenAI content list (images first, then text)."""
        content: List[Dict[str, Any]] = []
        for path in (image_paths or []):
            content.append(self._encode_image_content(path))
        if text:
            content.append({"type": "text", "text": text})
        return content

    @staticmethod
    def _encode_image_content(image_path: str) -> Dict[str, Any]:
        path = Path(image_path)
        if path.exists():
            b64 = base64.b64encode(path.read_bytes()).decode()
            ext = path.suffix.lstrip(".").lower() or "jpeg"
            return {
                "type": "image_url",
                "image_url": {"url": f"data:image/{ext};base64,{b64}"},
            }
        # Fall back to treating as URL
        return {"type": "image_url", "image_url": {"url": image_path}}
