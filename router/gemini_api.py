from typing import Any, Dict, List, Optional

from .base import BaseRouter
from .http_utils import encode_image_inline, post_json, require_api_key


class GeminiAPIRouter(BaseRouter):
    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_key_env: str = "GEMINI_API_KEY",
        base_url: str = "https://generativelanguage.googleapis.com/v1beta",
        max_new_tokens: int = 128,
        timeout: int = 90,
        system_prompt: str = "",
    ) -> None:
        self.model = model
        self.api_key = require_api_key(api_key=api_key, api_key_env=api_key_env)
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt

    def _to_contents(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        contents: List[Dict[str, Any]] = []
        for msg in history_messages:
            parts: List[Dict[str, Any]] = []
            text = str(msg.get("text", "")).strip()
            if text:
                parts.append({"text": text})
            for image_path in msg.get("images", []) or []:
                parts.append({"inline_data": encode_image_inline(image_path)})
            if not parts:
                continue
            role = "model" if msg.get("role") == "assistant" else "user"
            contents.append({"role": role, "parts": parts})

        final_parts: List[Dict[str, Any]] = [
            {
                "text": (
                    "Answer based on the conversation and images above. "
                    "Be concise and factual.\n"
                    f"Question: {question}"
                )
            }
        ]
        for image_path in question_images or []:
            final_parts.append({"inline_data": encode_image_inline(image_path)})
        contents.append({"role": "user", "parts": final_parts})
        return contents

    def answer(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "systemInstruction": {
                "parts": [{"text": self.system_prompt}]
            },
            "contents": self._to_contents(history_messages, question, question_images),
            "generationConfig": {
                "temperature": 0,
                "maxOutputTokens": self.max_new_tokens,
            },
        }
        response = post_json(
            url=f"{self.base_url}/models/{self.model}:generateContent?key={self.api_key}",
            headers={},
            payload=payload,
            timeout=self.timeout,
        )
        try:
            parts = response["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
            return text.strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected Gemini response shape: {response}") from exc
