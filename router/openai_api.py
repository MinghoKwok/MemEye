from typing import Any, Dict, List, Optional

from .base import BaseRouter
from .http_utils import encode_image_data_url, post_json, require_api_key

# Many providers cap images per request (e.g. OpenRouter/Parasail: 30).
_MAX_IMAGES_PER_REQUEST = 30


class OpenAIAPIRouter(BaseRouter):
    def __init__(
        self,
        model: str,
        api_key: str = "",
        api_key_env: str = "OPENAI_API_KEY",
        base_url: str = "https://api.openai.com/v1",
        max_new_tokens: int = 128,
        timeout: int = 90,
        system_prompt: str = "",
        max_images: int = _MAX_IMAGES_PER_REQUEST,
    ) -> None:
        self.model = model
        self.api_key = require_api_key(api_key=api_key, api_key_env=api_key_env)
        self.base_url = base_url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.system_prompt = system_prompt
        self.max_images = max_images

    @staticmethod
    def _truncate_images(
        history_messages: List[Dict[str, Any]],
        question_images: List[str],
        max_images: int,
    ) -> List[Dict[str, Any]]:
        """Drop oldest history images so the total stays within *max_images*.

        Question-turn images are always kept. History images are removed
        from the earliest messages first (FIFO).
        """
        q_count = len(question_images)
        budget = max(0, max_images - q_count)

        # Count total history images
        total = sum(len(m.get("images", []) or []) for m in history_messages)
        if total <= budget:
            return history_messages

        print(
            f"[WARN] Truncating history images from {total} to {budget} "
            f"(max_images={max_images}, question_images={q_count})"
        )

        # Walk from the front (oldest) and strip images until within budget
        to_drop = total - budget
        result: List[Dict[str, Any]] = []
        for msg in history_messages:
            imgs = list(msg.get("images", []) or [])
            if to_drop > 0 and imgs:
                n_remove = min(len(imgs), to_drop)
                imgs = imgs[n_remove:]
                to_drop -= n_remove
            new_msg = dict(msg)
            new_msg["images"] = imgs
            result.append(new_msg)
        return result

    def _to_messages(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        q_imgs = question_images or []
        history_messages = self._truncate_images(
            history_messages, q_imgs, self.max_images,
        )

        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": self.system_prompt,
            }
        ]

        for msg in history_messages:
            content: List[Dict[str, Any]] = []
            text = str(msg.get("text", "")).strip()
            if text:
                content.append({"type": "text", "text": text})
            for image_path in msg.get("images", []) or []:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": encode_image_data_url(image_path), "detail": "high"},
                    }
                )
            if content:
                messages.append({"role": msg.get("role", "user"), "content": content})

        final_content: List[Dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "Answer based on the conversation and images above. "
                    "Be concise and factual.\n"
                    f"Question: {question}"
                ),
            }
        ]
        for image_path in question_images or []:
            final_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": encode_image_data_url(image_path), "detail": "high"},
                }
            )
        messages.append({"role": "user", "content": final_content})
        return messages

    def answer(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._to_messages(history_messages, question, question_images),
            **({
                "max_completion_tokens": self.max_new_tokens
            } if any(self.model.startswith(p) for p in ("gpt-5", "o3", "o4")) else {
                "max_tokens": self.max_new_tokens
            }),
            "temperature": 0,
        }
        response = post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload=payload,
            timeout=self.timeout,
        )
        try:
            usage = response.get("usage", {})
            self.last_usage = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
            return str(response["choices"][0]["message"]["content"]).strip()
        except Exception as exc:
            raise RuntimeError(f"Unexpected OpenAI response shape: {response}") from exc
