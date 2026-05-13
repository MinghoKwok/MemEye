from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseRouter(ABC):
    system_prompt: str = ""
    last_usage: Dict[str, int] = {}  # populated after each answer() call

    @abstractmethod
    def answer(
        self,
        history_messages: List[Dict[str, Any]],
        question: str,
        question_images: Optional[List[str]] = None,
    ) -> str:
        """
        Generate an answer given multimodal history messages and a question.

        history_messages format:
        [
          {
            "role": "user" | "assistant",
            "text": "...",
            "images": ["/abs/path/a.jpg", ...]   # optional
          },
          ...
        ]

        question_images is an optional list of image paths attached to the
        question itself (e.g. a face crop the model must identify against
        avatars seen in past chat screenshots). Routers must include these
        as additional image parts on the final user turn so multimodal
        models can see them. Routers without image support should ignore
        the field; tasks that depend on it will fail intentionally.
        """
        raise NotImplementedError
