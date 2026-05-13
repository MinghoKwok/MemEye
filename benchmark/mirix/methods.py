from __future__ import annotations

from typing import Any, Dict, Optional

from ..methods import HistoryMethod
from .official import OfficialMIRIXMethod


def get_mirix_method(method_name: str, config: Optional[Dict[str, Any]] = None) -> HistoryMethod:
    if method_name != OfficialMIRIXMethod.name:
        raise ValueError(
            f"Unsupported MIRIX method: {method_name!r}. "
            "Only the official mirix adapter is supported."
        )
    return OfficialMIRIXMethod(config=config)
