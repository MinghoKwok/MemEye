import base64
import json
import mimetypes
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import error, request


def require_api_key(api_key: str = "", api_key_env: str = "") -> str:
    if api_key:
        return api_key
    if api_key_env:
        value = os.environ.get(api_key_env, "").strip()
        if value:
            return value
        raise RuntimeError(f"Missing API key in environment variable: {api_key_env}")
    raise RuntimeError("Missing API key configuration.")


def encode_image_data_url(image_path: str, max_long_edge: int = 768) -> str:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"

    raw = path.read_bytes()

    # Resize large images to keep total payload under API limits
    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(raw))
        w, h = img.size
        if max(w, h) > max_long_edge:
            scale = max_long_edge / max(w, h)
            img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=80)
            raw = buf.getvalue()
            mime_type = "image/jpeg"
    except Exception:
        pass  # fall back to original bytes if PIL unavailable

    data = base64.b64encode(raw).decode("ascii")
    return f"data:{mime_type};base64,{data}"


def encode_image_inline(image_path: str) -> Dict[str, str]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(path.name)
    if not mime_type:
        mime_type = "application/octet-stream"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"mime_type": mime_type, "data": data}


def _parse_retry_after(body: str) -> Optional[float]:
    """Extract seconds from '...try again in 1.933s...' in a 429 body."""
    m = re.search(r"try again in ([\d.]+)s", body)
    return float(m.group(1)) if m else None


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: int = 60,
    max_retries: int = 20,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        req = request.Request(
            url,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except error.HTTPError as exc:
            details = exc.read().decode("utf-8", errors="replace")
            if exc.code in (429, 502, 503, 504) and attempt < max_retries - 1:
                wait = (_parse_retry_after(details) or (2 ** attempt)) + 2.0
                label = "rate limit" if exc.code == 429 else f"server error {exc.code}"
                print(f"[WARN] {label} — waiting {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
                last_err = exc
                continue
            raise RuntimeError(f"HTTP {exc.code} calling {url}: {details}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"Network error calling {url}: {exc}") from exc
    raise RuntimeError(f"Request failed after {max_retries} attempts: {last_err}")
