import argparse
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

from openai import OpenAI


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTION_PROMPT = (
    "Generate a detailed caption for this image. Include all visually observable "
    "information that may be useful for answering future memory questions: objects, "
    "people or characters, their identities or distinguishing attributes, colors, "
    "shapes, textures, positions, spatial relations, or changes. Be specific and "
    "exhaustive, but do not infer information that is not visible."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Caption a Mem-Gallery-format dialog dataset and write image_caption fields in-place."
    )
    parser.add_argument("--dialog-json", required=True, help="Path to the dialog JSON file.")
    parser.add_argument("--image-root", default="data/image", help="Root directory for dataset images.")
    parser.add_argument("--model", default="gpt-5.1", help="Vision-capable OpenAI model for one-time captioning.")
    parser.add_argument("--output-json", default="", help="Optional output path. Defaults to in-place overwrite.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate captions even if image_caption already exists.")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print summary without writing changes.")
    return parser.parse_args()


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    cwd_candidate = (Path.cwd() / path).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    return (REPO_ROOT / path).resolve()


def resolve_image_path(raw_image_path: str, dialog_json_path: Path, image_root: Path) -> Path:
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
        candidates.append((image_root / normalized).resolve())
        candidates.append((image_root / source_path.name).resolve())

    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Could not resolve image path '{raw_image_path}' from '{dialog_json_path}'.")


def _mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(path.name)
    return mime_type or "image/jpeg"


def caption_image(client: OpenAI, model: str, image_path: Path) -> str:
    import base64

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
        response = client.responses.create(
            model=model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": CAPTION_PROMPT,
                        },
                        {
                            "type": "input_image",
                            "image_url": f"data:{_mime_type(image_path)};base64,{encoded}",
                        },
                    ],
                }
            ],
        )
    text = getattr(response, "output_text", "")
    return str(text).strip()


def caption_image_compat(client: OpenAI, model: str, image_path: Path) -> str:
    import base64

    with image_path.open("rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": CAPTION_PROMPT,
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{_mime_type(image_path)};base64,{encoded}"},
                    },
                ],
            }
        ],
        temperature=0.0,
    )
    content = response.choices[0].message.content
    return str(content).strip()


def generate_caption(client: OpenAI, model: str, image_path: Path) -> str:
    try:
        return caption_image_compat(client, model, image_path)
    except Exception:
        return caption_image(client, model, image_path)


def process_dataset(
    payload: Dict[str, Any],
    dialog_json_path: Path,
    image_root: Path,
    model: str,
    overwrite: bool,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY must be set in the environment before running caption preprocessing.")
    client = OpenAI(api_key=api_key)

    sessions = payload.get("multi_session_dialogues", []) or []
    for session in sessions:
        dialogues = session.get("dialogues", []) or []
        for dialogue in dialogues:
            input_images = dialogue.get("input_image", []) or []
            if not input_images:
                continue

            current_captions = dialogue.get("image_caption", []) or []
            if current_captions and len(current_captions) == len(input_images) and not overwrite:
                continue

            captions: List[str] = []
            for raw_image_path in input_images:
                image_path = resolve_image_path(str(raw_image_path), dialog_json_path, image_root)
                captions.append(generate_caption(client, model, image_path))
            dialogue["image_caption"] = captions
    return payload


def main() -> None:
    args = parse_args()
    dialog_json_path = resolve_path(args.dialog_json)
    image_root = resolve_path(args.image_root)
    output_path = resolve_path(args.output_json) if args.output_json else dialog_json_path

    with dialog_json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    updated = process_dataset(
        payload=payload,
        dialog_json_path=dialog_json_path,
        image_root=image_root,
        model=args.model,
        overwrite=args.overwrite,
    )

    if args.dry_run:
        total_captioned = 0
        for session in updated.get("multi_session_dialogues", []) or []:
            for dialogue in session.get("dialogues", []) or []:
                total_captioned += len(dialogue.get("image_caption", []) or [])
        print(f"Validated caption payload. Total captions present: {total_captioned}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(updated, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"Wrote captioned dataset to {output_path}")


if __name__ == "__main__":
    main()
