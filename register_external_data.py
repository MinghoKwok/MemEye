import argparse
import re
from pathlib import Path

import yaml


def slugify_task_name(stem: str) -> str:
    text = re.sub(r"[^A-Za-z0-9]+", "_", stem).strip("_")
    return text.lower()


def build_task_config(dialog_json: Path, image_root: Path) -> dict:
    task_name = slugify_task_name(dialog_json.stem)
    return {
        "name": task_name,
        "dataset": {
            "dialog_json": str(dialog_json.resolve()),
            "image_root": str(image_root.resolve()),
        },
        "eval": {
            "mode": "open",
            "output_json": f"output/results_{task_name}.json",
            "max_questions": 0,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register an external MemEye data root by generating MemEye task configs."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="Path to the external data root. Expected to contain dialog/ and image/ subdirectories.",
    )
    parser.add_argument(
        "--task-config-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "config" / "tasks_external",
        help="Directory where generated task configs will be written.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing generated task configs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    dialog_root = data_root / "dialog"
    image_root = data_root / "image"

    if not dialog_root.exists():
        raise FileNotFoundError(f"Missing dialog directory: {dialog_root}")
    if not image_root.exists():
        raise FileNotFoundError(f"Missing image directory: {image_root}")

    args.task_config_dir.mkdir(parents=True, exist_ok=True)

    dialog_files = sorted(dialog_root.glob("*.json"))
    if not dialog_files:
        raise FileNotFoundError(f"No dialog JSON files found under: {dialog_root}")

    written = []
    skipped = []
    for dialog_json in dialog_files:
        task_name = slugify_task_name(dialog_json.stem)
        out_path = args.task_config_dir / f"{task_name}.yaml"
        if out_path.exists() and not args.overwrite:
            skipped.append(str(out_path))
            continue

        payload = build_task_config(dialog_json, image_root)
        with out_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=False)
        written.append(str(out_path))

    print(f"data_root={data_root}")
    print(f"dialog_files={len(dialog_files)}")
    print(f"written={len(written)}")
    print(f"skipped={len(skipped)}")
    for path in written:
        print(f"WROTE {path}")
    for path in skipped:
        print(f"SKIPPED {path}")


if __name__ == "__main__":
    main()
