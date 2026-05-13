import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from openai import OpenAI

from benchmark.common import load_json, write_json, write_jsonl
from benchmark.evaluator import llm_judge_score, summarize_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Posthoc LLM-as-a-judge scoring for locked benchmark predictions."
    )
    parser.add_argument(
        "--root",
        type=str,
        required=True,
        help="Root directory to scan for locked run folders containing predictions.jsonl.",
    )
    parser.add_argument(
        "--judge-model",
        type=str,
        default="gpt-5.2",
        help="Judge model name (default: gpt-5.2).",
    )
    parser.add_argument(
        "--judge-api-key",
        type=str,
        default="",
        help="API key for judge model. Falls back to OPENAI_API_KEY env var if omitted.",
    )
    parser.add_argument(
        "--judge-base-url",
        type=str,
        default="",
        help="Base URL for judge API. Leave empty for default OpenAI endpoint.",
    )
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-timeout", type=int, default=60)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rescore rows even if judge fields already exist.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def iter_run_dirs(root: Path) -> List[Path]:
    return sorted({path.parent for path in root.rglob("predictions.jsonl")})


def should_score_row(row: Dict[str, Any], force: bool) -> bool:
    if row.get("mode") != "open":
        return False
    if not force and row.get("judge") is not None:
        return False
    return True


def update_metrics_payload(
    metrics_payload: Dict[str, Any],
    rows: List[Dict[str, Any]],
    judge_model: str,
) -> Dict[str, Any]:
    updated = dict(metrics_payload)
    updated["summary"] = summarize_results(rows)
    judge_meta = dict(updated.get("judge", {}))
    judge_meta.update(
        {
            "enabled": True,
            "model": judge_model,
            "scored_open_rows": sum(1 for row in rows if row.get("mode") == "open" and row.get("judge") is not None),
        }
    )
    updated["judge"] = judge_meta
    return updated


def rewrite_output_summary(run_dir: Path, metrics_payload: Dict[str, Any]) -> None:
    for summary_path in run_dir.glob("results_*.json"):
        write_json(summary_path, metrics_payload)


def score_run_dir(
    run_dir: Path,
    client: OpenAI,
    prompt_template: str,
    judge_model: str,
    max_retries: int,
    timeout: int,
    force: bool,
) -> Dict[str, int]:
    predictions_path = run_dir / "predictions.jsonl"
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "config.json"
    if not predictions_path.exists() or not metrics_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"Missing required files in {run_dir}")

    _ = load_json(config_path)
    rows = load_jsonl(predictions_path)
    scored = 0
    skipped = 0

    for idx, row in enumerate(rows, start=1):
        if not should_score_row(row, force):
            skipped += 1
            continue
        result = llm_judge_score(
            question=str(row.get("question", "")),
            ground_truth=str(row.get("gt", "")),
            model_output=str(row.get("pred", "")),
            client=client,
            model_name=judge_model,
            prompt_template=prompt_template,
            max_retries=max_retries,
            timeout=timeout,
        )
        row["judge"] = result["score"]
        row["judge_reasoning"] = result.get("reasoning", "")
        scored += 1
        print(f"[JUDGE] {run_dir.name} row={idx} score={row['judge']}")

    metrics_payload = load_json(metrics_path)
    updated_metrics = update_metrics_payload(metrics_payload, rows, judge_model=judge_model)
    write_jsonl(predictions_path, rows)
    write_json(metrics_path, updated_metrics)
    rewrite_output_summary(run_dir, updated_metrics)
    return {"scored": scored, "skipped": skipped, "total": len(rows)}


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    if not root.exists():
        raise FileNotFoundError(f"Root directory does not exist: {root}")

    api_key = args.judge_api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is required for LLM judge scoring.")

    client = OpenAI(
        api_key=api_key,
        base_url=args.judge_base_url or None,
    )
    prompt_path = Path(__file__).resolve().parent / "benchmark" / "llm_judge.txt"
    prompt_template = prompt_path.read_text(encoding="utf-8")

    run_dirs = iter_run_dirs(root)
    if not run_dirs:
        print(f"[INFO] No run directories found under {root}")
        return

    print(f"[INFO] Found {len(run_dirs)} run directories under {root}")
    totals = {"runs": 0, "scored": 0, "skipped": 0, "rows": 0}
    for run_dir in run_dirs:
        stats = score_run_dir(
            run_dir=run_dir,
            client=client,
            prompt_template=prompt_template,
            judge_model=args.judge_model,
            max_retries=args.judge_max_retries,
            timeout=args.judge_timeout,
            force=args.force,
        )
        totals["runs"] += 1
        totals["scored"] += stats["scored"]
        totals["skipped"] += stats["skipped"]
        totals["rows"] += stats["total"]
        print(
            f"[DONE] {run_dir} scored={stats['scored']} skipped={stats['skipped']} total={stats['total']}"
        )

    print(
        f"[SUMMARY] runs={totals['runs']} scored={totals['scored']} "
        f"skipped={totals['skipped']} total_rows={totals['rows']}"
    )


if __name__ == "__main__":
    main()
