import argparse
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import run_modular_benchmark


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run modular multimodal memory benchmark experiments.")
    parser.add_argument("--task-config", type=str, default="config/tasks/brand_memory_test.yaml")
    parser.add_argument("--model-config", type=str, default="config/models/gpt_4_1_nano.yaml")
    parser.add_argument("--method-config", type=str, default="config/methods/full_context_multimodal.yaml")
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        default="open",
        choices=["open", "mcq", "both"],
        help="Evaluation mode: open-ended, MCQ, or both (default: open).",
    )

    # Optional metric flags
    parser.add_argument("--enable-bert-score", action="store_true",
                        help="Enable BERTScore (roberta-large). Slow — downloads ~440 MB on first use.")
    parser.add_argument("--enable-llm-judge", action="store_true",
                        help="Enable LLM-as-a-judge scoring.")
    parser.add_argument("--judge-model", type=str, default="gpt-4.1-nano",
                        help="Judge model name (default: gpt-4.1-nano).")
    parser.add_argument("--judge-api-key", type=str, default="",
                        help="API key for judge model. Falls back to OPENAI_API_KEY env var if omitted.")
    parser.add_argument("--judge-base-url", type=str, default="",
                        help="Base URL for judge API. Leave empty for default OpenAI endpoint.")
    parser.add_argument("--judge-max-retries", type=int, default=3)
    parser.add_argument("--judge-timeout", type=int, default=60)

    return parser.parse_args()


def build_judge_config(args: argparse.Namespace) -> Optional[Dict[str, Any]]:
    if not args.enable_llm_judge:
        return None
    return {
        "model": args.judge_model,
        "api_key": args.judge_api_key or os.environ.get("OPENAI_API_KEY", ""),
        "base_url": args.judge_base_url or None,
        "max_retries": args.judge_max_retries,
        "timeout": args.judge_timeout,
    }


def main() -> None:
    args = parse_args()
    run_modular_benchmark(
        task_config_path=args.task_config,
        model_config_path=args.model_config,
        method_config_path=args.method_config,
        output_root=args.output_root,
        mode=args.mode,
        max_questions=args.max_questions,
        enable_bert_score=args.enable_bert_score,
        enable_llm_judge=args.enable_llm_judge,
        judge_config=build_judge_config(args),
    )


if __name__ == "__main__":
    main()
