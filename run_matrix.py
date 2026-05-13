import argparse
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark import run_benchmark_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a model x method multimodal memory benchmark matrix.")
    parser.add_argument("--task-config", type=str, default="config/tasks/brand_memory_test.yaml")
    parser.add_argument("--model-config", dest="model_configs", action="append", required=True)
    parser.add_argument("--method-config", dest="method_configs", action="append", required=True)
    parser.add_argument("--output-root", type=str, default="runs")
    parser.add_argument("--max-questions", type=int, default=0)
    parser.add_argument(
        "--mode",
        type=str,
        default="",
        choices=["", "open", "mcq", "both"],
        help="Override eval mode for all runs. Leave empty to use task config default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_benchmark_matrix(
        task_config_path=args.task_config,
        model_config_paths=args.model_configs,
        method_config_paths=args.method_configs,
        output_root=args.output_root,
        mode=args.mode,
        max_questions=args.max_questions,
    )
    print(f"[INFO] Saved matrix summary: {result['matrix_dir']}")


if __name__ == "__main__":
    main()
