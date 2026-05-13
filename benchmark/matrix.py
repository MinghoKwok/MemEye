import datetime as dt
from pathlib import Path
from typing import Any, Dict, List

from .common import SCRIPT_DIR, load_yaml, resolve_config_path, write_csv, write_json, write_text
from .runner import run_modular_benchmark


def _summary_row(payload: Dict[str, Any]) -> Dict[str, Any]:
    summary = payload.get("summary", {})
    overall = summary.get("overall", {})
    return {
        "task_name":      payload.get("task_name", ""),
        "model_name":     payload.get("model_name", ""),
        "method_name":    payload.get("method_name", ""),
        "mode":           payload.get("mode", ""),
        "num_qas_run":    payload.get("num_qas_run", 0),
        "open_count":     summary.get("open_count", 0),
        "em":             overall.get("em", ""),
        "contains_gt":    overall.get("contains_gt", ""),
        "f1":             overall.get("f1", ""),
        "bleu":           overall.get("bleu", ""),
        "bleu_1":         overall.get("bleu_1", ""),
        "bleu_2":         overall.get("bleu_2", ""),
        "bert":           overall.get("bert", ""),
        "judge":          overall.get("judge", ""),
        "mcq_count":      summary.get("mcq_count", 0),
        "mcq_valid_rate": summary.get("mcq_valid_rate", ""),
        "run_dir":        payload.get("run_dir", ""),
    }


def _render_markdown(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return "# Benchmark Summary\n\nNo runs were executed.\n"

    headers = [
        "task_name", "model_name", "method_name", "mode",
        "num_qas_run", "em", "contains_gt", "f1", "bleu", "bleu_1", "bleu_2", "bert", "judge",
    ]
    lines = ["# Benchmark Summary", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join("---" for _ in headers) + " |")
    for row in rows:
        values = [str(row.get(header, "")) for header in headers]
        lines.append("| " + " | ".join(values) + " |")
    lines.append("")
    return "\n".join(lines)


def run_benchmark_matrix(
    task_config_path: str,
    model_config_paths: List[str],
    method_config_paths: List[str],
    output_root: str = "",
    mode: str = "",
    max_questions: int = 0,
    enable_bert_score: bool = False,
    enable_llm_judge: bool = False,
    judge_config: Dict[str, Any] = None,
) -> Dict[str, Any]:
    if not model_config_paths:
        raise ValueError("model_config_paths must not be empty")
    if not method_config_paths:
        raise ValueError("method_config_paths must not be empty")

    task_cfg = load_yaml(resolve_config_path(task_config_path))
    task_name = str(task_cfg.get("name", "task"))

    rows: List[Dict[str, Any]] = []
    payloads: List[Dict[str, Any]] = []
    for model_config_path in model_config_paths:
        for method_config_path in method_config_paths:
            payload = run_modular_benchmark(
                task_config_path=task_config_path,
                model_config_path=model_config_path,
                method_config_path=method_config_path,
                output_root=output_root,
                mode=mode,
                max_questions=max_questions,
                enable_bert_score=enable_bert_score,
                enable_llm_judge=enable_llm_judge,
                judge_config=judge_config,
            )
            payloads.append(payload)
            rows.append(_summary_row(payload))

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    matrix_root_raw = Path(output_root) if output_root else Path("runs")
    matrix_root = matrix_root_raw if matrix_root_raw.is_absolute() else (SCRIPT_DIR / matrix_root_raw)
    matrix_dir = (matrix_root / task_name / "matrices" / ts).resolve()

    summary_payload = {
        "task_name": task_name,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
        "rows": rows,
        "runs": payloads,
    }
    write_json(matrix_dir / "summary.json", summary_payload)
    write_csv(
        matrix_dir / "summary.csv",
        rows,
        fieldnames=[
            "task_name", "model_name", "method_name", "mode", "num_qas_run",
            "open_count", "em", "f1", "bleu", "bleu_1", "bleu_2", "bert", "judge",
            "mcq_count", "mcq_valid_rate", "run_dir",
        ],
    )
    write_text(matrix_dir / "summary.md", _render_markdown(rows))
    return {"matrix_dir": str(matrix_dir), "rows": rows, "runs": payloads}
