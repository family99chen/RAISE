import json
import os
import sys
from typing import Any, Dict

if os.getenv("PIPELINE_DEBUG") == "1" or os.getenv("EVAL_DEBUG") == "1":
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass

from raisex.core.evaluation_service import theoretical_getupperbound, theoretical_getupperbound_multimodal


def _write_report(path: str, payload: Dict[str, Any]) -> None:
    report_dir = os.path.dirname(path)
    if report_dir:
        os.makedirs(report_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    import argparse

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_report = os.path.join(base_dir, "outputs", "thupperbound_report.json")

    parser = argparse.ArgumentParser(description="Theoretical upperbound evaluation.")
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--eval_mode",
        default="both",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--report_path",
        default=default_report,
        help="Path to write report JSON.",
    )
    default_config = os.path.join(base_dir, "algorithms", "configforalgo.yaml")
    default_mm_config = os.path.join(
        base_dir, "algorithms", "configforalgomultimodal.yaml"
    )
    parser.add_argument(
        "--config_yaml",
        default=default_config,
        help="Path to algo config with search space.",
    )
    parser.add_argument(
        "--multimodal",
        action="store_true",
        help="Use multimodal theoretical upperbound.",
    )
    parser.add_argument(
        "--max_evals", type=int, default=None,
        help="Accepted for CLI compatibility; ignored by thupperbound.",
    )
    args = parser.parse_args()

    if args.multimodal and args.config_yaml == default_config:
        args.config_yaml = default_mm_config

    if args.multimodal:
        result = theoretical_getupperbound_multimodal(
            qa_json_path=args.qa_json,
            corpus_json_path=args.corpus_json,
            config_path=args.config_yaml,
            eval_mode=args.eval_mode,
        )
    else:
        result = theoretical_getupperbound(
            qa_json_path=args.qa_json,
            corpus_json_path=args.corpus_json,
            config_path=args.config_yaml,
            eval_mode=args.eval_mode,
        )
    payload = {
        "eval_report": result.get("eval_report"),
        "outputs": result.get("outputs"),
    }
    _write_report(args.report_path, payload)
    metrics = (result.get("eval_report") or {}).get("metrics") or {}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
