import argparse
import os

from raisex.search.algorithms.grpo import _parse_score_weights, rl_search


def main() -> None:
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_algo_config = os.path.join(os.path.dirname(__file__), "configforalgo.yaml")
    default_report = os.path.join(base_dir, "outputs", "ppo_report.json")

    parser = argparse.ArgumentParser(description="PPO search for RAG.")
    parser.add_argument("--qa_json", required=True, help="Path to QA JSON/JSONL.")
    parser.add_argument("--corpus_json", required=True, help="Path to corpus JSON.")
    parser.add_argument(
        "--config_yaml",
        default=default_algo_config,
        help="Path to algo config with search space.",
    )
    parser.add_argument(
        "--eval_mode",
        default="both",
        choices=["avg", "per_item", "both"],
        help="Evaluation mode.",
    )
    parser.add_argument(
        "--report_path",
        default=default_report,
        help="Path to write PPO report JSON.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=50,
        help="Number of PPO episodes.",
    )
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Samples per PPO update.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.05,
        help="Learning rate.",
    )
    parser.add_argument(
        "--kl_coeff",
        type=float,
        default=0.01,
        help="KL penalty coefficient.",
    )
    parser.add_argument(
        "--clip_ratio",
        type=float,
        default=0.2,
        help="PPO clip ratio.",
    )
    parser.add_argument(
        "--update_epochs",
        type=int,
        default=4,
        help="Policy update epochs per group.",
    )
    parser.add_argument(
        "--score_weights",
        default="",
        help="Weighted metrics, e.g. 'bertf11,llmaaj2'.",
    )
    args = parser.parse_args()

    score_weights = _parse_score_weights(args.score_weights)
    rl_search(
        qa_json_path=args.qa_json,
        corpus_json_path=args.corpus_json,
        config_path=args.config_yaml,
        eval_mode=args.eval_mode,
        report_path=args.report_path,
        episodes=args.episodes,
        seed=args.seed,
        learning_rate=args.lr,
        group_size=args.group_size,
        kl_coeff=args.kl_coeff,
        clip_ratio=args.clip_ratio,
        update_epochs=args.update_epochs,
        score_weights=score_weights,
        algorithm_label="ppo",
        algorithm_variant="ppo",
    )


if __name__ == "__main__":
    main()
