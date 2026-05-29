"""
Evaluate a checkpoint on the exp 003 WTQ test split.

Usage:
    python -m src.experiments.exp_003_table_bias.test \\
        --checkpoint_path ./checkpoints/exp_003_table_bias/<run_name> \\
        [--include_f1]
"""

import argparse
import json
import os

from transformers import AutoTokenizer

from ...train.eval import evaluate_checkpoint
from .load_data import load_data

EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a checkpoint on the exp 003 WTQ test split."
    )
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--include_f1", action="store_true")
    parser.add_argument("--max_cells", type=int, default=200)
    parser.add_argument("--max_test",  type=int, default=-1)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    graph_bias_config_path = os.path.join(args.checkpoint_path, "graph_bias_config.json")
    with open(graph_bias_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    print("Loading WTQ test split...")
    _, _, test_dataset = load_data(
        tokenizer,
        max_cells=args.max_cells,
        max_test=args.max_test,
    )

    evaluate_checkpoint(
        checkpoint_path=args.checkpoint_path,
        test_dataset=test_dataset,
        experiment_dir=EXPERIMENT_DIR,
        include_f1=args.include_f1,
    )
