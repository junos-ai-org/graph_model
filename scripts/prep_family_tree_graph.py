#!/usr/bin/env python3
"""Prepare the Family Tree graph dataset for kgqa-001 reproduction.

The fork's `src/experiments/our_tests/family_tree_prep.py` ships with the
graph-dataset save block commented out (only the LLaGA path is active). This
script calls the same underlying functions and writes the .gtds files to the
canonical path the trainer expects.

Use --n_train / --n_val / --n_test to control dataset size. Defaults match
the values in family_tree_prep.py's __main__ (paper-ish config).
"""
import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.experiments.our_tests.family_tree_gen import generate_dataset
from src.experiments.our_tests.family_tree_prep import (
    prepare_graph_dataset,
    save_graph_dataset,
    GetGraphLabels,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3500)
    ap.add_argument("--n_val", type=int, default=200)
    ap.add_argument("--n_test", type=int, default=1000)
    ap.add_argument("--output", default="./src/experiments/our_tests/family_tree_graph_dataset")
    ap.add_argument("--model", default="meta-llama/Llama-3.2-1B")
    ap.add_argument("--max_rrwp_steps", type=int, default=16)
    ap.add_argument("--magnetic_q", type=float, default=0.25)
    args = ap.parse_args()

    print(f"--- Family Tree prep: n_train={args.n_train} n_val={args.n_val} n_test={args.n_test} ---")
    raw_datasets = generate_dataset(
        n_train=args.n_train,
        n_val=args.n_val,
        n_test=args.n_test,
        return_dict=True,
    )
    print(f"Raw datasets: {[(k, len(v)) for k, v in raw_datasets.items()]}")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    get_graph_labels = GetGraphLabels(question_end=[32, 25], tokenizer=tokenizer)

    print("Building incidence graphs (prepare_graph_dataset)...")
    graph_datasets = prepare_graph_dataset(raw_datasets)

    params = {
        "tokenizer": tokenizer,
        "max_length": 32_768,
        "max_rrwp_steps": args.max_rrwp_steps,
        "magnetic_q": args.magnetic_q,
        "get_graph_labels": get_graph_labels,
    }

    print(f"Computing features + saving to {args.output} ...")
    save_graph_dataset(graph_datasets, args.output, params)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
