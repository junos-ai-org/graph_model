"""
Exp 003 — structural attention bias for tabular reasoning on WikiTableQuestions.

Two conditions, identical otherwise:
  Cond A:  --baseline  → BIAS_PARAMS all False (LoRA-only baseline)
  Cond C:  (default)   → BIAS_PARAMS all True  (full structural bias)

See docs/reproduction_plan_003.md in the research repo for the full runbook.
"""

import argparse
import os

from ...utils import GraphCollator
from ...train import (
    init_model, select_active_params, print_trainable_parameters,
    training_run, save_run_metadata, get_device,
)
from .load_data import load_data

EXPERIMENT_DIR  = os.path.dirname(os.path.abspath(__file__))
EXPERIMENT_NAME = "exp_003_table_bias"

# Default = Cond C (all four exp 003 biases on). The --baseline flag flips
# every key here to False to produce Cond A.
BIAS_PARAMS = {
    # Existing GTLM biases — all OFF for exp 003. We isolate the structural
    # contribution of the four new table biases by training without SPD /
    # RRWP / Magnetic.
    "spd":            False,
    "max_spd":        8,
    "laplacian":      False,
    "rwse":           False,
    "rrwp":           False,
    "max_rw_steps":   16,
    "magnetic":       False,
    "magnetic_dim":   32,
    "magnetic_q":     0.25,
    # New exp 003 table-structural biases.
    "col_header":         True,   # α: cell ↔ its column header
    "header_clique":      True,   # ε: column-header ↔ column-header
    "row_anchor_agg":     True,   # γ: row anchor ↔ every cell in its row
    "row_anchor_spine":   True,   # δ: row anchor ↔ row anchor
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Exp 003: structural attention bias on WikiTableQuestions."
    )
    parser.add_argument("--model_name",       type=str,   default="meta-llama/Llama-3.2-1B")
    parser.add_argument("--num_epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",       type=int,   default=8)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate",    type=float, default=5e-5)
    parser.add_argument("--bias_learning_rate", type=float, default=1e-2)
    parser.add_argument("--eval_every",       type=int,   default=200)
    parser.add_argument("--lora_r",           type=int,   default=32,
                        help="LoRA rank. Set to 0 to disable LoRA.")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--include_f1",       action="store_true")
    parser.add_argument("--wandb_project",    type=str,   default="GraphLLM",
                        help="WandB project name. Pass 'none' to disable logging.")
    parser.add_argument("--run_name",         type=str,   default=None)
    parser.add_argument("--active_params",    nargs="+",
                        default=["graph_bias"],
                        help="Trainable substrings besides LoRA. 'graph_bias' picks up all table biases.")
    # Exp-003-specific knobs
    parser.add_argument("--max_cells",        type=int,   default=200,
                        help="Skip tables with more than this many cells (excluding the question node).")
    parser.add_argument("--max_train",        type=int,   default=-1,
                        help="Cap on train-split size (for smoke runs). -1 = use all.")
    parser.add_argument("--max_val",          type=int,   default=-1,
                        help="Cap on val-split size. -1 = use all.")
    parser.add_argument("--max_test",         type=int,   default=-1,
                        help="Cap on test-split size. -1 = use all.")
    parser.add_argument("--baseline",         action="store_true",
                        help="Cond A: zero out all four exp-003 BIAS_PARAMS to train the LoRA-only baseline.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Cond A vs Cond C selector
    bias_params = dict(BIAS_PARAMS)
    if args.baseline:
        for key in ("col_header", "header_clique", "row_anchor_agg", "row_anchor_spine"):
            bias_params[key] = False

    wandb_project = None if args.wandb_project.lower() == "none" else args.wandb_project

    lora_config = {
        "r":              args.lora_r,
        "lora_alpha":     args.lora_r * 2,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        "lora_dropout":   0.05,
        "bias":           "none",
    } if args.lora_r > 0 else None

    cond_tag = "A-baseline" if args.baseline else "C-fullbias"
    run_name = args.run_name or f"exp_003_{cond_tag}_r{args.lora_r}_lr{args.learning_rate}"
    run_name = save_run_metadata(
        run_name=run_name,
        experiment_dir=EXPERIMENT_DIR,
        base_model=args.model_name,
        active_params=args.active_params,
        lr=args.learning_rate,
        bias_lr=args.bias_learning_rate,
        lora_config=lora_config,
        num_epochs=args.num_epochs,
        bias_params=bias_params,
        k_hop=0,
        condition=cond_tag,
        max_cells=args.max_cells,
    )

    device = get_device()
    model, tokenizer = init_model(
        model_name=args.model_name,
        device=device,
        bias_params=bias_params,
        k_hop=0,
    )

    print(f"Loading WTQ data (max_cells={args.max_cells}, max_train={args.max_train})...")
    train_dataset, eval_dataset, test_dataset = load_data(
        tokenizer,
        max_cells=args.max_cells,
        max_train=args.max_train,
        max_val=args.max_val,
        max_test=args.max_test,
        seed=args.seed,
    )
    collator = GraphCollator(k_hop=0)

    print(f"Train: {len(train_dataset)}  Eval: {len(eval_dataset)}  Test: {len(test_dataset)}")
    print(f"Condition: {cond_tag}")

    model = select_active_params(model, active_params=args.active_params, lora=lora_config)
    print_trainable_parameters(model)

    training_run(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        test_dataset=test_dataset,
        collator=collator,
        run_name=run_name,
        experiment_name=EXPERIMENT_NAME,
        experiment_dir=EXPERIMENT_DIR,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        bias_learning_rate=args.bias_learning_rate,
        accumulation_steps=args.accumulation_steps,
        active_params=args.active_params,
        eval_every=args.eval_every,
        gradient_checkpointing=not args.no_gradient_checkpointing,
        include_f1=args.include_f1,
        wandb_project=wandb_project,
        seed=args.seed,
    )
