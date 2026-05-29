"""
Generation-based denotation accuracy eval for exp 003 checkpoints (the exp 004
harness; lives here because it shares the WTQ loader and per-cell metadata
machinery with the training code).

What it does:
  1. Reconstructs a trained checkpoint (LoRA + bias params) via the fork's
     existing `_load_model_from_checkpoint`.
  2. Builds the WTQ test split with prompt-node text = `"Question: q Answer: "`
     (no gold answer; we want the model to generate it).
  3. Runs greedy `model.generate(input_graph_batch=batch, max_new_tokens=128)`
     batched, threading the fork's existing graph-aware generate() override.
  4. Decodes the generated suffix, parses it as a comma/semicolon-separated
     list, normalizes, and computes set-equality denotation accuracy against
     the gold answer set.
  5. Writes a JSON with per-example records + aggregate denotation_accuracy.

Usage:
    python -m src.experiments.exp_003_table_bias.eval_denotation \\
        --checkpoint_path /workspace/code/checkpoints/exp_003_table_bias/<run>/checkpoint-<step> \\
        --out             /workspace/logs/condX_denotation.json \\
        [--max_test 20]   # smoke; default -1 = all
        [--batch_size 4]
        [--max_new_tokens 128]

Caveat (also noted in docs/reproduction_plan_004.md):
  The official WTQ eval (Pasupat & Liang 2015) does richer normalization
  (number parsing, date parsing, article stripping). We do simple
  lowercase + strip-trailing-punctuation only. Absolute numbers will be
  a few points below published WTQ figures; deltas between two conditions
  evaluated under identical post-processing are preserved.
"""

import argparse
import json
import re
import time
from pathlib import Path
from typing import List

import torch
from transformers import AutoTokenizer

from ...train.eval import _load_model_from_checkpoint
from ...utils import GraphCollator, TextGraphDataset
from .load_data import (
    WTQ_HF_PATH,
    _build_graph_for_example,
    _normalize_answer_list,
)


# ── Parsing / normalization ───────────────────────────────────────────────────

_PUNCT_RE = re.compile(r'[.,;!?]+$')
_WS_RE    = re.compile(r'\s+')


def _normalize_cell(s: str) -> str:
    """Normalize a single answer cell for set comparison."""
    s = str(s).strip().lower()
    s = _PUNCT_RE.sub('', s)
    s = _WS_RE.sub(' ', s).strip()
    return s


_ANSWER_MARKER_RE = re.compile(r'(?i)answer\s*:\s*')


def _parse_predicted(text: str) -> List[str]:
    """Parse the model's generated answer string into a list of normalized cells.

    Under greedy generation with `max_new_tokens=128`, the trained model
    frequently emits a long suffix that re-echoes table content and the
    question before producing its actual answer (a learned pattern from
    the training format "Question: q Answer: gold"). So we:
      1. Find the LAST 'Answer:' marker in the generated text.
      2. Take what follows it, up to the next newline.
      3. Split on comma / semicolon, normalize each cell.

    Fallback: if no 'Answer:' marker is found, treat the first line of
    the suffix as the answer (matches the simpler eval the parser was
    originally written for; rare in practice).
    """
    matches = list(_ANSWER_MARKER_RE.finditer(text))
    if matches:
        last = matches[-1]
        tail = text[last.end():]
    else:
        tail = text
    # Stop at the first newline OR an echoed "Question:" — the model often
    # tries to continue with another question after its answer.
    tail = tail.split('\n')[0]
    tail = re.split(r'(?i)\bquestion\s*:', tail)[0]
    parts = re.split(r'[,;]', tail)
    return [_normalize_cell(p) for p in parts if _normalize_cell(p)]


def _gold_cells(answers) -> List[str]:
    if isinstance(answers, list):
        return [_normalize_cell(a) for a in answers]
    return [_normalize_cell(answers)]


def _denotation_match(pred: List[str], gold: List[str]) -> bool:
    return set(pred) == set(gold)


# ── Dataset construction (no gold in prompt; gold returned separately) ────────

def _load_eval_dataset(tokenizer, max_cells: int, max_test: int, seed: int):
    """Build the WTQ test split with prompt-node text containing NO gold answer.

    Returns:
        (TextGraphDataset, gold_answers: list of list[str], questions: list[str])
        with all three in matching order.
    """
    from datasets import load_dataset
    import random

    rng = random.Random(seed)
    ds_raw = load_dataset(WTQ_HF_PATH)
    pool = list(ds_raw['test'])
    nu_pool = [ex for ex in pool if ex['id'].startswith('nu-')]

    graphs = []
    gold_answers = []
    questions = []
    skipped = 0
    for ex in nu_pool:
        question = ex['question']
        answers  = ex['answers']
        table    = ex['table']
        header   = table['header']
        rows     = table['rows']

        n_cells  = len(header) + sum(len(r) for r in rows)
        if n_cells > max_cells:
            skipped += 1
            continue

        # CRUCIAL: pass empty answer_str so prompt_node text = "Question: q Answer: "
        g = _build_graph_for_example(question, "", header, rows)
        graphs.append(g)
        gold_answers.append(answers if isinstance(answers, list) else [answers])
        questions.append(question)

        if max_test > 0 and len(graphs) >= max_test:
            break

    ds = TextGraphDataset(graphs, dataset_label="wtq-eval-test")
    ds.compute_table_metadata()
    ds.tokenize(tokenizer, max_length=64, add_eos=False)
    # No labels — we're not training.

    print(f"  eval n={len(ds)}  skipped (max_cells={max_cells})={skipped}")
    return ds, gold_answers, questions


# ── Suffix extraction ─────────────────────────────────────────────────────────

def _extract_generated_suffix(
    output_ids: torch.Tensor,           # (B, prefix_len + gen_len)
    prefix_len: int,                    # number of input tokens (left-padded same length per row)
    tokenizer,
) -> List[str]:
    """Decode each row's generated suffix to a string.

    With left-padding, all rows have their original (padded) tokens at
    positions [0, prefix_len) and generated tokens at [prefix_len, end).
    """
    decoded = []
    suffix_ids = output_ids[:, prefix_len:]
    for row in suffix_ids:
        # Trim trailing PAD/EOS
        ids = row.tolist()
        if tokenizer.eos_token_id is not None and tokenizer.eos_token_id in ids:
            ids = ids[:ids.index(tokenizer.eos_token_id)]
        if tokenizer.pad_token_id is not None:
            ids = [t for t in ids if t != tokenizer.pad_token_id]
        decoded.append(tokenizer.decode(ids, skip_special_tokens=True))
    return decoded


# ── Main eval loop ────────────────────────────────────────────────────────────

def evaluate(
    checkpoint_path: str,
    out_path: str,
    max_test: int,
    batch_size: int,
    max_new_tokens: int,
    seed: int,
    max_cells: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading checkpoint: {checkpoint_path}")
    model, config = _load_model_from_checkpoint(checkpoint_path, device)
    model.eval()

    # Tokenizer: reconstruct from the saved base model name (same pattern as test.py)
    graph_bias_config_path = Path(checkpoint_path) / "graph_bias_config.json"
    with open(graph_bias_config_path) as f:
        base_model_name = json.load(f)["base_model_name_or_path"]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("building eval dataset (WTQ test, no gold in prompt) ...")
    ds, gold_answers, questions = _load_eval_dataset(
        tokenizer, max_cells=max_cells, max_test=max_test, seed=seed,
    )
    collator = GraphCollator(k_hop=0)

    correct = 0
    total = 0
    records = []
    t_start = time.time()

    for batch_start in range(0, len(ds), batch_size):
        batch_indices = list(range(batch_start, min(batch_start + batch_size, len(ds))))
        items = [ds[i] for i in batch_indices]
        batch = collator(items)

        # Send tensors to device (the generate() override also does this, but
        # doing it once here avoids per-step copies during the decode loop).
        for k, v in list(batch.items()):
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
            elif isinstance(v, tuple) and all(isinstance(t, torch.Tensor) for t in v):
                batch[k] = tuple(t.to(device) for t in v)

        # Greedy generation
        with torch.no_grad():
            output = model.generate(
                input_graph_batch=batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=1,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # output: (B, prefix_len + gen_len). The fork's generate() override
        # uses padding_side="left" in _prepare_inputs, so the prefix occupies
        # the same column range across rows. Decode after the prefix.
        # We don't know `prefix_len` directly here, but we can read it back
        # from the prepared input that generate() built; it equals
        # output.shape[1] - max_new_tokens UNLESS the generation stopped
        # early via EOS for all rows (HF then truncates) — handle that.
        prefix_len = output.shape[1] - max_new_tokens
        if prefix_len < 0:
            # All rows stopped before max_new_tokens — HF still emits the
            # full prefix, so total = prefix + actual_gen. We need to compute
            # prefix_len from the model's prepare step. As a fallback, the
            # prefix tokens end at the position where generated content begins.
            # Recompute by calling _prepare_inputs explicitly.
            prepared = model._prepare_inputs(
                batch['input_ids'], batch['prompt_node'], padding_side="left"
            )
            prefix_len = prepared['input_ids'].shape[1]

        suffix_strs = _extract_generated_suffix(output, prefix_len, tokenizer)

        for local_i, idx in enumerate(batch_indices):
            pred_str = suffix_strs[local_i].strip()
            pred_cells = _parse_predicted(pred_str)
            gold_cells = _gold_cells(gold_answers[idx])
            ok = _denotation_match(pred_cells, gold_cells)

            records.append({
                "idx":      idx,
                "question": questions[idx],
                "gold":     gold_answers[idx],
                "predicted_text": pred_str,
                "predicted_cells": pred_cells,
                "gold_cells":      gold_cells,
                "correct":         bool(ok),
            })
            correct += int(ok)
            total   += 1

        if (batch_start // batch_size) % 25 == 0:
            elapsed = time.time() - t_start
            rate = total / max(elapsed, 1e-6)
            eta = (len(ds) - total) / max(rate, 1e-6)
            print(f"  [{total:>5}/{len(ds)}]  acc={correct/total:.4f}  "
                  f"rate={rate:.2f} ex/s  eta={eta/60:.1f} min")

    elapsed = time.time() - t_start
    denotation_accuracy = correct / total if total else 0.0
    summary = {
        "checkpoint_path": checkpoint_path,
        "n_examples":      total,
        "n_correct":       correct,
        "denotation_accuracy": denotation_accuracy,
        "elapsed_seconds": elapsed,
        "config": {
            "max_new_tokens": max_new_tokens,
            "do_sample":      False,
            "num_beams":      1,
            "max_cells":      max_cells,
            "batch_size":     batch_size,
            "seed":           seed,
        },
        "records": records,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2))
    print(f"\ndone:  denotation_accuracy = {denotation_accuracy:.4f}  "
          f"({correct}/{total})  elapsed = {elapsed/60:.1f} min")
    print(f"wrote {out_path}")
    return summary


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint_path", type=str, required=True)
    p.add_argument("--out",             type=str, required=True,
                   help="Output JSON path (e.g. /workspace/logs/condA_denotation.json).")
    p.add_argument("--max_test",        type=int, default=-1,
                   help="Cap on number of test examples; -1 = all.")
    p.add_argument("--batch_size",      type=int, default=4)
    p.add_argument("--max_new_tokens",  type=int, default=128)
    p.add_argument("--seed",            type=int, default=42)
    p.add_argument("--max_cells",       type=int, default=200)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(
        checkpoint_path=args.checkpoint_path,
        out_path=args.out,
        max_test=args.max_test,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
        max_cells=args.max_cells,
    )
