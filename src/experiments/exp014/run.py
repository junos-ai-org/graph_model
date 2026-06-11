"""Exp 014 — GTLM paradigm on tables: prep / train / eval stages.

Variant `gtlm-graph`: Llama-3.2-1B + LoRA(r=16, q/k/v/o) + GTLM's generic
graph biases (SPD + RRWP + Magnetic) over the user-specified table graph
(cells + column nodes; cell->column and cell->left edges), bidirectional
prefix over the graph, per-node RoPE reset, causal question/answer suffix.

Trained on the exp013 TableInstruct-DP pool (identical split: seed 42,
max_cells=200, 17% val carve); 3 epochs; final-epoch checkpoint evaluated on
the TableBench DP test file (non-viz), records keyed by TableBench `id` for
paired McNemar against exp013's plain-lora DP records.

Usage (on a GPU pod, from the graph_model repo root):
    python -m src.experiments.exp014.run prep  --out /workspace/exp014
    python -m src.experiments.exp014.run train --out /workspace/exp014
    python -m src.experiments.exp014.run eval  --out /workspace/exp014 \
        --checkpoint /workspace/exp014/runs/gtlm-graph/final
Smoke: add --max-examples 40 / --eval-max 20.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from ...models.llama_k_hop import KHopLlamaConfig, KHopGraphLlamaForCausalLM
from ...models.llama_utils import load_bias_parameters
from ...utils.text_graph_collator import GraphCollator
from ...utils.text_graph_trainer import GraphTrainer
from .features import compute_features_chunk
from .table_graph import (add_prompt_node, build_table_graph,
                          prompt_ids_and_labels, tokenize_node_texts,
                          total_tokens, QUESTION_TEMPLATE)
from .tablebench_data import TableBenchDP

BASE_MODEL = "meta-llama/Llama-3.2-1B"
TRAIN_MAX_TOKENS = 1024     # mirror exp013's max_seq_len drop guard
EVAL_MAX_TOKENS = 4096      # mirror exp013's eval guard
EVAL_MAX_NODES = 1500       # safety cap for O(N^3) feature math
MAX_NEW_TOKENS = 64         # matches exp013's DP eval
SEED = 42
BIAS_KW = dict(spd=True, rrwp=True, magnetic=True,
               max_spd=32, max_rw_steps=8, magnetic_dim=32)
FEATURE_CHUNK = 16


from .scoring import denotation_match


# ── example -> graph item ────────────────────────────────────────────────────

def build_example(tokenizer, ex: dict, with_answer: bool):
    """(graph, node_input_ids, labels|None). Caller applies the token cap."""
    g = build_table_graph(ex["header"], ex["rows"])
    g = add_prompt_node(g, ex["question"])
    node_ids = tokenize_node_texts(tokenizer, g)
    p = g.graph["prompt_node"]
    if with_answer:
        ids, labels = prompt_ids_and_labels(tokenizer, ex["question"],
                                            ex["answer"], tokenizer.eos_token_id)
        node_ids[p] = ids
        return g, node_ids, torch.tensor(labels, dtype=torch.long)
    return g, node_ids, None


class GraphItemDataset(Dataset):
    """List of pre-built item dicts in GraphCollator's expected schema."""

    def __init__(self, items: list[dict]):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def make_item(g, node_input_ids, labels, feats) -> dict:
    item = {
        "num_nodes": g.number_of_nodes(),
        "prompt_node": g.graph["prompt_node"],
        "edges": list(g.edges()),
        "input_ids": node_input_ids,
        **feats,
    }
    if labels is not None:
        item["labels"] = labels
    return item


# ── prep ─────────────────────────────────────────────────────────────────────

def stage_prep(args, tokenizer):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    suffix = f"_n{args.max_examples}" if args.max_examples > 0 else ""
    cache = out / f"train_items{suffix}.pt"
    if cache.exists() and not args.force:
        print(f"prep: {cache} exists, skipping (use --force to rebuild)")
        return

    ds = TableBenchDP(seed=SEED, max_cells=200)
    train = ds.split("train")
    print(f"train split: {len(train)} examples (exp013 expects 4506 at full scale)")
    if args.max_examples > 0:
        train = train[: args.max_examples]

    built, n_dropped = [], 0
    for ex in train:
        g, node_ids, labels = build_example(tokenizer, ex, with_answer=True)
        if total_tokens(node_ids) > TRAIN_MAX_TOKENS:
            n_dropped += 1
            continue
        built.append((g, node_ids, labels))
    print(f"kept {len(built)} train examples "
          f"(dropped {n_dropped} over {TRAIN_MAX_TOKENS} tokens)")
    if not built:
        raise RuntimeError("no train examples survived the token cap")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Chunk by similar node count to limit padded-batch waste.
    order = sorted(range(len(built)), key=lambda i: built[i][0].number_of_nodes())
    items: list[dict | None] = [None] * len(built)
    for s in range(0, len(order), FEATURE_CHUNK):
        idxs = order[s: s + FEATURE_CHUNK]
        feats = compute_features_chunk([built[i][0] for i in idxs], device)
        for i, f in zip(idxs, feats):
            g, node_ids, labels = built[i]
            items[i] = make_item(g, node_ids, labels, f)
        if (s // FEATURE_CHUNK) % 20 == 0:
            print(f"features {s + len(idxs)}/{len(built)}", flush=True)
    torch.save(items, cache)
    print(f"prep done -> {cache} ({len(items)} items)")


# ── model construction ───────────────────────────────────────────────────────

class Exp014Model(KHopGraphLlamaForCausalLM):
    """Fixes generation position_ids: upstream prepare_inputs_for_generation
    slices a static position_ids tensor, so every generated token reuses the
    last prompt position. Extend positions monotonically instead (continuing
    the prompt node's RoPE positions).

    NOTE the guard is conditional on input_ids outgrowing position_ids: on
    transformers==4.50.3 (the pod pin) HF never extends position_ids, so this
    branch fires every step; on >=4.52 HF extends them itself and this branch
    is correctly inert. Do not remove the guard."""

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      attention_mask=None, inputs_embeds=None,
                                      **kwargs):
        pos = kwargs.get("position_ids")
        if pos is not None and input_ids.shape[1] > pos.shape[1]:
            k = input_ids.shape[1] - pos.shape[1]
            step = torch.arange(1, k + 1, device=pos.device).unsqueeze(0)
            kwargs["position_ids"] = torch.cat(
                [pos, pos[:, -1:].expand(pos.shape[0], k) + step], dim=1)
        return super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, inputs_embeds=inputs_embeds, **kwargs)


def load_model(checkpoint: str | None):
    config = KHopLlamaConfig.from_pretrained(BASE_MODEL, k_hop=0, **BIAS_KW)
    model = Exp014Model.from_pretrained(
        BASE_MODEL, config=config, attn_implementation="eager",
        torch_dtype=torch.bfloat16)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    from peft import LoraConfig, PeftModel, get_peft_model
    if checkpoint is None:
        for p in model.parameters():
            p.requires_grad = False
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                          target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                          task_type="CAUSAL_LM")
        model = get_peft_model(model, lora)
        for n, p in model.named_parameters():
            if "graph_bias" in n:
                p.requires_grad = True
    else:
        model = PeftModel.from_pretrained(model, checkpoint)
        lb = [p for n, p in model.named_parameters() if "lora_B" in n]
        assert lb and any(torch.linalg.norm(p.float()).item() > 0 for p in lb), \
            f"all lora_B weights zero — adapter at {checkpoint} loaded as no-op"
        load_bias_parameters(model, checkpoint)
        mags = {n: p.detach().abs().max().item()
                for n, p in model.named_parameters() if "graph_bias" in n}
        mx = max(mags.values()) if mags else 0.0
        assert mags, "no graph_bias parameters found on the model"
        print(f"loaded {len(mags)} graph_bias tensors, max|param|={mx:.4f}")
        if mx == 0.0:
            print("WARNING: graph biases are all-zero — structural effect INERT")
    return model


def require_wandb():
    if not os.environ.get("WANDB_API_KEY") and \
            os.environ.get("EXP014_ALLOW_NO_WANDB") != "1":
        raise RuntimeError("WANDB_API_KEY unset (set EXP014_ALLOW_NO_WANDB=1 "
                           "only for local smoke)")


# ── train ────────────────────────────────────────────────────────────────────

def stage_train(args, tokenizer):
    require_wandb()
    os.environ.setdefault("WANDB_PROJECT", "graph-reasoning-llm")
    out = Path(args.out)
    suffix = f"_n{args.max_examples}" if args.max_examples > 0 else ""
    items = torch.load(out / f"train_items{suffix}.pt", weights_only=False)
    train_ds = GraphItemDataset(items)
    print(f"training on {len(train_ds)} examples")

    model = load_model(None)
    trainable = [(n, p.numel()) for n, p in model.named_parameters() if p.requires_grad]
    n_lora = sum(k for n, k in trainable if "lora" in n)
    n_bias = sum(k for n, k in trainable if "graph_bias" in n)
    print(f"trainable: lora={n_lora:,} graph_bias={n_bias:,}")
    assert n_lora > 0 and n_bias > 0

    from transformers import TrainingArguments
    run_dir = out / "runs" / "gtlm-graph"
    targs = TrainingArguments(
        output_dir=str(run_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=args.epochs,
        eval_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb" if os.environ.get("WANDB_API_KEY") else "none",
        run_name=os.environ.get("WANDB_RUN_GROUP", "exp_014") + "/gtlm-graph",
        seed=SEED, data_seed=SEED,
    )
    trainer = GraphTrainer(
        model=model, args=targs, train_dataset=train_ds,
        data_collator=GraphCollator(tokenizer=tokenizer, k_hop=0),
        active_params=["graph_bias"], bias_lr=1e-2,
    )
    trainer.train()
    final = run_dir / "final"
    trainer.save_model(str(final))
    print(f"final checkpoint -> {final}")


# ── eval ─────────────────────────────────────────────────────────────────────

def stage_eval(args, tokenizer):
    model = load_model(args.checkpoint)
    model.eval()
    device = next(model.parameters()).device
    collator = GraphCollator(tokenizer=tokenizer, k_hop=0)

    ds = TableBenchDP(seed=SEED, max_cells=200)
    out_dir = Path(args.out) / "eval" / "gtlm-graph"
    out_dir.mkdir(parents=True, exist_ok=True)
    rec_path = out_dir / "records.jsonl"

    # Resume support: a pod kill must not lose finished generations. Records
    # are appended and flushed per example; already-scored ids are skipped.
    done: dict[str, bool] = {}
    if rec_path.exists():
        for line in rec_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = bool(r["correct"])
        print(f"resuming: {len(done)} examples already scored in {rec_path}")

    n, correct, skipped, failed = len(done), sum(done.values()), [], []
    with rec_path.open("a") as rec_f:
        for i, ex in enumerate(ds.test(apply_max_cells=False)):
            if args.eval_max > 0 and n >= args.eval_max:
                break
            if ex["id"] in done:
                continue
            try:
                g, node_ids, _ = build_example(tokenizer, ex, with_answer=False)
                if total_tokens(node_ids) > EVAL_MAX_TOKENS or \
                        g.number_of_nodes() > EVAL_MAX_NODES:
                    skipped.append(ex["id"])
                    continue
                feats = compute_features_chunk([g], device)[0]
                batch = collator([make_item(g, node_ids, None, feats)])
                batch.pop("labels", None)
                prep_len = sum(len(x) for x in node_ids)
                # Training ran under Trainer's bf16 autocast; mirror it here so
                # fp32 feature tensors meet the bf16 bias-MLP weights cleanly.
                with torch.no_grad(), torch.autocast(
                        device_type=device.type, dtype=torch.bfloat16,
                        enabled=device.type == "cuda"):
                    gen = model.generate(input_graph_batch=batch,
                                         max_new_tokens=MAX_NEW_TOKENS,
                                         do_sample=False,
                                         pad_token_id=tokenizer.eos_token_id)
                pred = tokenizer.decode(gen[0][prep_len:],
                                        skip_special_tokens=True).strip()
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                failed.append({"id": ex["id"], "error": "cuda OOM"})
                print(f"[gtlm-graph] OOM on id={ex['id']} — skipped", flush=True)
                continue
            ok = denotation_match(pred, [ex["answer"]])
            correct += int(ok)
            n += 1
            rec_f.write(json.dumps({
                "id": ex["id"], "qtype": ex["qtype"], "qsubtype": ex["qsubtype"],
                "question": ex["question"], "pred": pred,
                "gold": ex["answer"], "correct": ok}) + "\n")
            rec_f.flush()
            if n % 25 == 0:
                print(f"[gtlm-graph] {n} scored ({i + 1} seen) "
                      f"acc so far {correct / n:.4f}", flush=True)

    per: dict[str, list[int]] = {}
    for line in rec_path.read_text().splitlines():
        r = json.loads(line)
        for key in (r["qtype"], f'{r["qtype"]}/{r["qsubtype"]}'):
            per.setdefault(key, [0, 0])
            per[key][0] += int(r["correct"])
            per[key][1] += 1
    acc = correct / n if n else 0.0
    results = {
        "variant": "gtlm-graph", "checkpoint": args.checkpoint,
        "n": n, "accuracy": acc, "n_skipped_over_len": len(skipped),
        "skipped_ids": skipped, "failed": failed,
        "max_new_tokens": MAX_NEW_TOKENS,
        "eval_max_tokens": EVAL_MAX_TOKENS, "bias_kw": BIAS_KW,
        "per_category": {k: {"correct": c, "n": t, "accuracy": c / t}
                         for k, (c, t) in sorted(per.items())},
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[gtlm-graph] n={n} accuracy={acc:.4f} skipped={len(skipped)} -> {out_dir}")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["prep", "train", "eval"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-examples", type=int, default=-1)
    ap.add_argument("--eval-max", type=int, default=-1)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    if args.stage == "prep":
        stage_prep(args, tokenizer)
    elif args.stage == "train":
        stage_train(args, tokenizer)
    else:
        if not args.checkpoint:
            raise SystemExit("--checkpoint required for eval")
        stage_eval(args, tokenizer)


if __name__ == "__main__":
    main()
