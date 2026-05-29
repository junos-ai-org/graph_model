"""
WikiTableQuestions (WTQ) loader for exp 003 — cell-as-node graph build.

Each table cell becomes a graph node carrying:
  - 'text'        : str  — the cell's text content
  - 'column_id'   : int  — which column (0-indexed; -1 for the question node)
  - 'row_id'     : int   — which row    (0 = header row; -1 for the question node)
  - 'is_header'   : bool — true for header-row cells; false for the question node
  - 'original_id' : int  — sequential node index (kept for parity with other experiments)

Plus one additional node at the end of every graph: the **question node**, whose
text is `"Question: {q} Answer: {ans}"`. This node is the dataset's `prompt_node`.
Its text is masked everywhere except the answer span (handled by `compute_labels`).

Tables with more than `max_cells` cells are SKIPPED (not truncated) — truncation
would break the structural prior we're testing. Skipped count is logged.

Edges: cells in the same row are connected (a clique). Same for columns. This is
the underlying graph topology; the structural biases derive their signal from the
table metadata, not from these edges (K-hop is disabled in exp 003).
"""

import random
from typing import Tuple

import networkx as nx
import torch
from datasets import load_dataset

from ...utils import TextGraphDataset

# HF Hub path for WTQ. The historical "Stanford/wikitablequestions" is a
# script-based loader, which HF datasets no longer supports — load_dataset
# raises "Dataset scripts are no longer supported". Resolved alternative:
# lighteval/wikitablequestions ships pre-converted parquet with the right
# schema (question/answers/table.{header,rows}/table_md). One quirk: it
# merges the original train + test into a single "test" split (18486 rows),
# distinguishable by id prefix:
#   nt-*  → 14142 training-pool rows  (original train.tsv)
#   nu-*  → 4344  test rows           (original pristine-unseen-tables)
# We further carve ~17% of the nt-* pool into a validation split, matching
# the original train/dev ratio (~14149 train + ~2831 dev = 17% dev).
WTQ_HF_PATH = "lighteval/wikitablequestions"
WTQ_VAL_FRACTION = 0.17  # of the nt-* pool; deterministic via seed shuffle


def _normalize_answer_list(answers) -> str:
    """Join a multi-answer list into the canonical comma-separated string used at eval."""
    if isinstance(answers, list):
        return ", ".join(str(a) for a in answers)
    return str(answers)


def _build_graph_for_example(question: str, answer_str: str, header, rows) -> nx.DiGraph:
    """Build one cell-as-node graph for a WTQ example."""
    g = nx.DiGraph()
    n_cols = len(header)
    n_rows = len(rows) + 1  # +1 for the header row
    node_id = 0
    # Per-row node ids, so we can wire intra-row and intra-col edges
    row_to_nodes: list[list[int]] = []

    # Row 0 = header row
    row_nodes = []
    for col_id, head_text in enumerate(header):
        g.add_node(node_id,
                   text=str(head_text),
                   column_id=col_id,
                   row_id=0,
                   is_header=True,
                   original_id=node_id)
        row_nodes.append(node_id)
        node_id += 1
    row_to_nodes.append(row_nodes)

    # Data rows
    for r, row_vals in enumerate(rows, start=1):
        row_nodes = []
        for col_id in range(n_cols):
            cell_text = str(row_vals[col_id]) if col_id < len(row_vals) else ""
            g.add_node(node_id,
                       text=cell_text,
                       column_id=col_id,
                       row_id=r,
                       is_header=False,
                       original_id=node_id)
            row_nodes.append(node_id)
            node_id += 1
        row_to_nodes.append(row_nodes)

    # Question (prompt) node. Sentinel column/row -1 so the table-structural
    # biases see it as "not part of the table" and don't try to wire it.
    q_node = node_id
    g.add_node(q_node,
               text=f"Question: {question} Answer: {answer_str}",
               column_id=-1,
               row_id=-1,
               is_header=False,
               original_id=q_node)
    g.graph['prompt_node'] = q_node

    # Edges: cliques per row and per column (cells co-attend within their row/col
    # via the underlying graph topology). The structural biases don't depend on
    # these — only the optional K-hop gate would. We add them so the graph is
    # well-formed for any downstream code that inspects edges.
    for row_nodes in row_to_nodes:
        for i in range(len(row_nodes)):
            for j in range(i + 1, len(row_nodes)):
                g.add_edge(row_nodes[i], row_nodes[j])
                g.add_edge(row_nodes[j], row_nodes[i])
    for col_id in range(n_cols):
        col_nodes = [row_to_nodes[r][col_id] for r in range(n_rows) if col_id < len(row_to_nodes[r])]
        for i in range(len(col_nodes)):
            for j in range(i + 1, len(col_nodes)):
                g.add_edge(col_nodes[i], col_nodes[j])
                g.add_edge(col_nodes[j], col_nodes[i])

    return g


def load_data(
    tokenizer,
    max_cells: int = 200,
    max_train: int = -1,
    max_val: int = -1,
    max_test: int = -1,
    seed: int = 42,
) -> Tuple[TextGraphDataset, TextGraphDataset, TextGraphDataset]:
    """
    Load WikiTableQuestions and return (train, val, test) TextGraphDatasets
    with cells as nodes plus a question/prompt node per example.

    max_cells caps total cells per table (excluding the question node).
    max_train/val/test (>0) cap the size of each split for smoke runs.
    """
    rng = random.Random(seed)
    ds_raw = load_dataset(WTQ_HF_PATH)

    # lighteval/wikitablequestions merges train+test into one "test" split
    # (18486 rows). Re-split by id prefix (nt-* = train pool, nu-* = test
    # pool), then carve a deterministic ~17% val out of the train pool.
    pool = list(ds_raw['test'])
    nt_pool = [ex for ex in pool if ex['id'].startswith('nt-')]
    nu_pool = [ex for ex in pool if ex['id'].startswith('nu-')]
    rng.shuffle(nt_pool)  # before val carve so the carve is deterministic-by-seed
    n_val = int(len(nt_pool) * WTQ_VAL_FRACTION)
    val_pool = nt_pool[:n_val]
    train_pool = nt_pool[n_val:]
    test_pool = nu_pool

    pools = {
        'train':      (train_pool, max_train),
        'validation': (val_pool,   max_val),
        'test':       (test_pool,  max_test),
    }
    built: dict[str, TextGraphDataset] = {}
    skip_counts: dict[str, int] = {}

    for split_key, (examples, cap) in pools.items():
        graphs: list[nx.DiGraph] = []
        skipped = 0
        for ex in examples:
            question = ex['question']
            answers  = ex['answers']
            table    = ex['table']
            header   = table['header']
            rows     = table['rows']

            n_cells  = len(header) + sum(len(r) for r in rows)
            if n_cells > max_cells:
                skipped += 1
                continue

            ans_str  = _normalize_answer_list(answers)
            g = _build_graph_for_example(question, ans_str, header, rows)
            graphs.append(g)

            if cap > 0 and len(graphs) >= cap:
                break

        # Shuffle the train split for SGD diversity; val/test stay deterministic.
        if split_key == 'train':
            rng.shuffle(graphs)

        skip_counts[split_key] = skipped
        built[split_key] = TextGraphDataset(graphs, dataset_label=f"wtq-{split_key}")

    print(
        f"WTQ split counts (after max_cells={max_cells} filter):  "
        f"train={len(built['train'])} (skipped {skip_counts['train']})  "
        f"val={len(built['validation'])} (skipped {skip_counts['validation']})  "
        f"test={len(built['test'])} (skipped {skip_counts['test']})"
    )

    # Materialize structural features. Only table_metadata is strictly needed by
    # the exp 003 biases; the others are dummy zeros if a future ablation wants
    # to add SPD / RRWP / Magnetic alongside.
    for split_key, ds in built.items():
        ds.compute_table_metadata()
        ds.tokenize(tokenizer, max_length=64, add_eos=True)

    # Labels: prompt node text is "Question: ... Answer: {gold}". Mask everything
    # up through " Answer: " to -100; supervise the answer tokens.
    answer_marker = " Answer: "
    marker_token_ids = tokenizer(answer_marker, add_special_tokens=False)['input_ids']
    n_marker = len(marker_token_ids)

    def get_labels(example):
        input_ids = example['input_ids'][example['prompt_node']]
        labels = torch.tensor(input_ids, dtype=torch.long)
        # Find the FIRST occurrence of the answer-marker token sequence; mask
        # everything up to and including it. If we can't find it (rare edge
        # case from BPE merges), fall back to masking the first 75% as the
        # question prefix.
        ids = list(input_ids)
        cut = -1
        for i in range(len(ids) - n_marker + 1):
            if ids[i:i + n_marker] == marker_token_ids:
                cut = i + n_marker
                break
        if cut < 0:
            cut = int(0.75 * len(ids))
        labels[:cut] = -100
        return labels

    for ds in built.values():
        ds.compute_labels(get_labels)

    return built['train'], built['validation'], built['test']
