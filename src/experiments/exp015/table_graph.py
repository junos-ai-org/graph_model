"""Exp 015 — table → graph construction (pure logic, no heavy deps).

User-specified construction, NOTHING table-specific beyond modeling the table
as a graph and handing it to GTLM's generic biases (SPD + RRWP + Magnetic):

  * every cell is a node (text = the cell string)
  * every column has a column node (text = the header string)
  * edges: cell -> its column node; cell -> cell to its LEFT (if any);
    cell -> cell to its RIGHT (if any)
    (each row is a bidirectional chain — every adjacent cell pair gets
    edges in BOTH directions — every cell also links to its column)
  * the question/answer is the GTLM prompt node — appended LAST, with NO
    graph edges (the question concerns the whole table; minimal faithful
    choice, documented in experiments/014 README)

Node index layout for a C-column, R-row table:
    0 .. C-1                 column nodes
    C + r*C + c              cell (r, c)   (row-major)
    C + R*C                  prompt node (== g.graph['prompt_node'])

Tokenization of the prompt node is PREFIX-STABLE: the question prefix and the
answer continuation are tokenized separately and concatenated, so the eval
prompt ids are exactly the training prefix ids.
"""
from __future__ import annotations

import networkx as nx

QUESTION_TEMPLATE = "\n\nQuestion: {q}\nAnswer:"


def build_table_graph(header: list[str], rows: list[list[str]]) -> nx.DiGraph:
    """Build the directed table graph (WITHOUT the prompt node).

    Ragged rows: short rows are padded with empty-text cells; cells BEYOND the
    header width are dropped — the same behavior as the dlm `plain` builder
    exp013 trained on (`rows[r][c] if c < len(rows[r]) else ""` over
    range(n_cols)), kept deliberately for comparability."""
    n_cols = len(header)
    g = nx.DiGraph()
    for c in range(n_cols):
        g.add_node(c, text=str(header[c]))
    for r, row in enumerate(rows):
        for c in range(n_cols):
            idx = n_cols + r * n_cols + c
            cell = str(row[c]) if c < len(row) else ""
            g.add_node(idx, text=cell)
            g.add_edge(idx, c)                  # cell -> its column node
            if c > 0:
                g.add_edge(idx, idx - 1)        # cell -> cell to its left
            if c < n_cols - 1:
                g.add_edge(idx, idx + 1)        # cell -> cell to its right
    return g


def add_prompt_node(g: nx.DiGraph, question: str) -> nx.DiGraph:
    """Append the prompt node (no edges) and mark it. Mutates and returns g."""
    p = g.number_of_nodes()
    g.add_node(p, text=QUESTION_TEMPLATE.format(q=question))
    g.graph["prompt_node"] = p
    return g


def tokenize_node_texts(tokenizer, g: nx.DiGraph) -> list[list[int]]:
    """Per-node token ids. Empty-text nodes get a single space token so every
    node owns at least one position (keeps _prepare_inputs simple/safe)."""
    out = []
    for i in range(g.number_of_nodes()):
        ids = tokenizer(g.nodes[i]["text"], add_special_tokens=False)["input_ids"]
        if not ids:
            ids = tokenizer(" ", add_special_tokens=False)["input_ids"]
        out.append(list(ids))
    return out


def prompt_ids_and_labels(tokenizer, question: str, answer: str,
                          eos_token_id: int) -> tuple[list[int], list[int]]:
    """(prompt_node_input_ids, labels) for training.

    ids    = tok(prefix) + tok(" " + answer) + [eos]
    labels = [-100]*len(prefix_ids) + answer_ids + [eos]   (same length as ids)

    Prefix-stable: eval uses tok(prefix) verbatim, so train/eval prompts agree
    token-for-token.
    """
    prefix_ids = tokenizer(QUESTION_TEMPLATE.format(q=question),
                           add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(f" {answer}", add_special_tokens=False)["input_ids"]
    ids = list(prefix_ids) + list(answer_ids) + [eos_token_id]
    labels = [-100] * len(prefix_ids) + list(answer_ids) + [eos_token_id]
    assert len(ids) == len(labels)
    return ids, labels


def total_tokens(node_input_ids: list[list[int]]) -> int:
    return sum(len(x) for x in node_input_ids)
