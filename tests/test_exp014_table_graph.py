"""Exp 014 pure-logic unit tests: table->graph construction, prompt
tokenization/labels, structural features, and the generation position fix.

Run:  pytest tests/test_exp014_table_graph.py -v   (CPU-only, no datasets/HF)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import networkx as nx
import pytest
import torch

from src.experiments.exp014.table_graph import (
    add_prompt_node, build_table_graph, prompt_ids_and_labels,
    tokenize_node_texts, total_tokens, QUESTION_TEMPLATE)
from src.experiments.exp014.features import (
    batched_floyd_warshall, compute_features_chunk, UNREACHABLE)
from src.experiments.exp014.scoring import denotation_match


class FakeTokenizer:
    """Char-codepoint tokenizer; '' -> no tokens. eos=1."""
    eos_token_id = 1

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


# ── Graph construction ────────────────────────────────────────────────────────
# 2 cols x 2 rows: nodes 0,1 = column nodes; 2,3 = row0 cells; 4,5 = row1 cells

HEADER = ["name", "age"]
ROWS = [["alice", "30"], ["bob", "41"]]


def test_node_layout_and_texts():
    g = build_table_graph(HEADER, ROWS)
    assert g.number_of_nodes() == 6
    assert g.nodes[0]["text"] == "name" and g.nodes[1]["text"] == "age"
    assert g.nodes[2]["text"] == "alice" and g.nodes[3]["text"] == "30"
    assert g.nodes[4]["text"] == "bob" and g.nodes[5]["text"] == "41"


def test_edges_exactly_cell_to_column_and_left():
    g = build_table_graph(HEADER, ROWS)
    expected = {
        (2, 0), (3, 1),          # row 0 cells -> column nodes
        (4, 0), (5, 1),          # row 1 cells -> column nodes
        (3, 2), (5, 4),          # cell -> left cell (row chains)
    }
    assert set(g.edges()) == expected


def test_ragged_row_padded_with_empty_cell():
    g = build_table_graph(HEADER, [["only"]])
    assert g.nodes[3]["text"] == ""           # missing 2nd cell becomes ""
    assert (3, 1) in g.edges() and (3, 2) in g.edges()


def test_prompt_node_is_last_and_isolated():
    g = add_prompt_node(build_table_graph(HEADER, ROWS), "who?")
    p = g.graph["prompt_node"]
    assert p == 6 and g.number_of_nodes() == 7
    assert g.degree(p) == 0
    assert g.nodes[p]["text"] == QUESTION_TEMPLATE.format(q="who?")


def test_spd_sanity_on_hand_built_table():
    g = build_table_graph(HEADER, ROWS)
    spd = batched_floyd_warshall([g], torch.device("cpu"))[0]
    assert spd[2, 0] == 1            # cell -> its column
    assert spd[3, 2] == 1            # cell -> left cell
    assert spd[3, 0] == 2            # right cell -> left column via chain
    assert spd[0, 2] == UNREACHABLE  # edges are directed: column is a sink
    assert spd[2, 1] == UNREACHABLE  # no path from col-0 cell to col-1 node
    assert (spd.diagonal() == 0).all()


# ── Tokenization / labels ─────────────────────────────────────────────────────

def test_tokenize_empty_cell_gets_space_token():
    tok = FakeTokenizer()
    g = add_prompt_node(build_table_graph(HEADER, [["x", ""]]), "q")
    ids = tokenize_node_texts(tok, g)
    assert all(len(x) >= 1 for x in ids)
    assert ids[3] == [ord(" ")]
    assert total_tokens(ids) == sum(len(x) for x in ids)


def test_prompt_labels_mask_question_supervise_answer_eos():
    tok = FakeTokenizer()
    ids, labels = prompt_ids_and_labels(tok, "q1", "42", tok.eos_token_id)
    prefix = tok(QUESTION_TEMPLATE.format(q="q1"))["input_ids"]
    ans = tok(" 42")["input_ids"]
    assert ids == prefix + ans + [tok.eos_token_id]
    assert labels[:len(prefix)] == [-100] * len(prefix)
    assert labels[len(prefix):] == ans + [tok.eos_token_id]
    assert len(ids) == len(labels)


def test_prompt_prefix_stability():
    """Eval prompt ids must be an exact prefix of train prompt ids."""
    tok = FakeTokenizer()
    ids, _ = prompt_ids_and_labels(tok, "the q", "ans", tok.eos_token_id)
    eval_ids = tok(QUESTION_TEMPLATE.format(q="the q"))["input_ids"]
    assert ids[:len(eval_ids)] == eval_ids


# ── Features ─────────────────────────────────────────────────────────────────

def test_compute_features_chunk_shapes_and_finiteness():
    gs = [add_prompt_node(build_table_graph(HEADER, ROWS), "q"),
          add_prompt_node(build_table_graph(["a"], [["1"], ["2"], ["3"]]), "q2")]
    feats = compute_features_chunk(gs, torch.device("cpu"))
    for g, f in zip(gs, feats):
        n = g.number_of_nodes()
        assert f["shortest_path_dists"].shape == (n, n)
        assert f["rrwp"].shape == (n, n, 8)
        assert f["magnetic_V"].shape[0] == n and f["magnetic_V"].shape[2] == 2
        assert f["magnetic_lambdas"].shape[0] == f["magnetic_V"].shape[1]
        assert torch.isfinite(f["rrwp"]).all()
        # RRWP step 0 is the identity
        assert torch.allclose(f["rrwp"][:, :, 0], torch.eye(n))


def test_rrwp_rows_are_probability_distributions():
    g = add_prompt_node(build_table_graph(HEADER, ROWS), "q")
    f = compute_features_chunk([g], torch.device("cpu"))[0]
    sums = f["rrwp"].sum(dim=1)                # (n, steps)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


# ── Scoring / generation fix ─────────────────────────────────────────────────

def test_denotation_match():
    assert denotation_match("The answer is Bob.", ["bob"])
    assert not denotation_match("alice", ["bob"])
    assert not denotation_match("anything", [""])


def test_position_ids_extension_logic():
    """The override extends positions monotonically past the prompt."""
    pos = torch.tensor([[0, 1, 2, 0, 1]])
    input_ids = torch.zeros(1, 8, dtype=torch.long)
    k = input_ids.shape[1] - pos.shape[1]
    step = torch.arange(1, k + 1).unsqueeze(0)
    ext = torch.cat([pos, pos[:, -1:].expand(1, k) + step], dim=1)
    assert ext.tolist() == [[0, 1, 2, 0, 1, 2, 3, 4]]
