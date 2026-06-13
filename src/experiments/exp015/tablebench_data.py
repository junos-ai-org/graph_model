"""Exp 014 — TableBench / TableInstruct-DP loading.

FAITHFUL port of graph-reasoning-llm src/dlm/data/datasets/tablebench.py
(exp013's loader): same filters, same pool construction order, same seeded
17% val carve (random.Random(seed).shuffle over the pool list — the
permutation depends only on seed and length, so train/val membership is
positionally IDENTICAL to exp013's). Train pool expected: 4506 examples
(seed=42, max_cells=200).

Differences from the dlm original (additive only):
  * examples are plain dicts, not dlm RawExample dataclasses
  * the test loader can yield full DP-file metadata (id / qtype / qsubtype)
    and can disable the max_cells filter (exp013's DP eval did not apply it)
"""
from __future__ import annotations

import json
import random
from typing import Iterable

TABLEBENCH_HF_PATH = "Multilingual-Multimodal-NLP/TableBench"
TABLEINSTRUCT_HF_PATH = "Multilingual-Multimodal-NLP/TableInstruct"
DP_FILENAME = "TableBench_DP.jsonl"
VAL_FRACTION = 0.17
FINAL_ANSWER_MARKER = "Final Answer:"


def parse_final_answer(response: str) -> str:
    if FINAL_ANSWER_MARKER not in response:
        return ""
    tail = response.rsplit(FINAL_ANSWER_MARKER, 1)[1].strip().splitlines()
    return tail[0].strip() if tail else ""


def _parse_table(raw) -> tuple[list[str], list[list[str]]]:
    obj = json.loads(raw) if isinstance(raw, str) else raw
    header = [str(c) for c in obj["columns"]]
    rows = [[str(c) for c in r] for r in obj["data"]]
    return header, rows


def _key(question: str, header: list[str], rows: list[list[str]]) -> str:
    return json.dumps([question.strip(), header, rows])


def _carve(pool: list, split: str, seed: int) -> list:
    pool = list(pool)
    random.Random(seed).shuffle(pool)
    n_val = int(len(pool) * VAL_FRACTION)
    return pool[:n_val] if split == "val" else pool[n_val:]


class TableBenchDP:
    def __init__(self, seed: int = 42, exclude_viz: bool = True,
                 max_cells: int = 200):
        self.seed = seed
        self.exclude_viz = exclude_viz
        self.max_cells = max_cells
        self._test_cache: list | None = None
        self._pool_cache: list | None = None

    def _is_viz(self, e: dict) -> bool:
        return self.exclude_viz and str(e.get("qtype", "")).lower() == "visualization"

    def _too_big(self, header, rows) -> bool:
        return len(header) + sum(len(r) for r in rows) > self.max_cells

    def _test_rows(self) -> list[dict]:
        """All non-viz DP test rows with full metadata, BEFORE max_cells."""
        if self._test_cache is not None:
            return self._test_cache
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(TABLEBENCH_HF_PATH, DP_FILENAME, repo_type="dataset")
        out = []
        with open(path) as f:
            for line in f:
                e = json.loads(line)
                if self._is_viz(e):
                    continue
                header, rows = _parse_table(e["table"])
                out.append({"id": e["id"], "qtype": e["qtype"],
                            "qsubtype": e["qsubtype"], "question": e["question"],
                            "header": header, "rows": rows,
                            "answer": str(e.get("answer", ""))})
        self._test_cache = out
        return out

    def test(self, apply_max_cells: bool = False) -> Iterable[dict]:
        """Non-viz DP test rows in file order (exp013 DP-eval iteration order)."""
        for e in self._test_rows():
            if apply_max_cells and self._too_big(e["header"], e["rows"]):
                continue
            yield e

    def train_pool(self) -> list[dict]:
        """The leakage-guarded TableInstruct-DP pool, in exp013's exact order."""
        if self._pool_cache is not None:
            return self._pool_cache
        from datasets import load_dataset
        test_keys = {_key(e["question"], e["header"], e["rows"])
                     for e in self._test_rows()}
        pool, n_noans, n_leak, n_big = [], 0, 0, 0
        for e in load_dataset(TABLEINSTRUCT_HF_PATH, split="train"):
            if e["instruction_type"] != "DP" or self._is_viz(e):
                continue
            ans = parse_final_answer(e["response"])
            if not ans:
                n_noans += 1
                continue
            header, rows = _parse_table(e["table"])
            if _key(e["question"], header, rows) in test_keys:
                n_leak += 1
                continue
            if self._too_big(header, rows):
                n_big += 1
                continue
            pool.append({"question": e["question"], "header": header,
                         "rows": rows, "answer": ans})
        if not pool:
            raise RuntimeError("tablebench train pool is EMPTY — label drift?")
        print(f"tablebench pool={len(pool)} DP examples (dropped {n_noans} "
              f"no-answer, {n_leak} test-overlap, {n_big} over "
              f"max_cells={self.max_cells})")
        self._pool_cache = pool
        return pool

    def split(self, which: str) -> list[dict]:
        if which not in ("train", "val"):
            raise ValueError(f"unknown split {which!r}")
        return _carve(self.train_pool(), which, self.seed)
