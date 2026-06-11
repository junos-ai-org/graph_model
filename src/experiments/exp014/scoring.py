"""Exp 014 scoring — ported verbatim from dlm.eval (graph-reasoning-llm) so
exp014 predictions are graded identically to exp013's."""
from __future__ import annotations


def _normalize(s: str) -> str:
    return " ".join(s.lower().split()).strip(" .")


def denotation_match(pred: str, gold: list[str]) -> bool:
    p = _normalize(pred)
    return any(_normalize(g) and _normalize(g) in p for g in gold)
