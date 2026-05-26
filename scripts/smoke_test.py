#!/usr/bin/env python3
"""Stage A smoke test for kgqa-001 (reproduction of GTLM Family Tree).

Per the runbook (graph-reasoning-llm/docs/reproduction_plan.md, Stage A):
  - import every module under src/models/ and src/utils/
  - construct a tiny 3-node NetworkX DiGraph with text + prompt_node
  - build a TextGraphDataset from it
  - run compute_shortest_path_distances, compute_rrwp, compute_magnetic_lap
  - tokenize and print token IDs

No GPU required. Exits 0 on success.
"""
import importlib
import pkgutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


def _import_tree(pkg_name: str) -> list[tuple[str, Exception]]:
    pkg = importlib.import_module(pkg_name)
    failed: list[tuple[str, Exception]] = []
    for _, modname, _ in pkgutil.walk_packages(pkg.__path__, prefix=pkg.__name__ + "."):
        try:
            importlib.import_module(modname)
            print(f"  OK  {modname}")
        except Exception as e:
            print(f"  ERR {modname}: {type(e).__name__}: {e}")
            failed.append((modname, e))
    return failed


def main() -> int:
    print("===== smoke_test.py =====")

    print("\n--- importing src.models ---")
    fails_m = _import_tree("src.models")
    print("\n--- importing src.utils ---")
    fails_u = _import_tree("src.utils")
    if fails_m or fails_u:
        print(f"\nIMPORT FAILURES: {len(fails_m) + len(fails_u)}")
        return 1
    print("All modules imported.")

    print("\n--- building tiny 3-node NetworkX DiGraph ---")
    import networkx as nx
    g = nx.DiGraph()
    g.add_node(0, text="Alice")
    g.add_node(1, text="Bob")
    g.add_node(2, text="Carol")
    g.add_edge(0, 1)
    g.add_edge(1, 2)
    g.add_edge(0, 2)
    g.graph["prompt_node"] = 0
    print(f"  nodes: {list(g.nodes(data=True))}")
    print(f"  edges: {list(g.edges())}")
    print(f"  graph attrs: {dict(g.graph)}")

    print("\n--- building TextGraphDataset ---")
    from src.utils import TextGraphDataset
    ds = TextGraphDataset([g])
    print(f"  len(ds)={len(ds)}")

    print("\n--- computing graph features ---")
    ds.compute_shortest_path_distances()
    print(f"  SPD shape: {ds[0]['shortest_path_dists'].shape}")
    ds.compute_rrwp(max_rrwp_steps=6)
    print(f"  RRWP shape: {ds[0]['rrwp'].shape}")
    ds.compute_magnetic_lap(q=0.25)
    print(f"  magnetic_V shape: {ds[0]['magnetic_V'].shape}")
    print(f"  magnetic_lambdas shape: {ds[0]['magnetic_lambdas'].shape}")

    print("\n--- tokenizing with meta-llama/Llama-3.2-1B ---")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
    ds.tokenize(tok)
    item = ds[0]
    print(f"  tokens-per-node lengths: {[len(x) for x in item['input_ids']]}")
    for i, ids in enumerate(item["input_ids"]):
        print(f"  node {i} ({g.nodes[i]['text']!r}) input_ids: {list(ids)}")

    print("\n===== smoke OK =====")
    return 0


if __name__ == "__main__":
    sys.exit(main())
