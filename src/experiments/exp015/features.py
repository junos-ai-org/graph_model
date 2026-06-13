"""Exp 014 — GTLM structural features (SPD / RRWP / Magnetic) per graph.

Thin wrappers over the repo's own utilities (src/utils/rrwp.py,
src/utils/magnetic_lap.py) plus a batched Floyd–Warshall, returning per-graph
tensors shaped exactly as GraphCollator expects:
    shortest_path_dists : (n, n)  int32   (32767 = unreachable, 0 diagonal)
    rrwp                : (n, n, max_rw_steps) float32
    magnetic_V          : (n, m_eff, 2) float32
    magnetic_lambdas    : (m_eff,) float32
"""
from __future__ import annotations

import importlib.util
import os

import networkx as nx
import numpy as np
import torch

# Import rrwp / magnetic_lap by file path: going through the `src.utils`
# package would execute src/utils/__init__.py, which drags in `datasets`
# (unavailable on the local dev Mac). These two modules only need
# torch/numpy/networkx.
_UTILS = os.path.join(os.path.dirname(__file__), "..", "..", "utils")


def _load(name):
    spec = importlib.util.spec_from_file_location(
        f"exp015_{name}", os.path.join(_UTILS, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


compute_rrwp = _load("rrwp").compute_rrwp
get_magnetic_laplacian_coords = _load("magnetic_lap").get_magnetic_laplacian_coords

UNREACHABLE = 32767


def batched_floyd_warshall(graphs: list[nx.DiGraph], device) -> list[torch.Tensor]:
    """Directed all-pairs shortest paths for a chunk of graphs (padded batch)."""
    counts = [g.number_of_nodes() for g in graphs]
    max_n = max(counts)
    B = len(graphs)
    dist = torch.full((B, max_n, max_n), float(UNREACHABLE), device=device)
    for i, g in enumerate(graphs):
        n = counts[i]
        adj = torch.from_numpy(nx.to_numpy_array(g)).to(device)
        d = dist[i, :n, :n]
        d[adj > 0] = 1.0
        d.fill_diagonal_(0)
    for k in range(max_n):
        dist = torch.min(dist, dist[:, :, k].unsqueeze(2) + dist[:, k, :].unsqueeze(1))
    dist = dist.clamp(max=UNREACHABLE)
    return [dist[i, :n, :n].to(torch.int32).cpu() for i, n in enumerate(counts)]


def compute_features_chunk(graphs: list[nx.DiGraph], device,
                           max_rw_steps: int = 8,
                           magnetic_q: float = 0.25) -> list[dict]:
    """Features for a chunk of graphs. Chunk by similar node count for speed."""
    use_gpu = device.type == "cuda"
    spds = batched_floyd_warshall(graphs, device)
    rrwps = compute_rrwp(graphs, max_distance=max_rw_steps, use_gpu=use_gpu)
    V_list, L_list = get_magnetic_laplacian_coords(graphs, q=magnetic_q,
                                                   use_gpu=use_gpu, m=0)
    out = []
    for g, spd, rrwp_flat, V, lams in zip(graphs, spds, rrwps, V_list, L_list):
        n = g.number_of_nodes()
        rrwp = torch.from_numpy(np.asarray(rrwp_flat, dtype=np.float32)) \
                    .reshape(n, n, max_rw_steps)
        V_t = torch.from_numpy(np.asarray(V, dtype=np.float32)).reshape(n, -1, 2)
        lam_t = torch.from_numpy(np.asarray(lams, dtype=np.float32)).reshape(-1)
        assert V_t.shape[1] == lam_t.shape[0], \
            f"magnetic shape mismatch: V {tuple(V_t.shape)} vs lambdas {tuple(lam_t.shape)}"
        for name, t in (("spd", spd.float()), ("rrwp", rrwp), ("magnetic_V", V_t),
                        ("magnetic_lambdas", lam_t)):
            if not torch.isfinite(t).all():
                raise FloatingPointError(f"non-finite values in {name} (n={n})")
        out.append({"shortest_path_dists": spd, "rrwp": rrwp,
                    "magnetic_V": V_t, "magnetic_lambdas": lam_t})
    return out
