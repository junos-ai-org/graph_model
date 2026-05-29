"""
Unit tests for the four exp-003 table-structural biases:
  ColumnHeaderBias (α), HeaderCliqueBias (ε),
  RowAnchorAggBias (γ), RowAnchorSpineBias (δ).

Run with:  pytest tests/test_table_bias.py -v
from the repo root (graph_model/).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
import torch

from src.models.graph_bias import (
    ColumnHeaderBias,
    HeaderCliqueBias,
    RowAnchorAggBias,
    RowAnchorSpineBias,
)


class _BiasConfig:
    """Minimal stand-in — none of these biases consult bias_config in __init__."""


# ─── A hand-built 3-col × 2-data-row table ─────────────────────────────────────
# Node layout (B=1, N=9):
#   ids:        [ 0,  1,  2,  3,  4,  5,  6,  7,  8 ]
#   row_id:     [ 0,  0,  0,  1,  1,  1,  2,  2,  2 ]
#   col_id:     [ 0,  1,  2,  0,  1,  2,  0,  1,  2 ]
#   is_header:  [ T,  T,  T,  F,  F,  F,  F,  F,  F ]
#   anchor:     [ F,  F,  F,  T,  F,  F,  T,  F,  F ]
#
#   header_node_id_per_cell[i]:
#     headers (0,1,2) → -1
#     data: each cell points to header at its col → [-1,-1,-1, 0, 1, 2, 0, 1, 2]
#
#   row_anchor_id_per_cell[i]:
#     headers (0,1,2) → -1
#     anchors (3, 6) → themselves
#     other data: point to their row's anchor → [-1,-1,-1, 3, 3, 3, 6, 6, 6]

@pytest.fixture
def small_table_batch():
    header_node_id = torch.tensor([[-1, -1, -1,  0,  1,  2,  0,  1,  2]], dtype=torch.long)
    row_anchor_id  = torch.tensor([[-1, -1, -1,  3,  3,  3,  6,  6,  6]], dtype=torch.long)
    is_header      = torch.tensor([[ True,  True,  True, False, False, False, False, False, False]])
    is_row_anchor  = torch.tensor([[False, False, False,  True, False, False,  True, False, False]])
    return {
        'header_node_id_per_cell': header_node_id,
        'row_anchor_id_per_cell':  row_anchor_id,
        'is_header':               is_header,
        'is_row_anchor':           is_row_anchor,
    }


# ─── Shared assertions ───────────────────────────────────────────────────────

def _assert_shape_symmetric_diag_zero(b: torch.Tensor, B: int, H: int, N: int):
    assert b.shape == (B, H, N, N), f"expected {(B,H,N,N)}, got {b.shape}"
    # Symmetric
    assert torch.allclose(b, b.transpose(-1, -2)), "bias is not symmetric"
    # Diagonal is zero
    diag = b.diagonal(dim1=-2, dim2=-1)
    assert torch.all(diag == 0), "diagonal is nonzero"


# ─── ColumnHeaderBias (α) ─────────────────────────────────────────────────────

class TestColumnHeaderBias:
    H, N = 4, 9

    def _module(self):
        return ColumnHeaderBias(num_heads=self.H, head_dim=64, bias_config=_BiasConfig())

    def test_zero_init_returns_zero_bias(self, small_table_batch):
        m = self._module()
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        assert torch.all(b == 0), "alpha=0 should produce identically-zero bias"

    def test_shape_symmetric_diag_zero(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.alpha.fill_(0.5)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        _assert_shape_symmetric_diag_zero(b, 1, self.H, self.N)

    def test_correct_nonzero_positions(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.alpha.fill_(1.0)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        # Expected: cells 3,4,5 (data row 1) point to headers 0,1,2;
        #           cells 6,7,8 (data row 2) point to headers 0,1,2.
        # 6 (cell→header) entries × 2 (symmetric) = 12 nonzero entries per head.
        nonzero_mask = (b[0, 0] != 0)
        assert nonzero_mask.sum().item() == 12
        # Spot-check a specific pair: cell 4 (col 1, row 1) ↔ header 1.
        assert b[0, 0, 4, 1].item() == 1.0
        assert b[0, 0, 1, 4].item() == 1.0
        # Spot-check a NON-pair: cell 4 ↔ header 0 (different column).
        assert b[0, 0, 4, 0].item() == 0.0
        # Header ↔ header should be 0 (that's HeaderCliqueBias's job).
        assert b[0, 0, 0, 1].item() == 0.0


# ─── HeaderCliqueBias (ε) ─────────────────────────────────────────────────────

class TestHeaderCliqueBias:
    H, N = 4, 9

    def _module(self):
        return HeaderCliqueBias(num_heads=self.H, head_dim=64, bias_config=_BiasConfig())

    def test_zero_init_returns_zero(self, small_table_batch):
        m = self._module()
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        assert torch.all(b == 0)

    def test_shape_symmetric_diag_zero(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.epsilon.fill_(0.3)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        _assert_shape_symmetric_diag_zero(b, 1, self.H, self.N)

    def test_clique_over_headers_only(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.epsilon.fill_(1.0)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        # 3 headers form a clique → 3*2 = 6 nonzero off-diagonal entries per head.
        nonzero_mask = (b[0, 0] != 0)
        assert nonzero_mask.sum().item() == 6
        # All nonzero entries are among headers {0, 1, 2}.
        nonzero_idx = nonzero_mask.nonzero(as_tuple=False)
        for r, c in nonzero_idx.tolist():
            assert r in (0, 1, 2) and c in (0, 1, 2) and r != c


# ─── RowAnchorAggBias (γ) ─────────────────────────────────────────────────────

class TestRowAnchorAggBias:
    H, N = 4, 9

    def _module(self):
        return RowAnchorAggBias(num_heads=self.H, head_dim=64, bias_config=_BiasConfig())

    def test_zero_init_returns_zero(self, small_table_batch):
        m = self._module()
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        assert torch.all(b == 0)

    def test_shape_symmetric_diag_zero(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.gamma.fill_(0.7)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        _assert_shape_symmetric_diag_zero(b, 1, self.H, self.N)

    def test_anchor_to_row_cells_only(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.gamma.fill_(1.0)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        # Row 1: anchor 3 ↔ {4, 5}  → 2 pairs × 2 = 4 entries
        # Row 2: anchor 6 ↔ {7, 8}  → 4 entries
        # Total: 8 nonzero entries per head.
        nonzero_mask = (b[0, 0] != 0)
        assert nonzero_mask.sum().item() == 8
        # Spot-checks
        assert b[0, 0, 3, 4].item() == 1.0
        assert b[0, 0, 4, 3].item() == 1.0
        assert b[0, 0, 6, 7].item() == 1.0
        # Non-anchor non-anchor pair within a row should be zero
        assert b[0, 0, 4, 5].item() == 0.0
        # Cross-row anchor↔cell should be zero
        assert b[0, 0, 3, 7].item() == 0.0


# ─── RowAnchorSpineBias (δ) ───────────────────────────────────────────────────

class TestRowAnchorSpineBias:
    H, N = 4, 9

    def _module(self):
        return RowAnchorSpineBias(num_heads=self.H, head_dim=64, bias_config=_BiasConfig())

    def test_zero_init_returns_zero(self, small_table_batch):
        m = self._module()
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        assert torch.all(b == 0)

    def test_shape_symmetric_diag_zero(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.delta.fill_(0.4)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        _assert_shape_symmetric_diag_zero(b, 1, self.H, self.N)

    def test_anchor_clique_only(self, small_table_batch):
        m = self._module()
        with torch.no_grad():
            m.delta.fill_(1.0)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **small_table_batch)
        # 2 anchors (3, 6) form a clique → 2 nonzero off-diagonal entries per head.
        nonzero_mask = (b[0, 0] != 0)
        assert nonzero_mask.sum().item() == 2
        assert b[0, 0, 3, 6].item() == 1.0
        assert b[0, 0, 6, 3].item() == 1.0


# ─── Padding sanity ───────────────────────────────────────────────────────────

def test_padding_is_ignored():
    """
    Pad a single 9-node example out to N=12 (3 padding positions).
    The collator's sentinels (-1 / False) should make every bias treat the
    padding positions as inert: no nonzero entries involving them.
    """
    N_real, N_pad = 9, 12
    header_node_id = torch.full((1, N_pad), -1, dtype=torch.long)
    header_node_id[0, :N_real] = torch.tensor([-1, -1, -1,  0,  1,  2,  0,  1,  2])
    row_anchor_id  = torch.full((1, N_pad), -1, dtype=torch.long)
    row_anchor_id[0, :N_real]  = torch.tensor([-1, -1, -1,  3,  3,  3,  6,  6,  6])
    is_header = torch.zeros((1, N_pad), dtype=torch.bool)
    is_header[0, :N_real] = torch.tensor([True, True, True, False, False, False, False, False, False])
    is_row_anchor = torch.zeros((1, N_pad), dtype=torch.bool)
    is_row_anchor[0, :N_real] = torch.tensor([False, False, False, True, False, False, True, False, False])

    inputs = {
        'header_node_id_per_cell': header_node_id,
        'row_anchor_id_per_cell':  row_anchor_id,
        'is_header':               is_header,
        'is_row_anchor':           is_row_anchor,
    }

    H = 2
    for cls, param_name in [
        (ColumnHeaderBias,    'alpha'),
        (HeaderCliqueBias,    'epsilon'),
        (RowAnchorAggBias,    'gamma'),
        (RowAnchorSpineBias,  'delta'),
    ]:
        m = cls(num_heads=H, head_dim=64, bias_config=_BiasConfig())
        with torch.no_grad():
            getattr(m, param_name).fill_(1.0)
        b = m(dtype=torch.float32, device=torch.device('cpu'), **inputs)
        # No row or column involving padding positions [N_real, N_pad) should be nonzero
        assert torch.all(b[0, 0, N_real:, :] == 0), f"{cls.__name__}: padding rows nonzero"
        assert torch.all(b[0, 0, :, N_real:] == 0), f"{cls.__name__}: padding cols nonzero"
