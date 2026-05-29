"""
Model-agnostic graph attention bias module.

Each bias type is a BaseBias subclass registered in BIAS_TYPES.
GraphAttentionBias iterates that list, instantiates every enabled type, and
accumulates their outputs into a single (B, H, N, N) node-level tensor.

To add a new bias type:
  1. Create a class that inherits BaseBias.
  2. Set config_key to the attribute name on bias_config that enables it.
  3. Implement forward(**kwargs) → (B, H, N, N) tensor (or None if input absent).
  4. Append the class to BIAS_TYPES — no other changes needed.

Supported bias types
--------------------
SPDBias       - learnable lookup table keyed by shortest-path distance
LaplacianBias - learnable scalar weight × L2 distance between spectral embeddings
RWSEBias      - same pattern for random-walk structural encodings
RRWPBias      - small MLP applied to multi-hop random-walk probability vectors
MagneticBias  - complex-eigenvector-based directional encoding via deep-set MLP
K-hop gate (hard) - -inf for node pairs more than K hops apart; K=0 = disabled
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


# ── Base class ────────────────────────────────────────────────────────────────

class BaseBias(nn.Module):
    """
    Base class for a single graph attention bias type.

    Subclass protocol:
      config_key : str   — attribute on bias_config that enables this type.
      forward(**kwargs)  — consumes whatever it needs from the shared kwargs
                           dict and returns a (B, H, N, N) float tensor,
                           or None when its required input is absent.
    """

    config_key: str = ""

    @classmethod
    def is_enabled(cls, bias_config) -> bool:
        return bool(getattr(bias_config, cls.config_key, False))


# ── Bias type implementations ─────────────────────────────────────────────────

class SPDBias(BaseBias):
    """Learnable lookup table indexed by shortest-path distance."""

    config_key = 'spd'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.max_spd = getattr(bias_config, 'max_spd', 32)
        self.weights = nn.Parameter(torch.zeros(self.max_spd, num_heads))

    def forward(self, *, dtype, device, spd=None, **kwargs) -> Optional[torch.Tensor]:
        if spd is None:
            return None
        non_zero = (spd > 0).unsqueeze(1)
        idx = torch.clamp(spd - 1, 0, self.max_spd - 1)
        b = F.embedding(idx, self.weights).permute(0, 3, 1, 2).to(dtype)
        return b * non_zero                                         # (B, H, N, N)


class LaplacianBias(BaseBias):
    """Learnable scalar weight x pairwise L2 distance between spectral embeddings."""

    config_key = 'laplacian'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_heads) * 0.02)

    def forward(self, *, dtype, device, laplacian=None, **kwargs) -> Optional[torch.Tensor]:
        if laplacian is None:
            return None
        dist = torch.cdist(laplacian, laplacian, p=2.0)            # (B, N, N)
        return dist.unsqueeze(1) * self.weights.view(-1, 1, 1).to(dtype)


class RWSEBias(BaseBias):
    """Same scalar-weight pattern as LaplacianBias, for random-walk SE features."""

    config_key = 'rwse'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(num_heads) * 0.02)

    def forward(self, *, dtype, device, rwse=None, **kwargs) -> Optional[torch.Tensor]:
        if rwse is None:
            return None
        dist = torch.cdist(rwse, rwse, p=2.0)                      # (B, N, N)
        return dist.unsqueeze(1) * self.weights.view(-1, 1, 1).to(dtype)


class RRWPBias(BaseBias):
    """Small MLP applied to multi-hop random-walk probability vectors."""

    config_key = 'rrwp'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        max_rw_steps = getattr(bias_config, 'max_rw_steps', 8)
        hidden = 4 * max_rw_steps
        self.proj = nn.Sequential(
            nn.Linear(max_rw_steps, hidden, bias=True),
            nn.SiLU(),
            nn.Linear(hidden, num_heads, bias=True),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    def forward(self, *, dtype, device, rrwp=None, **kwargs) -> Optional[torch.Tensor]:
        if rrwp is None:
            return None
        b = self.proj(rrwp).permute(0, 3, 1, 2).contiguous()      # (B, H, N, N)
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


class MagneticBias(BaseBias):
    """Complex-eigenvector-based directional encoding via a deep-set MLP."""

    config_key = 'magnetic'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        magnetic_dim = getattr(bias_config, 'magnetic_dim', 32)
        self.lambda_lin = nn.Linear(1, head_dim, bias=True)
        self.deep_set = nn.Sequential(
            nn.Linear(head_dim * 2, magnetic_dim, bias=True),
            nn.SiLU(),
        )
        self.proj = nn.Sequential(
            nn.Linear(magnetic_dim * 2, magnetic_dim, bias=True),
            nn.SiLU(),
            nn.Linear(magnetic_dim, num_heads, bias=True),
        )
        nn.init.zeros_(self.proj[2].weight)
        nn.init.zeros_(self.proj[2].bias)
        self.proj[2]._is_hf_initialized = True

    def forward(
        self, *, dtype, device,
        magnetic: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        num_nodes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if magnetic is None or num_nodes is None:
            return None
        V, lambdas = magnetic # (B,N,N,2), (B,N)
        V_real, V_imag = V[..., 0], V[..., 1] # (B, N, N) each

        h_i   = self.lambda_lin(lambdas.unsqueeze(-1))                                          # (B, M, head_dim)
        valid = (torch.arange(lambdas.shape[1], device=device).unsqueeze(0)
                 < num_nodes.unsqueeze(1))                                                       # (B, M) bool
        # Divide by the number of valid eigenvalues, not num_nodes.
        # With full eigenvectors (M=N) these are equal; with truncated (M<N) they differ.
        n_valid = valid.float().sum(dim=1, keepdim=True).unsqueeze(-1).clamp(min=1)             # (B, 1, 1)
        h_avg   = (h_i * valid.unsqueeze(-1)).sum(1, keepdim=True) / n_valid                   # (B, 1, head_dim)

        phi = self.deep_set(torch.cat([h_i, h_avg.expand_as(h_i)], dim=-1))                                 # (B, N, magnetic_dim)

        real = (torch.einsum('bil,bjl,blk->bijk', V_real, V_real, phi) + torch.einsum('bil,bjl,blk->bijk', V_imag, V_imag, phi))
        imag = (torch.einsum('bil,bjl,blk->bijk', V_imag, V_real, phi) - torch.einsum('bil,bjl,blk->bijk', V_real, V_imag, phi))

        b = (self.proj(torch.cat([real, imag], dim=-1)).permute(0, 3, 1, 2).contiguous())                    # (B, H, N, N)
        diag = torch.eye(b.shape[-1], device=device, dtype=torch.bool)
        return b.masked_fill(diag.unsqueeze(0).unsqueeze(0), 0.0)


# ── Table-structural biases (exp 003) ─────────────────────────────────────────
#
# Four symmetric, one-scalar-per-head biases for tabular reasoning. See
# docs/reproduction_plan_003.md (graph-reasoning-llm research repo) for the
# (N,N) matrix picture and the design rationale.
#
# Inputs expected on the batch dict (provided by TextGraphDataset.compute_table_metadata
# + GraphCollator padding):
#   header_node_id_per_cell : (B, N) long, the col-header node id for each cell
#                              (-1 for header-row cells themselves or padding)
#   row_anchor_id_per_cell  : (B, N) long, the row-anchor node id for each cell
#                              (anchor cells point to themselves; -1 for header-row / padding)
#   is_header               : (B, N) bool, true for header-row cells
#   is_row_anchor           : (B, N) bool, true for first cell of each data row
#
# All four biases follow the same pattern: build an (B, N, N) bool mask from
# the metadata, AND-out the diagonal, multiply by a per-head learnable scalar.

def _zero_diag(M: torch.Tensor, N: int, device: torch.device) -> torch.Tensor:
    """AND-out the diagonal of an (B, N, N) bool mask."""
    eye = torch.eye(N, dtype=torch.bool, device=device).unsqueeze(0)
    return M & ~eye


class ColumnHeaderBias(BaseBias):
    """α: cell ↔ its column header (symmetric, one learnable scalar per head)."""

    config_key = 'col_header'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self, *, dtype, device,
        header_node_id_per_cell: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if header_node_id_per_cell is None:
            return None
        B, N = header_node_id_per_cell.shape
        H = self.alpha.shape[0]
        j_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
        header_i = header_node_id_per_cell.unsqueeze(2).expand(B, N, N)
        # M[b,i,j] = "j is i's column header"
        M = (header_i == j_idx) & (header_i >= 0)
        # Symmetric: also include "i is j's column header"
        M = _zero_diag(M | M.transpose(1, 2), N, device)
        alpha = self.alpha.to(dtype=dtype, device=device).view(1, H, 1, 1)
        return M.unsqueeze(1).to(dtype) * alpha                       # (B, H, N, N)


class HeaderCliqueBias(BaseBias):
    """ε: column header ↔ column header (clique, symmetric, one scalar per head)."""

    config_key = 'header_clique'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self, *, dtype, device,
        is_header: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if is_header is None:
            return None
        B, N = is_header.shape
        H = self.epsilon.shape[0]
        # M[b,i,j] = is_header[i] AND is_header[j]
        M = is_header.unsqueeze(2) & is_header.unsqueeze(1)
        M = _zero_diag(M, N, device)
        eps = self.epsilon.to(dtype=dtype, device=device).view(1, H, 1, 1)
        return M.unsqueeze(1).to(dtype) * eps


class RowAnchorAggBias(BaseBias):
    """γ: row anchor ↔ every cell in its row (symmetric, one scalar per head).

    Convention: row_anchor_id_per_cell[anchor] = anchor's own id (anchors point
    to themselves). -1 for header-row cells and padding.
    """

    config_key = 'row_anchor_agg'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self, *, dtype, device,
        row_anchor_id_per_cell: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if row_anchor_id_per_cell is None:
            return None
        B, N = row_anchor_id_per_cell.shape
        H = self.gamma.shape[0]
        j_idx = torch.arange(N, device=device).view(1, 1, N).expand(B, N, N)
        anchor_i = row_anchor_id_per_cell.unsqueeze(2).expand(B, N, N)
        # M[b,i,j] = "j is i's row anchor"
        M = (anchor_i == j_idx) & (anchor_i >= 0)
        # Symmetric: also "i is j's row anchor"
        M = _zero_diag(M | M.transpose(1, 2), N, device)
        gamma = self.gamma.to(dtype=dtype, device=device).view(1, H, 1, 1)
        return M.unsqueeze(1).to(dtype) * gamma


class RowAnchorSpineBias(BaseBias):
    """δ: row anchor ↔ row anchor (clique, symmetric, one scalar per head)."""

    config_key = 'row_anchor_spine'

    def __init__(self, num_heads: int, head_dim: int, bias_config):
        super().__init__()
        self.delta = nn.Parameter(torch.zeros(num_heads))

    def forward(
        self, *, dtype, device,
        is_row_anchor: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        if is_row_anchor is None:
            return None
        B, N = is_row_anchor.shape
        H = self.delta.shape[0]
        # M[b,i,j] = is_row_anchor[i] AND is_row_anchor[j]
        M = is_row_anchor.unsqueeze(2) & is_row_anchor.unsqueeze(1)
        M = _zero_diag(M, N, device)
        delta = self.delta.to(dtype=dtype, device=device).view(1, H, 1, 1)
        return M.unsqueeze(1).to(dtype) * delta


# ── Registration list ─────────────────────────────────────────────────────────

BIAS_TYPES: list[type[BaseBias]] = [
    SPDBias,
    LaplacianBias,
    RWSEBias,
    RRWPBias,
    MagneticBias,
    ColumnHeaderBias,
    HeaderCliqueBias,
    RowAnchorAggBias,
    RowAnchorSpineBias,
]
"""Add new bias types here — GraphAttentionBias picks them up automatically."""


# ── Top-level module ──────────────────────────────────────────────────────────

class GraphAttentionBias(nn.Module):
    """
    Per-layer, model-agnostic graph attention bias module.

    Instantiates every enabled bias type from BIAS_TYPES, accumulates their
    (B, H, N, N) outputs, then optionally applies a hard K-hop gate.
    Returns None when the result would be all-zero with no gate applied.
    """

    def __init__(
        self,
        num_heads:   int,
        head_dim:    int,
        layer_idx:   int,
        bias_config,
        k_hop:       int = 0,
    ):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_heads = num_heads
        self.k_hop     = k_hop

        active_types = [cls for cls in BIAS_TYPES if cls.is_enabled(bias_config)]
        self.bias_modules = nn.ModuleList(
            [cls(num_heads, head_dim, bias_config) for cls in active_types]
        )
        self._active = {cls.config_key for cls in active_types}

    # ── Convenience flags (used by the attention layer to skip fetching data) ──

    @property
    def has_soft_bias(self) -> bool:
        return len(self.bias_modules) > 0

    @property
    def require_spd(self) -> bool:       return 'spd'       in self._active

    @property
    def require_laplacian(self) -> bool: return 'laplacian' in self._active

    @property
    def require_rwse(self) -> bool:      return 'rwse'      in self._active

    @property
    def require_rrwp(self) -> bool:      return 'rrwp'      in self._active

    @property
    def require_magnetic(self) -> bool:  return 'magnetic'  in self._active

    @property
    def require_col_header(self) -> bool:       return 'col_header'       in self._active

    @property
    def require_header_clique(self) -> bool:    return 'header_clique'    in self._active

    @property
    def require_row_anchor_agg(self) -> bool:   return 'row_anchor_agg'   in self._active

    @property
    def require_row_anchor_spine(self) -> bool: return 'row_anchor_spine' in self._active

    @property
    def require_table_metadata(self) -> bool:
        """True if any table-structural bias is enabled."""
        return any(self._active & {'col_header', 'header_clique',
                                    'row_anchor_agg', 'row_anchor_spine'})

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        dtype:       torch.dtype,
        device:      torch.device,
        num_nodes:   Optional[torch.Tensor]                           = None,
        spd:         Optional[torch.Tensor]                           = None,
        laplacian:   Optional[torch.Tensor]                           = None,
        rwse:        Optional[torch.Tensor]                           = None,
        rrwp:        Optional[torch.Tensor]                           = None,
        magnetic:    Optional[Tuple[torch.Tensor, torch.Tensor]]      = None,
        header_node_id_per_cell: Optional[torch.Tensor]               = None,
        row_anchor_id_per_cell:  Optional[torch.Tensor]               = None,
        is_header:               Optional[torch.Tensor]               = None,
        is_row_anchor:           Optional[torch.Tensor]               = None,
        k_hop_mask:  Optional[torch.Tensor]                           = None,
        cache_dict:  Optional[dict]                                   = None,
    ) -> Optional[torch.Tensor]:
        """
        Compute the (B, H, N, N) node-level attention bias.

        Returns None when the result would be all-zero with no gate applied
        (no enabled soft biases and K=0 or no mask provided).
        """
        if not self.has_soft_bias and (self.k_hop == 0 or k_hop_mask is None):
            return None

        cache_key = f'cached_graph_bias_{self.layer_idx}'
        if not self.training and cache_dict is not None and cache_key in cache_dict:
            return cache_dict[cache_key]

        node_bias = None
        for module in self.bias_modules:
            b = module(
                dtype=dtype, device=device,
                num_nodes=num_nodes, spd=spd, laplacian=laplacian,
                rwse=rwse, rrwp=rrwp, magnetic=magnetic,
                header_node_id_per_cell=header_node_id_per_cell,
                row_anchor_id_per_cell=row_anchor_id_per_cell,
                is_header=is_header,
                is_row_anchor=is_row_anchor,
            )
            if b is not None:
                node_bias = b if node_bias is None else node_bias + b

        node_bias = self._apply_k_hop_gate(node_bias, k_hop_mask, dtype, device)

        if node_bias is not None and cache_dict is not None:
            cache_dict[cache_key] = node_bias

        return node_bias

    # ── K-hop gate ────────────────────────────────────────────────────────────

    def _apply_k_hop_gate(
        self,
        node_bias:   Optional[torch.Tensor],
        k_hop_mask:  Optional[torch.Tensor],
        dtype:       torch.dtype,
        device:      torch.device,
    ) -> Optional[torch.Tensor]:
        """Apply -inf to positions outside the K-hop neighbourhood."""
        if self.k_hop == 0 or k_hop_mask is None:
            return node_bias

        gate = k_hop_mask.unsqueeze(1)                             # (B, 1, N, N)

        if node_bias is None:
            B, N, _ = k_hop_mask.shape
            node_bias = torch.zeros(B, self.num_heads, N, N, dtype=dtype, device=device)

        return node_bias.masked_fill(~gate, torch.finfo(dtype).min)