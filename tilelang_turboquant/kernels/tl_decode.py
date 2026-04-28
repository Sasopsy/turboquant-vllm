"""C6 decode-kernel scaffold for future TileLang integration."""

from __future__ import annotations

import torch

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.kernels.scaffold import raise_kernel_not_implemented


def is_decode_kernel_available() -> bool:
    """Return whether a real TileLang decode kernel is wired in."""

    return False


def tl_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor | None,
    key_centroids: torch.Tensor,
    value_centroids: torch.Tensor,
    S_matrix: torch.Tensor,
    scale: float,
    cfg: TileLangTQConfig,
    mid_o_buf: torch.Tensor,
    lse_buf: torch.Tensor,
    output_buf: torch.Tensor,
    max_num_kv_splits: int,
) -> torch.Tensor:
    """Scaffold wrapper for the future TileLang decode kernel."""

    del (
        query,
        kv_cache,
        block_table,
        seq_lens,
        rotation,
        rotation_t,
        key_centroids,
        value_centroids,
        S_matrix,
        scale,
        cfg,
        mid_o_buf,
        lse_buf,
        output_buf,
        max_num_kv_splits,
    )
    raise_kernel_not_implemented()
