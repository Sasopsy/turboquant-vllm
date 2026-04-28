"""C6 store-kernel scaffold for future TileLang integration."""

from __future__ import annotations

import torch

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.kernels.scaffold import raise_kernel_not_implemented


def is_store_kernel_available() -> bool:
    """Return whether a real TileLang store kernel is wired in."""

    return False


def tl_turboquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor | None,
    key_midpoints: torch.Tensor,
    key_centroids: torch.Tensor,
    value_midpoints: torch.Tensor,
    value_centroids: torch.Tensor,
    S_matrix: torch.Tensor,
    cfg: TileLangTQConfig,
) -> None:
    """Scaffold wrapper for the future fused TileLang store kernel."""

    del (
        key,
        value,
        kv_cache,
        slot_mapping,
        rotation,
        rotation_t,
        key_midpoints,
        key_centroids,
        value_midpoints,
        value_centroids,
        S_matrix,
        cfg,
    )
    raise_kernel_not_implemented()
