"""TileLang TurboQuant standalone plugin package."""

from tilelang_turboquant.config import (
    SlotLayout,
    TileLangTQConfig,
    VARIANT_REGISTRY,
    VariantSpec,
    get_variant,
    get_variant_by_dtype_str,
)
from tilelang_turboquant.memory import (
    TileLangTQAttentionSpec,
    get_packed_kv_cache_shape,
)
from tilelang_turboquant.offline import (
    CodebookSpec,
    QJLSpec,
    RotationSpec,
    get_codebook,
    get_rotation,
    get_s_matrix,
    solve_lloyd_max_beta,
)

__all__ = [
    "SlotLayout",
    "TileLangTQAttentionSpec",
    "TileLangTQConfig",
    "VARIANT_REGISTRY",
    "VariantSpec",
    "CodebookSpec",
    "QJLSpec",
    "RotationSpec",
    "get_codebook",
    "get_packed_kv_cache_shape",
    "get_rotation",
    "get_s_matrix",
    "get_variant",
    "get_variant_by_dtype_str",
    "solve_lloyd_max_beta",
]
