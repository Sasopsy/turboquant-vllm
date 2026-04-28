"""TileLang TurboQuant standalone plugin package."""

from tilelang_turboquant.backend import (
    TileLangTQAttentionBackend,
    TileLangTQAttentionImpl,
    TileLangTQMetadata,
    TileLangTQMetadataBuilder,
)
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
from tilelang_turboquant.plugin import register_all
from tilelang_turboquant.quantization import (
    CACHE_DTYPE_ALIASES,
    VARIANT_BY_CACHE_DTYPE,
    TileLangTQ3BitConfig,
    TileLangTQ4BitConfig,
    TileLangTQKVCacheMethod,
    TileLangTQQuantizationConfig,
    normalize_cache_dtype,
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
    "CACHE_DTYPE_ALIASES",
    "TileLangTQAttentionBackend",
    "TileLangTQAttentionImpl",
    "TileLangTQMetadata",
    "TileLangTQMetadataBuilder",
    "get_codebook",
    "get_packed_kv_cache_shape",
    "get_rotation",
    "get_s_matrix",
    "get_variant",
    "get_variant_by_dtype_str",
    "normalize_cache_dtype",
    "register_all",
    "solve_lloyd_max_beta",
    "TileLangTQ3BitConfig",
    "TileLangTQ4BitConfig",
    "TileLangTQKVCacheMethod",
    "TileLangTQQuantizationConfig",
    "VARIANT_BY_CACHE_DTYPE",
]
