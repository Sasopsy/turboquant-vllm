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
from tilelang_turboquant.kernels import (
    TileLangKernelNotImplementedError,
    is_decode_kernel_available,
    is_store_kernel_available,
    tl_turboquant_decode_attention,
    tl_turboquant_store,
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
    "TileLangKernelNotImplementedError",
    "get_codebook",
    "get_packed_kv_cache_shape",
    "get_rotation",
    "get_s_matrix",
    "get_variant",
    "get_variant_by_dtype_str",
    "is_decode_kernel_available",
    "is_store_kernel_available",
    "normalize_cache_dtype",
    "register_all",
    "solve_lloyd_max_beta",
    "tl_turboquant_decode_attention",
    "tl_turboquant_store",
    "TileLangTQ3BitConfig",
    "TileLangTQ4BitConfig",
    "TileLangTQKVCacheMethod",
    "TileLangTQQuantizationConfig",
    "VARIANT_BY_CACHE_DTYPE",
]
