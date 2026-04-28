"""C3 quantization configuration and cache-method interfaces."""

from tilelang_turboquant.quantization.compat import (
    CACHE_DTYPE_ALIASES,
    VARIANT_BY_CACHE_DTYPE,
    install_cache_dtype_admission_shims,
    install_kv_spec_dispatch_shim,
    normalize_cache_dtype,
)
from tilelang_turboquant.quantization.kv_cache_method import TileLangTQKVCacheMethod
from tilelang_turboquant.quantization.quant_config import (
    TileLangTQ3BitConfig,
    TileLangTQ4BitConfig,
    TileLangTQQuantizationConfig,
)

__all__ = [
    "CACHE_DTYPE_ALIASES",
    "VARIANT_BY_CACHE_DTYPE",
    "TileLangTQ3BitConfig",
    "TileLangTQ4BitConfig",
    "TileLangTQKVCacheMethod",
    "TileLangTQQuantizationConfig",
    "install_cache_dtype_admission_shims",
    "install_kv_spec_dispatch_shim",
    "normalize_cache_dtype",
]

