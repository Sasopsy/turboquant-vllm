"""C1 config interfaces: variant registry + slot/layout config."""

from tilelang_turboquant.config.tq_config import SlotLayout, TileLangTQConfig
from tilelang_turboquant.config.variant_registry import (
    VARIANT_REGISTRY,
    VariantSpec,
    get_variant,
    get_variant_by_dtype_str,
)

__all__ = [
    "SlotLayout",
    "TileLangTQConfig",
    "VARIANT_REGISTRY",
    "VariantSpec",
    "get_variant",
    "get_variant_by_dtype_str",
]

