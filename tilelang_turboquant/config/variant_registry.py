"""C1 variant registry and lookup helpers."""

from dataclasses import dataclass

PLUGIN_KV_CACHE_DTYPE_3BIT = "tilelang_tq_3bit"
PLUGIN_KV_CACHE_DTYPE_4BIT = "tilelang_tq_4bit"

_ALLOWED_DTYPE_STRS = {
    PLUGIN_KV_CACHE_DTYPE_3BIT,
    PLUGIN_KV_CACHE_DTYPE_4BIT,
}


@dataclass(frozen=True)
class VariantSpec:
    """Canonical variant definition for TileLang TurboQuant plugin modes."""

    name: str
    key_quant_bits: int
    value_quant_bits: int
    key_use_qjl: bool
    value_use_qjl: bool
    kv_cache_dtype_str: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("variant name must be non-empty")
        if self.key_quant_bits < 2:
            raise ValueError("key_quant_bits must be >= 2")
        if self.value_quant_bits < 2:
            raise ValueError("value_quant_bits must be >= 2")
        if self.key_use_qjl and self.key_quant_bits < 3:
            raise ValueError("key_quant_bits must be >= 3 when key_use_qjl=True")
        if self.value_use_qjl and self.value_quant_bits < 3:
            raise ValueError(
                "value_quant_bits must be >= 3 when value_use_qjl=True"
            )
        if self.kv_cache_dtype_str not in _ALLOWED_DTYPE_STRS:
            raise ValueError(
                f"Unsupported kv_cache_dtype_str={self.kv_cache_dtype_str!r}; "
                f"allowed={sorted(_ALLOWED_DTYPE_STRS)}"
            )


VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "tq_3bit": VariantSpec(
        name="tq_3bit",
        key_quant_bits=3,
        value_quant_bits=3,
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_3BIT,
    ),
    "tq_4bit": VariantSpec(
        name="tq_4bit",
        key_quant_bits=4,
        value_quant_bits=4,
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_4BIT,
    ),
}

_DTYPE_TO_VARIANT_NAME: dict[str, str] = {
    spec.kv_cache_dtype_str: spec.name for spec in VARIANT_REGISTRY.values()
}


def get_variant(name: str) -> VariantSpec:
    """Return a variant by canonical variant name."""

    try:
        return VARIANT_REGISTRY[name]
    except KeyError as exc:
        known = ", ".join(sorted(VARIANT_REGISTRY.keys()))
        raise KeyError(f"Unknown variant {name!r}. Known variants: {known}") from exc


def get_variant_by_dtype_str(dtype_str: str) -> VariantSpec:
    """Return a variant by canonical plugin kv-cache dtype string."""

    try:
        variant_name = _DTYPE_TO_VARIANT_NAME[dtype_str]
    except KeyError as exc:
        known = ", ".join(sorted(_DTYPE_TO_VARIANT_NAME.keys()))
        raise KeyError(
            f"Unknown kv_cache dtype string {dtype_str!r}. Known dtypes: {known}"
        ) from exc
    return get_variant(variant_name)

