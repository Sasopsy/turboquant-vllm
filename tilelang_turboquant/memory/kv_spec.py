"""C1 plugin-local KV spec and packed cache shape helper."""

from dataclasses import dataclass, replace
from typing import Self

from vllm.v1.kv_cache_interface import FullAttentionSpec

from tilelang_turboquant.config.tq_config import TileLangTQConfig
from tilelang_turboquant.config.variant_registry import get_variant_by_dtype_str


@dataclass(frozen=True, kw_only=True)
class TileLangTQAttentionSpec(FullAttentionSpec):
    """Plugin-local full-attention KV spec for packed TileLang TQ cache."""

    tq_slot_size: int = 0
    tq_variant_name: str = ""

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list[Self]) -> Self:
        if not specs:
            raise ValueError("Cannot merge an empty spec list")
        first = specs[0]
        if not all(spec.tq_slot_size == first.tq_slot_size for spec in specs):
            raise ValueError(
                "All TQ layers in the same KV cache group must use the same tq_slot_size"
            )
        if not all(spec.tq_variant_name == first.tq_variant_name for spec in specs):
            raise ValueError(
                "All TQ layers in the same KV cache group must use the same tq_variant_name"
            )
        merged = super().merge(specs)
        # Replace the merged spec with the first spec's tq_slot_size and tq_variant_name.
        return replace(
            merged,
            tq_slot_size=first.tq_slot_size,
            tq_variant_name=first.tq_variant_name,
        )


def get_packed_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "tilelang_tq_3bit",
) -> tuple[int, int, int, int]:
    """Return packed cache shape: (num_blocks, block_size, heads, slot_bytes)."""

    variant = get_variant_by_dtype_str(cache_dtype_str)
    cfg = TileLangTQConfig.from_variant_name(variant.name, head_size)
    return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)

