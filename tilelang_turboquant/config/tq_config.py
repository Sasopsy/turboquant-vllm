"""C1 TileLang TQ slot sizing and packed layout config."""

import math
from dataclasses import dataclass

from tilelang_turboquant.config.variant_registry import VariantSpec, get_variant


@dataclass(frozen=True)
class TileLangTQConfig:
    """Derived sizing/bit-budget config from variant + head dimension."""

    variant_name: str
    head_dim: int

    @classmethod
    def from_variant_name(cls, variant_name: str, head_dim: int) -> "TileLangTQConfig":
        if head_dim <= 0:
            raise ValueError("head_dim must be > 0")
        if head_dim % 8 != 0:
            raise ValueError("head_dim must be divisible by 8")
        get_variant(variant_name)
        return cls(variant_name=variant_name, head_dim=head_dim)

    @property
    def variant(self) -> VariantSpec:
        return get_variant(self.variant_name)

    @property
    def key_mse_bits(self) -> int:
        if self.variant.key_use_qjl:
            return self.variant.key_quant_bits - 1
        return self.variant.key_quant_bits

    @property
    def value_mse_bits(self) -> int:
        if self.variant.value_use_qjl:
            return self.variant.value_quant_bits - 1
        return self.variant.value_quant_bits

    @property
    def key_n_centroids(self) -> int:
        return 2**self.key_mse_bits

    @property
    def value_n_centroids(self) -> int:
        return 2**self.value_mse_bits

    @property
    def key_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.key_mse_bits / 8)

    @property
    def key_norm_bytes(self) -> int:
        return 2

    @property
    def key_qjl_bits_bytes(self) -> int:
        if self.variant.key_use_qjl:
            return math.ceil(self.head_dim / 8)
        return 0

    @property
    def key_qjl_gamma_bytes(self) -> int:
        if self.variant.key_use_qjl:
            return 2
        return 0

    @property
    def value_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_mse_bits / 8)

    @property
    def value_norm_bytes(self) -> int:
        return 2

    @property
    def value_qjl_bits_bytes(self) -> int:
        if self.variant.value_use_qjl:
            return math.ceil(self.head_dim / 8)
        return 0

    @property
    def value_qjl_gamma_bytes(self) -> int:
        if self.variant.value_use_qjl:
            return 2
        return 0

    @property
    def key_side_bytes(self) -> int:
        return (
            self.key_mse_bytes
            + self.key_norm_bytes
            + self.key_qjl_bits_bytes
            + self.key_qjl_gamma_bytes
        )

    @property
    def value_side_bytes(self) -> int:
        return (
            self.value_mse_bytes
            + self.value_norm_bytes
            + self.value_qjl_bits_bytes
            + self.value_qjl_gamma_bytes
        )

    @property
    def slot_size_raw(self) -> int:
        return self.key_side_bytes + self.value_side_bytes

    @property
    def slot_size_aligned(self) -> int:
        return ((self.slot_size_raw + 15) // 16) * 16

    @property
    def padding_bytes(self) -> int:
        return self.slot_size_aligned - self.slot_size_raw


@dataclass(frozen=True)
class SlotLayout:
    """Byte-level packed layout for one K+V slot."""

    key_mse_offset: int
    key_mse_size: int
    key_norm_offset: int
    key_norm_size: int
    key_qjl_bits_offset: int
    key_qjl_bits_size: int
    key_qjl_gamma_offset: int
    key_qjl_gamma_size: int
    value_mse_offset: int
    value_mse_size: int
    value_norm_offset: int
    value_norm_size: int
    value_qjl_bits_offset: int
    value_qjl_bits_size: int
    value_qjl_gamma_offset: int
    value_qjl_gamma_size: int
    slot_size_aligned: int

    @classmethod
    def from_config(cls, cfg: TileLangTQConfig) -> "SlotLayout":
        key_mse_offset = 0
        key_norm_offset = key_mse_offset + cfg.key_mse_bytes
        key_qjl_bits_offset = key_norm_offset + cfg.key_norm_bytes
        key_qjl_gamma_offset = key_qjl_bits_offset + cfg.key_qjl_bits_bytes

        value_mse_offset = key_qjl_gamma_offset + cfg.key_qjl_gamma_bytes
        value_norm_offset = value_mse_offset + cfg.value_mse_bytes
        value_qjl_bits_offset = value_norm_offset + cfg.value_norm_bytes
        value_qjl_gamma_offset = value_qjl_bits_offset + cfg.value_qjl_bits_bytes

        return cls(
            key_mse_offset=key_mse_offset,
            key_mse_size=cfg.key_mse_bytes,
            key_norm_offset=key_norm_offset,
            key_norm_size=cfg.key_norm_bytes,
            key_qjl_bits_offset=key_qjl_bits_offset,
            key_qjl_bits_size=cfg.key_qjl_bits_bytes,
            key_qjl_gamma_offset=key_qjl_gamma_offset,
            key_qjl_gamma_size=cfg.key_qjl_gamma_bytes,
            value_mse_offset=value_mse_offset,
            value_mse_size=cfg.value_mse_bytes,
            value_norm_offset=value_norm_offset,
            value_norm_size=cfg.value_norm_bytes,
            value_qjl_bits_offset=value_qjl_bits_offset,
            value_qjl_bits_size=cfg.value_qjl_bits_bytes,
            value_qjl_gamma_offset=value_qjl_gamma_offset,
            value_qjl_gamma_size=cfg.value_qjl_gamma_bytes,
            slot_size_aligned=cfg.slot_size_aligned,
        )

    @property
    def raw_end(self) -> int:
        return self.value_qjl_gamma_offset + self.value_qjl_gamma_size

