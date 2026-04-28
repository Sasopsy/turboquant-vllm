"""C4 attention backend registration surface for TileLang TQ."""

from __future__ import annotations

import os

import torch

from tilelang_turboquant.backend.impl import TileLangTQAttentionImpl
from tilelang_turboquant.backend.metadata import TileLangTQMetadataBuilder
from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.config.variant_registry import get_variant_by_dtype_str
from tilelang_turboquant.memory import get_packed_kv_cache_shape
from tilelang_turboquant.quantization.compat import normalize_cache_dtype

from vllm.v1.attention.backend import AttentionBackend, AttentionType


_MIN_COMPUTE_CAPABILITY = 80
_SUPPORTED_KERNEL_BLOCK_SIZES = [16, 32, 64, 128]


def _rotation_mode_from_runtime_config() -> str:
    return os.environ.get("TILELANG_TQ_ROTATION_MODE", "qr")


def _allow_hadamard_padding() -> bool:
    return os.environ.get("TILELANG_TQ_ALLOW_HADAMARD_PADDING", "0") == "1"


def _is_power_of_two(value: int) -> bool:
    return value > 0 and (value & (value - 1) == 0)


class TileLangTQAttentionBackend(AttentionBackend):
    """Plugin-owned split-KV-update attention backend for packed TQ cache."""

    # Keep the plugin's canonical name visible for diagnostics even though the
    # current vLLM branch routes the class through the CUSTOM enum entry.
    CANONICAL_NAME = "TILELANG_TQ"
    accept_output_buffer = True
    forward_includes_kv_cache_update = False
    supported_dtypes = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes = ["tilelang_tq_3bit", "tilelang_tq_4bit"]

    @staticmethod
    def get_name() -> str:
        return TileLangTQAttentionBackend.CANONICAL_NAME

    @staticmethod
    def get_impl_cls() -> type[TileLangTQAttentionImpl]:
        return TileLangTQAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TileLangTQMetadataBuilder]:
        return TileLangTQMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int]:
        return list(_SUPPORTED_KERNEL_BLOCK_SIZES)

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "tilelang_tq_3bit",
    ) -> tuple[int, int, int, int]:
        normalized = normalize_cache_dtype(cache_dtype_str)
        return get_packed_kv_cache_shape(
            num_blocks,
            block_size,
            num_kv_heads,
            head_size,
            normalized,
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
        if kv_cache_dtype is None:
            return False
        try:
            normalized = normalize_cache_dtype(kv_cache_dtype)
        except ValueError:
            return False
        return normalized in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability) -> bool:
        if capability is None:
            return True
        major = getattr(capability, "major", None)
        minor = getattr(capability, "minor", None)
        if major is None and isinstance(capability, tuple) and capability:
            major = capability[0]
            minor = capability[1] if len(capability) > 1 else 0
        if major is None:
            return True
        return (int(major) * 10 + int(minor or 0)) >= _MIN_COMPUTE_CAPABILITY

    @classmethod
    def validate_configuration(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: str | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        use_per_head_quant_scales: bool,
        device_capability,
        attn_type: str,
    ) -> list[str]:
        invalid_reasons = super().validate_configuration(
            head_size,
            dtype,
            kv_cache_dtype,
            block_size,
            use_mla,
            has_sink,
            use_sparse,
            use_mm_prefix,
            use_per_head_quant_scales,
            device_capability,
            attn_type,
        )

        if kv_cache_dtype is not None:
            try:
                normalized = normalize_cache_dtype(kv_cache_dtype)
                get_variant_by_dtype_str(normalized)
            except (ValueError, KeyError):
                invalid_reasons.append("kv_cache_dtype does not resolve to a plugin variant")
            else:
                try:
                    TileLangTQConfig.from_variant_name(
                        get_variant_by_dtype_str(normalized).name,
                        head_size,
                    )
                except ValueError as exc:
                    invalid_reasons.append(str(exc))

        rotation_mode = _rotation_mode_from_runtime_config()
        if rotation_mode == "randomized_hadamard":
            if not _is_power_of_two(head_size) and not _allow_hadamard_padding():
                invalid_reasons.append(
                    "randomized_hadamard requires power-of-two head_size unless padding support is enabled"
                )
        elif rotation_mode != "qr":
            invalid_reasons.append(f"rotation mode {rotation_mode!r} not supported")

        return invalid_reasons
