"""C3 plugin quantization config classes for TileLang TurboQuant."""

from __future__ import annotations

from typing import Any

import torch

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.memory import TileLangTQAttentionSpec
from tilelang_turboquant.quantization.compat import VARIANT_BY_CACHE_DTYPE, normalize_cache_dtype
from tilelang_turboquant.quantization.kv_cache_method import TileLangTQKVCacheMethod

from vllm.model_executor.layers.quantization import register_quantization_config
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig, QuantizeMethodBase


class TileLangTQQuantizationConfig(QuantizationConfig):
    """Shared base config for plugin-owned TQ KV-cache quantization.

    This class is the `vLLM`-facing entry point for `quantization="tq_*"`.
    It does not perform the compression itself; instead it tells vLLM:
    1. which layers should get a TQ KV-cache method
    2. how checkpoint K/V scale names map into the attention layer
    3. how to build the plugin-local KV cache spec for allocation
    """

    variant_name: str = ""

    @classmethod
    def get_name(cls) -> str:
        return cls.variant_name

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TileLangTQQuantizationConfig":
        # The plugin variant is chosen by the registry key (`tq_3bit` /
        # `tq_4bit`), so there is no sidecar quant config file to parse here.
        return cls()

    def get_quant_method(
        self,
        layer: torch.nn.Module,
        prefix: str,
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.attention.attention import Attention

        if isinstance(layer, Attention):
            # Only decoder self-attention layers own KV cache in the path we
            # support today, so those are the only layers that get the plugin
            # KV-cache lifecycle method.
            return TileLangTQKVCacheMethod(self)
        return None

    def get_cache_scale(self, name: str) -> str | None:
        # vLLM already has a scalar-scale loading path. We reuse that narrow
        # hook only for K/V checkpoint scales; all non-scalar TQ artifacts are
        # handled later by `TileLangTQKVCacheMethod.process_weights_after_loading`.
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        return None

    def get_kv_cache_spec(
        self,
        layer: torch.nn.Module,
        vllm_config,
    ):
        from vllm.model_executor.layers.attention.attention import Attention

        if not isinstance(layer, Attention):
            return None

        # Quantization name (`tq_3bit`) and cache dtype
        # (`tilelang_tq_3bit`) are intentionally separate identifiers. This
        # check prevents a mismatched config/dtype pair from silently building
        # the wrong slot layout.
        normalized = normalize_cache_dtype(layer.kv_cache_dtype)
        expected_variant = VARIANT_BY_CACHE_DTYPE[normalized]
        if expected_variant != self.variant_name:
            raise ValueError(
                f"Quantization variant {self.variant_name!r} is incompatible with "
                f"kv_cache_dtype={layer.kv_cache_dtype!r}."
            )

        cfg = TileLangTQConfig.from_variant_name(self.variant_name, layer.head_size)
        # This is where C1's slot arithmetic becomes visible to vLLM's KV cache
        # allocator: we package the aligned slot size into a plugin-local spec.
        return TileLangTQAttentionSpec(
            block_size=vllm_config.cache_config.block_size,
            num_kv_heads=layer.num_kv_heads,
            head_size=layer.head_size,
            head_size_v=layer.head_size,
            dtype=layer.kv_cache_torch_dtype,
            tq_slot_size=cfg.slot_size_aligned,
            tq_variant_name=self.variant_name,
        )


@register_quantization_config("tq_3bit")
class TileLangTQ3BitConfig(TileLangTQQuantizationConfig):
    """Concrete registry entry for `quantization="tq_3bit"`."""

    variant_name = "tq_3bit"


@register_quantization_config("tq_4bit")
class TileLangTQ4BitConfig(TileLangTQQuantizationConfig):
    """Concrete registry entry for `quantization="tq_4bit"`."""

    variant_name = "tq_4bit"
