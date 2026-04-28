import unittest

import torch
from torch import nn

from tilelang_turboquant.plugin import register_all
from tilelang_turboquant.quantization import (
    TileLangTQ3BitConfig,
    TileLangTQ4BitConfig,
    normalize_cache_dtype,
)
from tilelang_turboquant.quantization.kv_cache_method import build_test_vllm_config

from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.quantization import get_quantization_config


def _make_attention_stub(
    *,
    kv_cache_dtype: str = "tilelang_tq_3bit",
    kv_cache_torch_dtype: torch.dtype = torch.int8,
    quant_config=None,
    head_size: int = 128,
    num_kv_heads: int = 8,
    num_heads: int = 8,
):
    layer = Attention.__new__(Attention)
    nn.Module.__init__(layer)
    layer.kv_cache_dtype = kv_cache_dtype
    layer.kv_cache_torch_dtype = kv_cache_torch_dtype
    layer.quant_config = quant_config
    layer.head_size = head_size
    layer.head_size_v = head_size
    layer.num_kv_heads = num_kv_heads
    layer.num_heads = num_heads
    return layer


class TestQuantConfig(unittest.TestCase):
    def setUp(self) -> None:
        register_all()

    def test_quant_config_name_3bit(self) -> None:
        self.assertEqual(TileLangTQ3BitConfig().get_name(), "tq_3bit")

    def test_quant_config_name_4bit(self) -> None:
        self.assertEqual(TileLangTQ4BitConfig().get_name(), "tq_4bit")

    def test_quant_method_only_for_attention(self) -> None:
        cfg = TileLangTQ3BitConfig()
        attention_layer = _make_attention_stub(quant_config=cfg)
        self.assertIsNotNone(cfg.get_quant_method(attention_layer, prefix="layers.0.attn"))
        self.assertIsNone(cfg.get_quant_method(nn.Linear(4, 4), prefix="linear"))

    def test_cache_dtype_normalization(self) -> None:
        self.assertEqual(normalize_cache_dtype("tq_3bit"), "tilelang_tq_3bit")
        self.assertEqual(normalize_cache_dtype("tilelang_tq_4bit"), "tilelang_tq_4bit")

    def test_variant_mismatch_rejected(self) -> None:
        cfg = TileLangTQ3BitConfig()
        attention_layer = _make_attention_stub(
            kv_cache_dtype="tilelang_tq_4bit",
            quant_config=cfg,
        )
        with self.assertRaises(ValueError):
            cfg.get_kv_cache_spec(attention_layer, build_test_vllm_config())

    def test_get_cache_scale_maps_k_proj(self) -> None:
        cfg = TileLangTQ3BitConfig()
        self.assertEqual(
            cfg.get_cache_scale("model.layers.0.k_proj.output_scale"),
            "model.layers.0.attn.k_scale",
        )

    def test_get_cache_scale_maps_v_proj(self) -> None:
        cfg = TileLangTQ3BitConfig()
        self.assertEqual(
            cfg.get_cache_scale("model.layers.0.v_proj.output_scale"),
            "model.layers.0.attn.v_scale",
        )

    def test_register_all_registers_quant_configs(self) -> None:
        register_all()
        self.assertIs(get_quantization_config("tq_3bit"), TileLangTQ3BitConfig)
        self.assertIs(get_quantization_config("tq_4bit"), TileLangTQ4BitConfig)

    def test_attention_get_kv_cache_spec_returns_plugin_spec(self) -> None:
        cfg = TileLangTQ3BitConfig()
        attention_layer = _make_attention_stub(quant_config=cfg)
        spec = Attention.get_kv_cache_spec(attention_layer, build_test_vllm_config())
        self.assertEqual(spec.tq_variant_name, "tq_3bit")
        self.assertEqual(spec.tq_slot_size, 112)


if __name__ == "__main__":
    unittest.main()

