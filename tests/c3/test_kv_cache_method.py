import unittest

import torch
from torch import nn

from tilelang_turboquant.plugin import register_all
from tilelang_turboquant.quantization import TileLangTQ3BitConfig

from vllm.model_executor.layers.attention.attention import set_default_quant_scales


def _make_runtime_layer(
    *,
    kv_cache_dtype: str = "tilelang_tq_3bit",
    head_size: int = 128,
    num_kv_heads: int = 8,
    num_heads: int = 8,
):
    layer = nn.Module()
    layer.kv_cache_dtype = kv_cache_dtype
    layer.head_size = head_size
    layer.head_size_v = head_size
    layer.num_kv_heads = num_kv_heads
    layer.num_heads = num_heads
    set_default_quant_scales(layer, register_buffer=True)
    return layer


class TestKVCacheMethod(unittest.TestCase):
    def setUp(self) -> None:
        register_all()
        self.quant_config = TileLangTQ3BitConfig()
        from tilelang_turboquant.quantization.kv_cache_method import TileLangTQKVCacheMethod

        self.method = TileLangTQKVCacheMethod(self.quant_config)

    def test_process_weights_defaults_scales(self) -> None:
        layer = _make_runtime_layer()
        self.method.create_weights(layer)
        self.method.process_weights_after_loading(layer)
        self.assertEqual(float(layer._k_scale.item()), 1.0)
        self.assertEqual(float(layer._v_scale.item()), 1.0)
        self.assertEqual(layer._k_scale_float, 1.0)
        self.assertEqual(layer._v_scale_float, 1.0)

    def test_process_weights_registers_separate_key_value_codebooks(self) -> None:
        layer = _make_runtime_layer()
        self.method.create_weights(layer)
        self.method.process_weights_after_loading(layer)
        self.assertIn("_tq_key_centroids", layer._buffers)
        self.assertIn("_tq_key_midpoints", layer._buffers)
        self.assertIn("_tq_value_centroids", layer._buffers)
        self.assertIn("_tq_value_midpoints", layer._buffers)

    def test_process_weights_registers_rotation_and_s(self) -> None:
        layer = _make_runtime_layer()
        self.method.create_weights(layer)
        self.method.process_weights_after_loading(layer)
        self.assertIn("_tq_rotation", layer._buffers)
        self.assertIn("_tq_rotation_t", layer._buffers)
        self.assertIn("_tq_S_matrix", layer._buffers)
        self.assertEqual(layer._tq_rotation.dtype, torch.float32)
        self.assertEqual(layer._tq_S_matrix.dtype, torch.float32)

    def test_process_weights_deletes_temp_params(self) -> None:
        layer = _make_runtime_layer()
        self.method.create_weights(layer)
        self.method.process_weights_after_loading(layer)
        self.assertFalse(hasattr(layer, "k_scale"))
        self.assertFalse(hasattr(layer, "v_scale"))
        self.assertFalse(hasattr(layer, "q_scale"))
        self.assertFalse(hasattr(layer, "prob_scale"))

    def test_scratch_buffers_nonpersistent(self) -> None:
        layer = _make_runtime_layer()
        self.method.create_weights(layer)
        self.method.process_weights_after_loading(layer)
        state_dict_keys = set(layer.state_dict().keys())
        self.assertNotIn("_tq_mid_o_buf", state_dict_keys)
        self.assertNotIn("_tq_lse_buf", state_dict_keys)
        self.assertNotIn("_tq_output_buf", state_dict_keys)
        self.assertIn("_tq_mid_o_buf", layer._non_persistent_buffers_set)
        self.assertIn("_tq_lse_buf", layer._non_persistent_buffers_set)
        self.assertIn("_tq_output_buf", layer._non_persistent_buffers_set)


if __name__ == "__main__":
    unittest.main()
