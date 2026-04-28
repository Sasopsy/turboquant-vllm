import inspect
import unittest
from unittest import mock

import torch
from torch import nn

import tilelang_turboquant.kernels as tq_kernels
from tilelang_turboquant.backend import TileLangTQAttentionImpl, TileLangTQMetadata
from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.kernels import TileLangKernelNotImplementedError
from tilelang_turboquant.plugin import register_all
from tilelang_turboquant.quantization import TileLangTQ3BitConfig
from tilelang_turboquant.quantization.kv_cache_method import TileLangTQKVCacheMethod

from vllm.model_executor.layers.attention.attention import set_default_quant_scales


SCALE = 1 / (8 ** 0.5)


def _make_runtime_layer(
    *,
    kv_cache_dtype: str = "tilelang_tq_3bit",
    head_size: int = 8,
    num_kv_heads: int = 2,
    num_heads: int = 2,
):
    layer = nn.Module()
    layer.kv_cache_dtype = kv_cache_dtype
    layer.head_size = head_size
    layer.head_size_v = head_size
    layer.num_kv_heads = num_kv_heads
    layer.num_heads = num_heads
    set_default_quant_scales(layer, register_buffer=True)
    method = TileLangTQKVCacheMethod(TileLangTQ3BitConfig())
    method.create_weights(layer)
    method.process_weights_after_loading(layer)
    return layer


def _make_impl() -> TileLangTQAttentionImpl:
    return TileLangTQAttentionImpl(
        num_heads=2,
        head_size=8,
        scale=SCALE,
        num_kv_heads=2,
        kv_cache_dtype="tilelang_tq_3bit",
    )


def _make_decode_metadata(seq_len: int = 3) -> TileLangTQMetadata:
    return TileLangTQMetadata(
        seq_lens=torch.tensor([seq_len], dtype=torch.int32),
        slot_mapping=torch.tensor([seq_len - 1], dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        query_start_loc=torch.tensor([0, 1], dtype=torch.int32),
        query_start_loc_host=(0, 1),
        num_actual_tokens=1,
        max_query_len=1,
        max_seq_len=seq_len,
        is_prefill=False,
        num_decodes=1,
        num_decode_tokens=1,
    )


class TestKernelScaffolding(unittest.TestCase):
    def setUp(self) -> None:
        register_all()
        torch.manual_seed(0)
        self.layer = _make_runtime_layer()
        self.impl = _make_impl()
        self.kv_cache = torch.zeros((2, 16, 2, 16), dtype=torch.int8)

    def test_store_wrapper_signature(self) -> None:
        params = list(inspect.signature(tq_kernels.tl_turboquant_store).parameters)
        self.assertEqual(
            params,
            [
                "key",
                "value",
                "kv_cache",
                "slot_mapping",
                "rotation",
                "rotation_t",
                "key_midpoints",
                "key_centroids",
                "value_midpoints",
                "value_centroids",
                "S_matrix",
                "cfg",
            ],
        )

    def test_decode_wrapper_signature(self) -> None:
        params = list(inspect.signature(tq_kernels.tl_turboquant_decode_attention).parameters)
        self.assertEqual(
            params,
            [
                "query",
                "kv_cache",
                "block_table",
                "seq_lens",
                "rotation",
                "rotation_t",
                "key_centroids",
                "value_centroids",
                "S_matrix",
                "scale",
                "cfg",
                "mid_o_buf",
                "lse_buf",
                "output_buf",
                "max_num_kv_splits",
            ],
        )

    def test_direct_wrapper_calls_raise_not_implemented(self) -> None:
        cfg = TileLangTQConfig.from_variant_name("tq_3bit", 8)
        with self.assertRaises(TileLangKernelNotImplementedError):
            tq_kernels.tl_turboquant_store(
                key=torch.zeros(1, 2, 8),
                value=torch.zeros(1, 2, 8),
                kv_cache=self.kv_cache,
                slot_mapping=torch.zeros(1, dtype=torch.int32),
                rotation=self.layer._tq_rotation,
                rotation_t=self.layer._tq_rotation_t,
                key_midpoints=self.layer._tq_key_midpoints,
                key_centroids=self.layer._tq_key_centroids,
                value_midpoints=self.layer._tq_value_midpoints,
                value_centroids=self.layer._tq_value_centroids,
                S_matrix=self.layer._tq_S_matrix,
                cfg=cfg,
            )
        with self.assertRaises(TileLangKernelNotImplementedError):
            tq_kernels.tl_turboquant_decode_attention(
                query=torch.zeros(1, 2, 8),
                kv_cache=self.kv_cache,
                block_table=torch.zeros((1, 1), dtype=torch.int32),
                seq_lens=torch.tensor([1], dtype=torch.int32),
                rotation=self.layer._tq_rotation,
                rotation_t=self.layer._tq_rotation_t,
                key_centroids=self.layer._tq_key_centroids,
                value_centroids=self.layer._tq_value_centroids,
                S_matrix=self.layer._tq_S_matrix,
                scale=SCALE,
                cfg=cfg,
                mid_o_buf=self.layer._tq_mid_o_buf,
                lse_buf=self.layer._tq_lse_buf,
                output_buf=self.layer._tq_output_buf,
                max_num_kv_splits=1,
            )

    def test_scaffold_availability_helpers_default_false(self) -> None:
        self.assertFalse(tq_kernels.is_store_kernel_available())
        self.assertFalse(tq_kernels.is_decode_kernel_available())

    def test_store_falls_back_to_reference_when_scaffold_unavailable(self) -> None:
        key = torch.randn(2, 2, 8) * 0.1
        value = torch.randn(2, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
        with mock.patch.object(self.impl, "_encode_slot", wraps=self.impl._encode_slot) as encode_slot:
            self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        self.assertGreater(encode_slot.call_count, 0)
        self.assertTrue(torch.any(self.kv_cache[0, 0] != 0))

    def test_decode_falls_back_to_reference_when_scaffold_unavailable(self) -> None:
        key = torch.randn(3, 2, 8) * 0.1
        value = torch.randn(3, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        metadata = _make_decode_metadata(seq_len=3)
        query = torch.randn(1, 2, 8) * 0.1
        with mock.patch.object(self.impl, "_load_cached_prefix", wraps=self.impl._load_cached_prefix) as load_prefix:
            output = self.impl._decode_attention(self.layer, query, self.kv_cache, metadata)
        self.assertGreater(load_prefix.call_count, 0)
        self.assertEqual(output.shape, query.shape)

    def test_available_store_wrapper_is_invoked_through_dispatch_seam(self) -> None:
        key = torch.randn(2, 2, 8) * 0.1
        value = torch.randn(2, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
        calls = {}

        def _fake_store(**kwargs):
            calls.update(kwargs)

        with mock.patch.object(tq_kernels, "is_store_kernel_available", return_value=True), \
             mock.patch.object(tq_kernels, "tl_turboquant_store", side_effect=_fake_store), \
             mock.patch.object(self.impl, "_encode_slot", side_effect=AssertionError("reference store should not run")):
            self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)

        self.assertIs(calls["cfg"], self.impl.tq_config)
        self.assertIs(calls["rotation"], self.layer._tq_rotation)
        self.assertIs(calls["S_matrix"], self.layer._tq_S_matrix)
        self.assertTrue(torch.equal(calls["slot_mapping"], slot_mapping))

    def test_available_decode_wrapper_is_invoked_through_dispatch_seam(self) -> None:
        metadata = _make_decode_metadata(seq_len=3)
        query = torch.randn(1, 2, 8) * 0.1
        calls = {}

        def _fake_decode(**kwargs):
            calls.update(kwargs)
            return torch.full_like(kwargs["query"], 7.0)

        with mock.patch.object(tq_kernels, "is_decode_kernel_available", return_value=True), \
             mock.patch.object(tq_kernels, "tl_turboquant_decode_attention", side_effect=_fake_decode), \
             mock.patch.object(self.impl, "_load_cached_prefix", side_effect=AssertionError("reference decode should not run")):
            output = self.impl._decode_attention(self.layer, query, self.kv_cache, metadata)

        self.assertTrue(torch.equal(output, torch.full_like(query, 7.0)))
        self.assertIs(calls["cfg"], self.impl.tq_config)
        self.assertIs(calls["rotation"], self.layer._tq_rotation)
        self.assertEqual(tuple(calls["block_table"].shape), (1, 1))
        self.assertEqual(int(calls["seq_lens"][0].item()), 3)

    def test_dispatch_seam_does_not_create_new_long_lived_buffers(self) -> None:
        before = set(self.layer._buffers.keys())
        key = torch.randn(1, 2, 8) * 0.1
        value = torch.randn(1, 2, 8) * 0.1
        slot_mapping = torch.tensor([0], dtype=torch.int32)
        metadata = _make_decode_metadata(seq_len=1)
        query = torch.randn(1, 2, 8) * 0.1

        with mock.patch.object(tq_kernels, "is_store_kernel_available", return_value=True), \
             mock.patch.object(tq_kernels, "tl_turboquant_store", return_value=None), \
             mock.patch.object(tq_kernels, "is_decode_kernel_available", return_value=True), \
             mock.patch.object(tq_kernels, "tl_turboquant_decode_attention", return_value=torch.zeros_like(query)):
            self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
            self.impl._decode_attention(self.layer, query, self.kv_cache, metadata)

        self.assertEqual(before, set(self.layer._buffers.keys()))


if __name__ == "__main__":
    unittest.main()
