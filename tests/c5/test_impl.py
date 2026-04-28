import unittest
from dataclasses import replace

import torch
from torch import nn

from tilelang_turboquant.backend import TileLangTQAttentionImpl, TileLangTQMetadata
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


def _make_metadata(
    *,
    query_lens: list[int],
    seq_lens: list[int],
    num_decodes: int,
    num_decode_tokens: int,
    slot_mapping: torch.Tensor | None = None,
) -> TileLangTQMetadata:
    starts = [0]
    for q_len in query_lens:
        starts.append(starts[-1] + q_len)
    total_tokens = starts[-1]
    if slot_mapping is None:
        slot_mapping = torch.arange(total_tokens, dtype=torch.int32)
    block_table = torch.zeros((len(seq_lens), 1), dtype=torch.int32)
    return TileLangTQMetadata(
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        slot_mapping=slot_mapping,
        block_table=block_table,
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        query_start_loc_host=tuple(starts),
        num_actual_tokens=total_tokens,
        max_query_len=max(query_lens) if query_lens else 0,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        is_prefill=max(query_lens) > 1 if query_lens else False,
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )


class _TracingImpl(TileLangTQAttentionImpl):
    def __init__(self) -> None:
        super().__init__(
            num_heads=2,
            head_size=8,
            scale=SCALE,
            num_kv_heads=2,
            kv_cache_dtype="tilelang_tq_3bit",
        )
        self.last_path = None

    def _decode_attention(self, layer, query, kv_cache, attn_metadata):
        self.last_path = "decode"
        return torch.zeros_like(query)

    def _prefill_attention(self, layer, query, key, value, kv_cache, attn_metadata):
        if attn_metadata.max_seq_len == attn_metadata.max_query_len:
            self.last_path = "prefill_first_chunk"
        elif attn_metadata.max_query_len <= self._continuation_decode_threshold:
            self.last_path = "prefill_small_continuation"
        else:
            self.last_path = "prefill_large_continuation"
        return torch.zeros_like(query)

    def _mixed_attention(self, layer, query, key, value, kv_cache, attn_metadata):
        self.last_path = "mixed"
        return torch.zeros_like(query)


class TestTileLangTQAttentionImpl(unittest.TestCase):
    def setUp(self) -> None:
        register_all()
        torch.manual_seed(0)
        self.layer = _make_runtime_layer()
        self.impl = _make_impl()
        self.kv_cache = torch.zeros((2, 16, 2, 16), dtype=torch.int8)

    def test_do_kv_cache_update_uses_slot_mapping_height(self) -> None:
        key = torch.randn(4, 2, 8) * 0.1
        value = torch.randn(4, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, -1, 1, 2], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        self.assertTrue(torch.any(self.kv_cache[0, 0] != 0))
        self.assertTrue(torch.any(self.kv_cache[0, 1] != 0))
        self.assertTrue(torch.any(self.kv_cache[0, 2] != 0))

    def test_do_kv_cache_update_skips_negative_slots_in_kernel_contract(self) -> None:
        key = torch.randn(2, 2, 8) * 0.1
        value = torch.randn(2, 2, 8) * 0.1
        slot_mapping = torch.tensor([-1, 1], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        self.assertTrue(torch.all(self.kv_cache[0, 0] == 0))
        self.assertTrue(torch.any(self.kv_cache[0, 1] != 0))

    def test_forward_decode_path_selected(self) -> None:
        impl = _TracingImpl()
        metadata = _make_metadata(query_lens=[1], seq_lens=[3], num_decodes=1, num_decode_tokens=1)
        impl.forward(self.layer, torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), torch.zeros(1, 2, 8), self.kv_cache, metadata)
        self.assertEqual(impl.last_path, "decode")

    def test_forward_first_chunk_prefill_path_selected(self) -> None:
        impl = _TracingImpl()
        metadata = _make_metadata(query_lens=[2], seq_lens=[2], num_decodes=0, num_decode_tokens=0)
        impl.forward(self.layer, torch.zeros(2, 2, 8), torch.zeros(2, 2, 8), torch.zeros(2, 2, 8), self.kv_cache, metadata)
        self.assertEqual(impl.last_path, "prefill_first_chunk")

    def test_forward_continuation_prefill_small_query_uses_decode_style_path(self) -> None:
        impl = _TracingImpl()
        metadata = _make_metadata(query_lens=[2], seq_lens=[140], num_decodes=0, num_decode_tokens=0)
        impl.forward(self.layer, torch.zeros(2, 2, 8), torch.zeros(2, 2, 8), torch.zeros(2, 2, 8), self.kv_cache, metadata)
        self.assertEqual(impl.last_path, "prefill_small_continuation")

    def test_forward_continuation_prefill_large_query_uses_dequant_flash_path(self) -> None:
        impl = _TracingImpl()
        metadata = _make_metadata(query_lens=[129], seq_lens=[160], num_decodes=0, num_decode_tokens=0)
        impl.forward(self.layer, torch.zeros(129, 2, 8), torch.zeros(129, 2, 8), torch.zeros(129, 2, 8), self.kv_cache, metadata)
        self.assertEqual(impl.last_path, "prefill_large_continuation")

    def test_mixed_batch_prefill_max_seq_len_rebased(self) -> None:
        metadata = _make_metadata(query_lens=[1, 3], seq_lens=[20, 5], num_decodes=1, num_decode_tokens=1)
        prefill = self.impl._slice_prefill_metadata(metadata, 1, 1)
        self.assertEqual(prefill.max_seq_len, 5)
        self.assertEqual(prefill.max_query_len, 3)

    def test_forward_no_metadata_returns_zero_output(self) -> None:
        output = self.impl.forward(
            self.layer,
            torch.randn(2, 2, 8),
            torch.randn(2, 2, 8),
            torch.randn(2, 2, 8),
            self.kv_cache,
            None,
        )
        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_store_then_decode_matches_reference_attention_within_tolerance(self) -> None:
        key = torch.randn(3, 2, 8) * 0.1
        value = torch.randn(3, 2, 8) * 0.1
        query = torch.randn(1, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        metadata = _make_metadata(
            query_lens=[1],
            seq_lens=[3],
            num_decodes=1,
            num_decode_tokens=1,
            slot_mapping=slot_mapping[-1:],
        )
        output = self.impl.forward(self.layer, query, key[-1:], value[-1:], self.kv_cache, metadata)
        scores = torch.einsum("hd,shd->hs", query[0], key) * SCALE
        probs = torch.softmax(scores, dim=-1)
        reference = torch.einsum("hs,shd->hd", probs, value).unsqueeze(0)
        self.assertLess((output - reference).abs().max().item(), 0.05)

    def test_mixed_decode_prefill_batch_matches_split_reference(self) -> None:
        key = torch.randn(3, 2, 8) * 0.1
        value = torch.randn(3, 2, 8) * 0.1
        query = torch.randn(3, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1, 2], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        metadata = _make_metadata(
            query_lens=[1, 2],
            seq_lens=[1, 2],
            num_decodes=1,
            num_decode_tokens=1,
            slot_mapping=slot_mapping,
        )
        mixed = self.impl.forward(self.layer, query, key, value, self.kv_cache, metadata)
        decode_meta = self.impl._slice_decode_metadata(metadata)
        prefill_meta = self.impl._slice_prefill_metadata(metadata, 1, 1)
        split = torch.zeros_like(query)
        split[:1] = self.impl._decode_attention(self.layer, query[:1], self.kv_cache, decode_meta)
        split[1:] = self.impl._prefill_attention(self.layer, query[1:], key[1:], value[1:], self.kv_cache, prefill_meta)
        self.assertTrue(torch.allclose(mixed, split, atol=1e-6, rtol=1e-6))

    def test_no_runtime_gpu_materialization_in_store_or_forward(self) -> None:
        before = set(self.layer._buffers.keys())
        key = torch.randn(2, 2, 8) * 0.1
        value = torch.randn(2, 2, 8) * 0.1
        query = torch.randn(1, 2, 8) * 0.1
        slot_mapping = torch.tensor([0, 1], dtype=torch.int32)
        self.impl.do_kv_cache_update(self.layer, key, value, self.kv_cache, slot_mapping)
        metadata = _make_metadata(
            query_lens=[1],
            seq_lens=[2],
            num_decodes=1,
            num_decode_tokens=1,
            slot_mapping=slot_mapping[-1:],
        )
        self.impl.forward(self.layer, query, key[-1:], value[-1:], self.kv_cache, metadata)
        self.assertEqual(before, set(self.layer._buffers.keys()))


if __name__ == "__main__":
    unittest.main()
