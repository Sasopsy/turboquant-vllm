import unittest
from types import SimpleNamespace

import torch

from tilelang_turboquant.backend import TileLangTQMetadataBuilder
from tilelang_turboquant.plugin import register_all

from vllm.v1.attention.backends.utils import CommonAttentionMetadata


def _make_builder() -> TileLangTQMetadataBuilder:
    return TileLangTQMetadataBuilder(
        kv_cache_spec=None,
        layer_names=["layers.0.attn"],
        vllm_config=SimpleNamespace(
            speculative_config=None,
            parallel_config=SimpleNamespace(decode_context_parallel_size=1),
        ),
        device=torch.device("cpu"),
    )


def _make_common_metadata(
    query_lens: list[int],
    seq_lens: list[int],
) -> CommonAttentionMetadata:
    starts = [0]
    for q_len in query_lens:
        starts.append(starts[-1] + q_len)
    num_tokens = starts[-1]
    block_table_width = max(1, max((seq + 15) // 16 for seq in seq_lens))
    block_table = torch.zeros((len(seq_lens), block_table_width), dtype=torch.int32)
    for idx in range(len(seq_lens)):
        block_table[idx, 0] = idx
    return CommonAttentionMetadata(
        query_start_loc=torch.tensor(starts, dtype=torch.int32),
        query_start_loc_cpu=torch.tensor(starts, dtype=torch.int32),
        seq_lens=torch.tensor(seq_lens, dtype=torch.int32),
        num_reqs=len(query_lens),
        num_actual_tokens=num_tokens,
        max_query_len=max(query_lens) if query_lens else 0,
        max_seq_len=max(seq_lens) if seq_lens else 0,
        block_table_tensor=block_table,
        slot_mapping=torch.arange(num_tokens, dtype=torch.int32),
        causal=True,
    )


class TestTileLangTQMetadataBuilder(unittest.TestCase):
    def setUp(self) -> None:
        register_all()
        self.builder = _make_builder()

    def test_metadata_builder_reorders_decode_prefix(self) -> None:
        common = _make_common_metadata(query_lens=[1, 1, 3], seq_lens=[5, 8, 3])
        metadata = self.builder.build(common_prefix_len=0, common_attn_metadata=common)
        self.assertEqual(metadata.num_decodes, 2)
        self.assertEqual(metadata.num_decode_tokens, 2)
        self.assertTrue(metadata.is_prefill)
        self.assertEqual(metadata.query_start_loc_host, (0, 1, 2, 5))

    def test_metadata_builder_build_for_cudagraph_capture(self) -> None:
        common = _make_common_metadata(query_lens=[1, 1, 1], seq_lens=[5, 5, 5])
        metadata = self.builder.build_for_cudagraph_capture(common)
        self.assertTrue(torch.equal(metadata.seq_lens, torch.ones_like(metadata.seq_lens)))
        self.assertEqual(metadata.max_seq_len, 1)
        self.assertEqual(metadata.num_actual_tokens, 3)


if __name__ == "__main__":
    unittest.main()
