import math
import unittest

import torch

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.memory import (
    TileLangTQAttentionSpec,
    get_packed_kv_cache_shape,
)


class TestTileLangTQAttentionSpec(unittest.TestCase):
    def test_real_page_size_formula(self) -> None:
        cfg = TileLangTQConfig.from_variant_name("tq_3bit", 128)
        spec = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=cfg.slot_size_aligned,
            tq_variant_name="tq_3bit",
        )
        expected = 16 * 8 * cfg.slot_size_aligned
        self.assertEqual(spec.real_page_size_bytes, expected)
        self.assertEqual(spec.page_size_bytes, expected)

    def test_merge_success(self) -> None:
        spec_a = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=112,
            tq_variant_name="tq_3bit",
        )
        spec_b = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=112,
            tq_variant_name="tq_3bit",
        )
        merged = TileLangTQAttentionSpec.merge([spec_a, spec_b])
        self.assertEqual(merged.tq_slot_size, 112)
        self.assertEqual(merged.tq_variant_name, "tq_3bit")
        self.assertEqual(merged.real_page_size_bytes, spec_a.real_page_size_bytes)

    def test_merge_rejects_slot_size_mismatch(self) -> None:
        spec_a = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=112,
            tq_variant_name="tq_3bit",
        )
        spec_b = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=144,
            tq_variant_name="tq_3bit",
        )
        with self.assertRaises(ValueError):
            TileLangTQAttentionSpec.merge([spec_a, spec_b])

    def test_merge_rejects_variant_mismatch(self) -> None:
        spec_a = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=112,
            tq_variant_name="tq_3bit",
        )
        spec_b = TileLangTQAttentionSpec(
            block_size=16,
            num_kv_heads=8,
            head_size=128,
            head_size_v=128,
            dtype=torch.int8,
            tq_slot_size=112,
            tq_variant_name="tq_4bit",
        )
        with self.assertRaises(ValueError):
            TileLangTQAttentionSpec.merge([spec_a, spec_b])

    def test_shape_product_matches_page_bytes(self) -> None:
        num_blocks = 7
        block_size = 16
        num_kv_heads = 8
        head_size = 128
        shape = get_packed_kv_cache_shape(
            num_blocks=num_blocks,
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            cache_dtype_str="tilelang_tq_3bit",
        )

        cfg = TileLangTQConfig.from_variant_name("tq_3bit", head_size)
        spec = TileLangTQAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_size,
            head_size_v=head_size,
            dtype=torch.int8,
            tq_slot_size=cfg.slot_size_aligned,
            tq_variant_name="tq_3bit",
        )

        element_size = torch.empty((), dtype=spec.dtype).element_size()
        expected_numel = num_blocks * spec.real_page_size_bytes // element_size
        self.assertEqual(math.prod(shape), expected_numel)

    def test_d128_page_sizes(self) -> None:
        cfg_3 = TileLangTQConfig.from_variant_name("tq_3bit", 128)
        cfg_4 = TileLangTQConfig.from_variant_name("tq_4bit", 128)
        self.assertEqual(16 * 8 * cfg_3.slot_size_aligned, 14336)
        self.assertEqual(16 * 8 * cfg_4.slot_size_aligned, 18432)

    def test_kv_cache_shape_tuple(self) -> None:
        shape_3 = get_packed_kv_cache_shape(10, 16, 8, 128, "tilelang_tq_3bit")
        shape_4 = get_packed_kv_cache_shape(10, 16, 8, 128, "tilelang_tq_4bit")
        self.assertEqual(shape_3, (10, 16, 8, 112))
        self.assertEqual(shape_4, (10, 16, 8, 144))

    def test_unknown_cache_dtype_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_packed_kv_cache_shape(1, 16, 8, 128, "unknown")


if __name__ == "__main__":
    unittest.main()

