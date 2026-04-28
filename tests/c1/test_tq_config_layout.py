import unittest

from tilelang_turboquant.config import SlotLayout, TileLangTQConfig


EXPECTED_SLOTS = {
    "tq_3bit": {
        64: {"slot_size_raw": 56, "slot_size_aligned": 64, "key_mse_bytes": 16},
        96: {"slot_size_raw": 80, "slot_size_aligned": 80, "key_mse_bytes": 24},
        128: {"slot_size_raw": 104, "slot_size_aligned": 112, "key_mse_bytes": 32},
        256: {"slot_size_raw": 200, "slot_size_aligned": 208, "key_mse_bytes": 64},
    },
    "tq_4bit": {
        64: {"slot_size_raw": 72, "slot_size_aligned": 80, "key_mse_bytes": 24},
        96: {"slot_size_raw": 104, "slot_size_aligned": 112, "key_mse_bytes": 36},
        128: {"slot_size_raw": 136, "slot_size_aligned": 144, "key_mse_bytes": 48},
        256: {"slot_size_raw": 264, "slot_size_aligned": 272, "key_mse_bytes": 96},
    },
}


class TestTileLangTQConfig(unittest.TestCase):
    def test_slot_size_exact_values(self) -> None:
        for variant_name, values_by_dim in EXPECTED_SLOTS.items():
            for head_dim, expected in values_by_dim.items():
                cfg = TileLangTQConfig.from_variant_name(variant_name, head_dim)
                self.assertEqual(cfg.slot_size_raw, expected["slot_size_raw"])
                self.assertEqual(cfg.slot_size_aligned, expected["slot_size_aligned"])
                self.assertEqual(cfg.key_mse_bytes, expected["key_mse_bytes"])
                self.assertEqual(cfg.key_qjl_bits_bytes, head_dim // 8)
                self.assertEqual(cfg.value_qjl_bits_bytes, head_dim // 8)
                self.assertEqual(cfg.key_norm_bytes, 2)
                self.assertEqual(cfg.value_norm_bytes, 2)
                self.assertEqual(cfg.key_qjl_gamma_bytes, 2)
                self.assertEqual(cfg.value_qjl_gamma_bytes, 2)

    def test_centroid_counts(self) -> None:
        cfg_3 = TileLangTQConfig.from_variant_name("tq_3bit", 128)
        self.assertEqual(cfg_3.key_mse_bits, 2)
        self.assertEqual(cfg_3.value_mse_bits, 2)
        self.assertEqual(cfg_3.key_n_centroids, 4)
        self.assertEqual(cfg_3.value_n_centroids, 4)

        cfg_4 = TileLangTQConfig.from_variant_name("tq_4bit", 128)
        self.assertEqual(cfg_4.key_mse_bits, 3)
        self.assertEqual(cfg_4.value_mse_bits, 3)
        self.assertEqual(cfg_4.key_n_centroids, 8)
        self.assertEqual(cfg_4.value_n_centroids, 8)

    def test_slot_size_aligned_multiple_of_16(self) -> None:
        for variant_name in ("tq_3bit", "tq_4bit"):
            for head_dim in (64, 96, 128, 256):
                cfg = TileLangTQConfig.from_variant_name(variant_name, head_dim)
                self.assertEqual(cfg.slot_size_aligned % 16, 0)
                self.assertGreaterEqual(cfg.slot_size_aligned, cfg.slot_size_raw)
                self.assertLess(cfg.padding_bytes, 16)

    def test_slot_layout_is_monotonic(self) -> None:
        for variant_name in ("tq_3bit", "tq_4bit"):
            cfg = TileLangTQConfig.from_variant_name(variant_name, 128)
            layout = SlotLayout.from_config(cfg)
            segments = [
                (layout.key_mse_offset, layout.key_mse_size),
                (layout.key_norm_offset, layout.key_norm_size),
                (layout.key_qjl_bits_offset, layout.key_qjl_bits_size),
                (layout.key_qjl_gamma_offset, layout.key_qjl_gamma_size),
                (layout.value_mse_offset, layout.value_mse_size),
                (layout.value_norm_offset, layout.value_norm_size),
                (layout.value_qjl_bits_offset, layout.value_qjl_bits_size),
                (layout.value_qjl_gamma_offset, layout.value_qjl_gamma_size),
            ]

            self.assertEqual(segments[0][0], 0)
            for idx in range(len(segments) - 1):
                curr_offset, curr_size = segments[idx]
                next_offset, _ = segments[idx + 1]
                self.assertEqual(curr_offset + curr_size, next_offset)

    def test_slot_layout_fits_inside_aligned_size(self) -> None:
        for variant_name in ("tq_3bit", "tq_4bit"):
            cfg = TileLangTQConfig.from_variant_name(variant_name, 128)
            layout = SlotLayout.from_config(cfg)
            self.assertLessEqual(layout.raw_end, layout.slot_size_aligned)
            self.assertEqual(layout.slot_size_aligned, cfg.slot_size_aligned)

    def test_invalid_head_dim_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TileLangTQConfig.from_variant_name("tq_3bit", 0)
        with self.assertRaises(ValueError):
            TileLangTQConfig.from_variant_name("tq_3bit", 127)
        with self.assertRaises(KeyError):
            TileLangTQConfig.from_variant_name("missing", 128)


if __name__ == "__main__":
    unittest.main()

