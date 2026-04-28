import unittest

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.offline import CodebookSpec, get_codebook


class TestC1C2Integration(unittest.TestCase):
    def test_key_value_bits_derive_from_c1_and_build_codebooks(self) -> None:
        for variant_name, expected_bits in (("tq_3bit", 2), ("tq_4bit", 3)):
            cfg = TileLangTQConfig.from_variant_name(variant_name=variant_name, head_dim=128)
            self.assertEqual(cfg.key_mse_bits, expected_bits)
            self.assertEqual(cfg.value_mse_bits, expected_bits)

            key_centroids, key_midpoints = get_codebook(
                CodebookSpec(head_dim=cfg.head_dim, mse_bits=cfg.key_mse_bits)
            )
            value_centroids, value_midpoints = get_codebook(
                CodebookSpec(head_dim=cfg.head_dim, mse_bits=cfg.value_mse_bits)
            )

            self.assertEqual(key_centroids.numel(), 2**expected_bits)
            self.assertEqual(value_centroids.numel(), 2**expected_bits)
            self.assertEqual(key_midpoints.numel(), 2**expected_bits - 1)
            self.assertEqual(value_midpoints.numel(), 2**expected_bits - 1)


if __name__ == "__main__":
    unittest.main()

