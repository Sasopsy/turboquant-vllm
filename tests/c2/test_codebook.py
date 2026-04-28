import unittest

import torch

from tilelang_turboquant.offline import CodebookSpec, get_codebook, solve_lloyd_max_beta


class TestCodebook(unittest.TestCase):
    def test_codebook_count_matches_bits(self) -> None:
        for bits in (1, 2, 3):
            centroids, midpoints = get_codebook(CodebookSpec(head_dim=128, mse_bits=bits))
            self.assertEqual(centroids.numel(), 2**bits)
            self.assertEqual(midpoints.numel(), 2**bits - 1)

    def test_codebook_centroids_sorted(self) -> None:
        centroids, _ = get_codebook(CodebookSpec(head_dim=128, mse_bits=3))
        self.assertTrue(torch.all(centroids[1:] > centroids[:-1]).item())

    def test_midpoints_match_centroids(self) -> None:
        centroids, midpoints = get_codebook(CodebookSpec(head_dim=128, mse_bits=3))
        expected = (centroids[:-1] + centroids[1:]) / 2
        self.assertTrue(torch.allclose(midpoints, expected, atol=1e-6, rtol=1e-6))

    def test_codebook_is_near_symmetric(self) -> None:
        centroids, _ = get_codebook(CodebookSpec(head_dim=128, mse_bits=3))
        self.assertLess(abs(centroids.mean().item()), 0.05)
        self.assertTrue(
            torch.allclose(centroids, -torch.flip(centroids, dims=[0]), atol=0.05, rtol=0)
        )

    def test_codebook_is_deterministic(self) -> None:
        spec = CodebookSpec(head_dim=128, mse_bits=3)
        c1, m1 = get_codebook(spec)
        c2, m2 = get_codebook(spec)
        self.assertTrue(torch.equal(c1, c2))
        self.assertTrue(torch.equal(m1, m2))

    def test_solver_output_contract(self) -> None:
        centroids, midpoints = solve_lloyd_max_beta(head_dim=96, mse_bits=2)
        self.assertEqual(centroids.dtype, torch.float32)
        self.assertEqual(midpoints.dtype, torch.float32)
        self.assertEqual(centroids.device.type, "cpu")
        self.assertEqual(midpoints.device.type, "cpu")
        self.assertTrue(centroids.is_contiguous())
        self.assertTrue(midpoints.is_contiguous())


if __name__ == "__main__":
    unittest.main()

