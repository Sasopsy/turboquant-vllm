import unittest
import warnings

import torch

from tilelang_turboquant.offline import QJLSpec, RotationSpec, get_rotation, get_s_matrix


class TestMatrices(unittest.TestCase):
    def test_qr_rotation_is_orthonormal(self) -> None:
        rotation, rotation_t, is_symmetric = get_rotation(
            RotationSpec(head_dim=128, mode="qr", seed=7)
        )
        eye = torch.eye(128, dtype=torch.float32)
        self.assertTrue(torch.allclose(rotation @ rotation_t, eye, atol=1e-5, rtol=1e-5))
        self.assertFalse(is_symmetric)

    def test_qr_rotation_is_deterministic(self) -> None:
        rot1, rot1_t, _ = get_rotation(RotationSpec(128, "qr", 11))
        rot2, rot2_t, _ = get_rotation(RotationSpec(128, "qr", 11))
        self.assertTrue(torch.equal(rot1, rot2))
        self.assertTrue(torch.equal(rot1_t, rot2_t))

    def test_hadamard_is_deterministic_for_power_of_two_dims(self) -> None:
        rot1, rot1_t, _ = get_rotation(RotationSpec(128, "randomized_hadamard", 33))
        rot2, rot2_t, _ = get_rotation(RotationSpec(128, "randomized_hadamard", 33))
        self.assertTrue(torch.equal(rot1, rot2))
        self.assertTrue(torch.equal(rot1_t, rot2_t))

        eye = torch.eye(128, dtype=torch.float32)
        self.assertTrue(torch.allclose(rot1 @ rot1_t, eye, atol=1e-5, rtol=1e-5))

    def test_hadamard_non_power_of_two_falls_back_to_qr_with_warning(self) -> None:
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            rot_fallback, rot_fallback_t, _ = get_rotation(
                RotationSpec(96, "randomized_hadamard", 101)
            )
        self.assertGreater(len(captured), 0)
        self.assertTrue(
            any("falling back to qr" in str(w.message).lower() for w in captured)
        )

        rot_qr, rot_qr_t, _ = get_rotation(RotationSpec(96, "qr", 101))
        self.assertTrue(torch.equal(rot_fallback, rot_qr))
        self.assertTrue(torch.equal(rot_fallback_t, rot_qr_t))

    def test_s_matrix_is_deterministic_and_seed_sensitive(self) -> None:
        s1 = get_s_matrix(QJLSpec(head_dim=128, seed=5))
        s2 = get_s_matrix(QJLSpec(head_dim=128, seed=5))
        s3 = get_s_matrix(QJLSpec(head_dim=128, seed=6))
        self.assertTrue(torch.equal(s1, s2))
        self.assertFalse(torch.equal(s1, s3))

    def test_tensor_contracts(self) -> None:
        rotation, rotation_t, _ = get_rotation(RotationSpec(64, "qr", 3))
        s_matrix = get_s_matrix(QJLSpec(64, 3))

        for tensor in (rotation, rotation_t, s_matrix):
            self.assertEqual(tensor.dtype, torch.float32)
            self.assertEqual(tensor.device.type, "cpu")
            self.assertTrue(tensor.is_contiguous())


if __name__ == "__main__":
    unittest.main()

