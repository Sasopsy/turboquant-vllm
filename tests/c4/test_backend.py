import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from tilelang_turboquant.backend import TileLangTQAttentionBackend
from tilelang_turboquant.memory import get_packed_kv_cache_shape
from tilelang_turboquant.plugin import register_all

from vllm.v1.attention.backends.registry import AttentionBackendEnum


class TestTileLangTQAttentionBackend(unittest.TestCase):
    def setUp(self) -> None:
        register_all()

    def test_backend_supports_only_plugin_cache_dtypes(self) -> None:
        self.assertTrue(TileLangTQAttentionBackend.supports_kv_cache_dtype("tilelang_tq_3bit"))
        self.assertTrue(TileLangTQAttentionBackend.supports_kv_cache_dtype("tq_4bit"))
        self.assertFalse(TileLangTQAttentionBackend.supports_kv_cache_dtype("fp8"))
        self.assertFalse(TileLangTQAttentionBackend.supports_kv_cache_dtype(None))

    def test_backend_shape_matches_c1_contract(self) -> None:
        expected = get_packed_kv_cache_shape(4, 16, 2, 128, "tilelang_tq_3bit")
        actual = TileLangTQAttentionBackend.get_kv_cache_shape(4, 16, 2, 128, "tilelang_tq_3bit")
        self.assertEqual(actual, expected)

    def test_validate_configuration_rejects_variant_mismatch(self) -> None:
        reasons = TileLangTQAttentionBackend.validate_configuration(
            head_size=128,
            dtype=torch.float16,
            kv_cache_dtype="fp8",
            block_size=16,
            use_mla=False,
            has_sink=False,
            use_sparse=False,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=SimpleNamespace(major=9, minor=0),
            attn_type="decoder",
        )
        self.assertTrue(any("kv_cache_dtype" in reason for reason in reasons))

    def test_validate_configuration_allows_non_power_of_two_in_qr_mode(self) -> None:
        with mock.patch.dict(os.environ, {"TILELANG_TQ_ROTATION_MODE": "qr"}, clear=False):
            reasons = TileLangTQAttentionBackend.validate_configuration(
                head_size=96,
                dtype=torch.float16,
                kv_cache_dtype="tilelang_tq_3bit",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
                use_mm_prefix=False,
                use_per_head_quant_scales=False,
                device_capability=SimpleNamespace(major=9, minor=0),
                attn_type="decoder",
            )
        self.assertEqual(reasons, [])

    def test_validate_configuration_rejects_unsupported_hadamard_shape(self) -> None:
        with mock.patch.dict(
            os.environ,
            {
                "TILELANG_TQ_ROTATION_MODE": "randomized_hadamard",
                "TILELANG_TQ_ALLOW_HADAMARD_PADDING": "0",
            },
            clear=False,
        ):
            reasons = TileLangTQAttentionBackend.validate_configuration(
                head_size=96,
                dtype=torch.float16,
                kv_cache_dtype="tilelang_tq_3bit",
                block_size=16,
                use_mla=False,
                has_sink=False,
                use_sparse=False,
                use_mm_prefix=False,
                use_per_head_quant_scales=False,
                device_capability=SimpleNamespace(major=9, minor=0),
                attn_type="decoder",
            )
        self.assertTrue(any("randomized_hadamard" in reason for reason in reasons))

    def test_backend_registered_under_custom_attention_backend(self) -> None:
        register_all()
        self.assertEqual(
            AttentionBackendEnum.CUSTOM.get_path(),
            "tilelang_turboquant.backend.backend.TileLangTQAttentionBackend",
        )


if __name__ == "__main__":
    unittest.main()
