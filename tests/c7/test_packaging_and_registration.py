import tomllib
import unittest
from pathlib import Path

from tilelang_turboquant.plugin import register_all

from vllm.config.cache import CacheConfig
from vllm.v1.attention.backends.registry import AttentionBackendEnum


REPO_ROOT = Path(__file__).resolve().parents[2]


class TestPackagingAndRegistration(unittest.TestCase):
    def test_entry_point_declares_general_plugin(self) -> None:
        pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text())
        entry_points = pyproject["project"]["entry-points"]["vllm.general_plugins"]
        self.assertEqual(
            entry_points["tilelang_turboquant"],
            "tilelang_turboquant.plugin:register_all",
        )

    def test_register_all_is_process_local_idempotent(self) -> None:
        register_all()
        register_all()

    def test_backend_registration_visible_after_plugin_load(self) -> None:
        register_all()
        self.assertEqual(
            AttentionBackendEnum.CUSTOM.get_path(),
            "tilelang_turboquant.backend.backend.TileLangTQAttentionBackend",
        )

    def test_register_all_applies_c4_shims_before_backend_use(self) -> None:
        register_all()
        cfg = CacheConfig(cache_dtype="tilelang_tq_3bit")
        self.assertEqual(cfg.cache_dtype, "tilelang_tq_3bit")
        self.assertEqual(
            AttentionBackendEnum.CUSTOM.get_path(),
            "tilelang_turboquant.backend.backend.TileLangTQAttentionBackend",
        )


if __name__ == "__main__":
    unittest.main()
