import unittest

from tilelang_turboquant.plugin import register_all

from vllm.config.cache import CacheConfig


class TestRegisterAll(unittest.TestCase):
    def test_register_all_is_idempotent(self) -> None:
        register_all()
        register_all()

    def test_register_all_admits_plugin_cache_dtype_literals(self) -> None:
        register_all()
        cfg = CacheConfig(cache_dtype="tilelang_tq_3bit")
        self.assertEqual(cfg.cache_dtype, "tilelang_tq_3bit")

    def test_register_all_normalizes_alias_cache_dtype(self) -> None:
        register_all()
        cfg = CacheConfig(cache_dtype="tq_3bit")
        self.assertEqual(cfg.cache_dtype, "tilelang_tq_3bit")


if __name__ == "__main__":
    unittest.main()

