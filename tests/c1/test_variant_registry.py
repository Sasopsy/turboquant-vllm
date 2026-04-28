import unittest

from tilelang_turboquant.config.variant_registry import (
    PLUGIN_KV_CACHE_DTYPE_3BIT,
    PLUGIN_KV_CACHE_DTYPE_4BIT,
    VARIANT_REGISTRY,
    VariantSpec,
    get_variant,
    get_variant_by_dtype_str,
)


class TestVariantRegistry(unittest.TestCase):
    def test_registry_keys(self) -> None:
        self.assertEqual(set(VARIANT_REGISTRY.keys()), {"tq_3bit", "tq_4bit"})

    def test_variant_dtype_strings_are_plugin_owned(self) -> None:
        self.assertEqual(
            get_variant("tq_3bit").kv_cache_dtype_str, PLUGIN_KV_CACHE_DTYPE_3BIT
        )
        self.assertEqual(
            get_variant("tq_4bit").kv_cache_dtype_str, PLUGIN_KV_CACHE_DTYPE_4BIT
        )

    def test_key_and_value_qjl_enabled_for_in_scope_variants(self) -> None:
        for name in ("tq_3bit", "tq_4bit"):
            spec = get_variant(name)
            self.assertTrue(spec.key_use_qjl)
            self.assertTrue(spec.value_use_qjl)

    def test_get_variant_by_dtype_str(self) -> None:
        self.assertEqual(
            get_variant_by_dtype_str(PLUGIN_KV_CACHE_DTYPE_3BIT).name, "tq_3bit"
        )
        self.assertEqual(
            get_variant_by_dtype_str(PLUGIN_KV_CACHE_DTYPE_4BIT).name, "tq_4bit"
        )

    def test_invalid_variant_spec_rejected(self) -> None:
        with self.assertRaises(ValueError):
            VariantSpec(
                name="",
                key_quant_bits=3,
                value_quant_bits=3,
                key_use_qjl=True,
                value_use_qjl=True,
                kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_3BIT,
            )
        with self.assertRaises(ValueError):
            VariantSpec(
                name="bad",
                key_quant_bits=1,
                value_quant_bits=3,
                key_use_qjl=False,
                value_use_qjl=True,
                kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_3BIT,
            )
        with self.assertRaises(ValueError):
            VariantSpec(
                name="bad",
                key_quant_bits=3,
                value_quant_bits=2,
                key_use_qjl=True,
                value_use_qjl=True,
                kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_3BIT,
            )
        with self.assertRaises(ValueError):
            VariantSpec(
                name="bad",
                key_quant_bits=2,
                value_quant_bits=3,
                key_use_qjl=True,
                value_use_qjl=True,
                kv_cache_dtype_str=PLUGIN_KV_CACHE_DTYPE_3BIT,
            )
        with self.assertRaises(ValueError):
            VariantSpec(
                name="bad",
                key_quant_bits=3,
                value_quant_bits=3,
                key_use_qjl=True,
                value_use_qjl=True,
                kv_cache_dtype_str="not_plugin_owned",
            )

    def test_unknown_variant_lookup_raises(self) -> None:
        with self.assertRaises(KeyError):
            get_variant("unknown")
        with self.assertRaises(KeyError):
            get_variant_by_dtype_str("unknown")


if __name__ == "__main__":
    unittest.main()

