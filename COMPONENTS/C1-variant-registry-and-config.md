# LLD: C1 — Variant Registry, KV Cache Spec & Memory Layout

**Document type:** LLD (Low-Level Design)  
**Component:** C1 — `config/variant_registry.py` + `config/tq_config.py` + plugin-local KV spec/backend shape contract  
**HLD reference:** §5 C1, §5 C2, §6 Slot Layout Reference, §7 Global Buffer Lifecycle, §11 Open Question #1  
**Status:** Canonical doc for variant constants, slot layout, KV spec, and packed-cache shape; valid for pre-TurboQuant `vLLM` trees

---

## 1. Purpose & Scope

This C1 document is the single source of truth for:

- variant registry entries
- per-variant bit budgets
- slot byte layout and alignment
- plugin-owned KV-cache dtype identifiers
- plugin-local `KVCacheSpec` shape and page-size math
- packed-cache reshape invariants

Nothing outside `VariantSpec`, `TileLangTQConfig`, `SlotLayout`, and the plugin-local KV spec is allowed to recompute slot arithmetic or page-size math from scratch.

This merged doc keeps four decisions fixed:

1. both keys and values use the TurboQuant MSE + QJL split
2. slot size is 16-byte aligned
3. cache-dtype literals are plugin-owned identifiers
4. the canonical KV spec is plugin-local, not upstream TurboQuant-dependent

---

## 2. Design Decisions

### 2.1 Value path matches the HLD

The in-scope variants are:

- `tq_3bit`: 2-bit MSE + 1-bit QJL for keys, and the same split for values
- `tq_4bit`: 3-bit MSE + 1-bit QJL for keys, and the same split for values

So the slot layout is symmetric between K and V:

```text
[key_mse | key_norm | key_qjl_bits | key_qjl_gamma |
 value_mse | value_norm | value_qjl_bits | value_qjl_gamma | padding]
```

There is no value min/max scale-zero path in this design.

### 2.2 Slot alignment follows the HLD

Canonical rule:

```python
slot_size_aligned = ceil(slot_size_raw / 16) * 16
```

### 2.3 Cache-dtype strings are plugin-owned

Canonical plugin-side identifiers:

- `tilelang_tq_3bit`
- `tilelang_tq_4bit`

Do not assume the target `vLLM` version already knows these strings or ships TurboQuant at all. Per-version admission and mapping belong to the integration layer.

### 2.4 Canonical KV spec is plugin-local

Assume the target tree may not ship:

- `TQFullAttentionSpec`
- TurboQuant cache-dtype literals
- any upstream TurboQuant integration

So the canonical spec is a plugin-local class extending `FullAttentionSpec`.

---

## 3. `VariantSpec`

**File:** `config/variant_registry.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class VariantSpec:
    name: str
    key_quant_bits: int
    value_quant_bits: int
    key_use_qjl: bool
    value_use_qjl: bool
    kv_cache_dtype_str: str

    def __post_init__(self) -> None:
        assert self.name, "variant name must be non-empty"
        assert self.key_quant_bits >= 2
        assert self.value_quant_bits >= 2
        if self.key_use_qjl:
            assert self.key_quant_bits >= 3
        if self.value_use_qjl:
            assert self.value_quant_bits >= 3
        assert self.kv_cache_dtype_str in {
            "tilelang_tq_3bit",
            "tilelang_tq_4bit",
        }
```

### Field semantics

| Field | Meaning |
|---|---|
| `name` | internal variant key, e.g. `tq_3bit` |
| `key_quant_bits` | total key bit budget including QJL when enabled |
| `value_quant_bits` | total value bit budget including QJL when enabled |
| `key_use_qjl` | whether keys reserve one bit channel for QJL |
| `value_use_qjl` | whether values reserve one bit channel for QJL |
| `kv_cache_dtype_str` | canonical plugin KV-cache dtype literal |

---

## 4. `VARIANT_REGISTRY`

```python
VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "tq_3bit": VariantSpec(
        name="tq_3bit",
        key_quant_bits=3,
        value_quant_bits=3,
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str="tilelang_tq_3bit",
    ),
    "tq_4bit": VariantSpec(
        name="tq_4bit",
        key_quant_bits=4,
        value_quant_bits=4,
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str="tilelang_tq_4bit",
    ),
}
```

Lookup helpers:

```python
def get_variant(name: str) -> VariantSpec:
    ...


def get_variant_by_dtype_str(dtype_str: str) -> VariantSpec:
    ...
```

---

## 5. `TileLangTQConfig`

**File:** `config/tq_config.py`

```python
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class TileLangTQConfig:
    variant_name: str
    head_dim: int

    @classmethod
    def from_variant_name(cls, variant_name: str, head_dim: int) -> "TileLangTQConfig":
        assert head_dim > 0 and head_dim % 8 == 0
        get_variant(variant_name)
        return cls(variant_name=variant_name, head_dim=head_dim)

    @property
    def variant(self) -> VariantSpec:
        return get_variant(self.variant_name)

    @property
    def key_mse_bits(self) -> int:
        return self.variant.key_quant_bits - 1 if self.variant.key_use_qjl else self.variant.key_quant_bits

    @property
    def value_mse_bits(self) -> int:
        return self.variant.value_quant_bits - 1 if self.variant.value_use_qjl else self.variant.value_quant_bits

    @property
    def key_n_centroids(self) -> int:
        return 2 ** self.key_mse_bits

    @property
    def value_n_centroids(self) -> int:
        return 2 ** self.value_mse_bits

    @property
    def key_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.key_mse_bits / 8)

    @property
    def key_norm_bytes(self) -> int:
        return 2

    @property
    def key_qjl_bits_bytes(self) -> int:
        return math.ceil(self.head_dim / 8) if self.variant.key_use_qjl else 0

    @property
    def key_qjl_gamma_bytes(self) -> int:
        return 2 if self.variant.key_use_qjl else 0

    @property
    def value_mse_bytes(self) -> int:
        return math.ceil(self.head_dim * self.value_mse_bits / 8)

    @property
    def value_norm_bytes(self) -> int:
        return 2

    @property
    def value_qjl_bits_bytes(self) -> int:
        return math.ceil(self.head_dim / 8) if self.variant.value_use_qjl else 0

    @property
    def value_qjl_gamma_bytes(self) -> int:
        return 2 if self.variant.value_use_qjl else 0

    @property
    def key_side_bytes(self) -> int:
        return self.key_mse_bytes + self.key_norm_bytes + self.key_qjl_bits_bytes + self.key_qjl_gamma_bytes

    @property
    def value_side_bytes(self) -> int:
        return self.value_mse_bytes + self.value_norm_bytes + self.value_qjl_bits_bytes + self.value_qjl_gamma_bytes

    @property
    def slot_size_raw(self) -> int:
        return self.key_side_bytes + self.value_side_bytes

    @property
    def slot_size_aligned(self) -> int:
        return ((self.slot_size_raw + 15) // 16) * 16

    @property
    def padding_bytes(self) -> int:
        return self.slot_size_aligned - self.slot_size_raw
```

---

## 6. `SlotLayout`

**File:** `config/tq_config.py`

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class SlotLayout:
    key_mse_offset: int
    key_mse_size: int
    key_norm_offset: int
    key_norm_size: int
    key_qjl_bits_offset: int
    key_qjl_bits_size: int
    key_qjl_gamma_offset: int
    key_qjl_gamma_size: int
    value_mse_offset: int
    value_mse_size: int
    value_norm_offset: int
    value_norm_size: int
    value_qjl_bits_offset: int
    value_qjl_bits_size: int
    value_qjl_gamma_offset: int
    value_qjl_gamma_size: int
    slot_size_aligned: int

    @classmethod
    def from_config(cls, cfg: TileLangTQConfig) -> "SlotLayout":
        key_mse_offset = 0
        key_norm_offset = key_mse_offset + cfg.key_mse_bytes
        key_qjl_bits_offset = key_norm_offset + cfg.key_norm_bytes
        key_qjl_gamma_offset = key_qjl_bits_offset + cfg.key_qjl_bits_bytes

        value_mse_offset = key_qjl_gamma_offset + cfg.key_qjl_gamma_bytes
        value_norm_offset = value_mse_offset + cfg.value_mse_bytes
        value_qjl_bits_offset = value_norm_offset + cfg.value_norm_bytes
        value_qjl_gamma_offset = value_qjl_bits_offset + cfg.value_qjl_bits_bytes

        return cls(
            key_mse_offset=key_mse_offset,
            key_mse_size=cfg.key_mse_bytes,
            key_norm_offset=key_norm_offset,
            key_norm_size=cfg.key_norm_bytes,
            key_qjl_bits_offset=key_qjl_bits_offset,
            key_qjl_bits_size=cfg.key_qjl_bits_bytes,
            key_qjl_gamma_offset=key_qjl_gamma_offset,
            key_qjl_gamma_size=cfg.key_qjl_gamma_bytes,
            value_mse_offset=value_mse_offset,
            value_mse_size=cfg.value_mse_bytes,
            value_norm_offset=value_norm_offset,
            value_norm_size=cfg.value_norm_bytes,
            value_qjl_bits_offset=value_qjl_bits_offset,
            value_qjl_bits_size=cfg.value_qjl_bits_bytes,
            value_qjl_gamma_offset=value_qjl_gamma_offset,
            value_qjl_gamma_size=cfg.value_qjl_gamma_bytes,
            slot_size_aligned=cfg.slot_size_aligned,
        )
```

Validation rules:

```python
layout = SlotLayout.from_config(cfg)
raw_end = layout.value_qjl_gamma_offset + layout.value_qjl_gamma_size

assert layout.key_mse_offset == 0
assert raw_end <= layout.slot_size_aligned
assert layout.slot_size_aligned % 16 == 0
assert cfg.padding_bytes < 16
```

---

## 7. Computed Slot Values

### `tq_3bit`

| `head_dim` | `key_mse_bytes` | `key_qjl_bits_bytes` | `value_mse_bytes` | `value_qjl_bits_bytes` | `slot_size_raw` | `slot_size_aligned` |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 16 | 8 | 16 | 8 | 56 | 64 |
| 96 | 24 | 12 | 24 | 12 | 80 | 80 |
| 128 | 32 | 16 | 32 | 16 | 104 | 112 |
| 256 | 64 | 32 | 64 | 32 | 200 | 208 |

### `tq_4bit`

| `head_dim` | `key_mse_bytes` | `key_qjl_bits_bytes` | `value_mse_bytes` | `value_qjl_bits_bytes` | `slot_size_raw` | `slot_size_aligned` |
|---|---:|---:|---:|---:|---:|---:|
| 64 | 24 | 8 | 24 | 8 | 72 | 80 |
| 96 | 36 | 12 | 36 | 12 | 104 | 112 |
| 128 | 48 | 16 | 48 | 16 | 136 | 144 |
| 256 | 96 | 32 | 96 | 32 | 264 | 272 |

### D=128 slot maps

**`tq_3bit`, `head_dim=128`, `slot_size_aligned=112`:**

```text
0    32   key_mse
32    2   key_norm
34   16   key_qjl_bits
50    2   key_qjl_gamma
52   32   value_mse
84    2   value_norm
86   16   value_qjl_bits
102   2   value_qjl_gamma
104   8   padding
```

**`tq_4bit`, `head_dim=128`, `slot_size_aligned=144`:**

```text
0    48   key_mse
48    2   key_norm
50   16   key_qjl_bits
66    2   key_qjl_gamma
68   48   value_mse
116   2   value_norm
118  16   value_qjl_bits
134   2   value_qjl_gamma
136   8   padding
```

---

## 8. Plugin-Local `KVCacheSpec`

The canonical C2 implementation is a plugin-local spec class that extends `FullAttentionSpec` and overrides `real_page_size_bytes`.

```python
from dataclasses import dataclass, replace

from vllm.v1.kv_cache_interface import FullAttentionSpec


@dataclass(frozen=True, kw_only=True)
class TileLangTQAttentionSpec(FullAttentionSpec):
    tq_slot_size: int = 0
    tq_variant_name: str = ""

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list["TileLangTQAttentionSpec"]) -> "TileLangTQAttentionSpec":
        merged = super().merge(specs)
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs)
        assert all(s.tq_variant_name == specs[0].tq_variant_name for s in specs)
        return replace(
            merged,
            tq_slot_size=specs[0].tq_slot_size,
            tq_variant_name=specs[0].tq_variant_name,
        )
```

If a newer target tree already ships `TQFullAttentionSpec`, the integration layer may alias to it or subclass it, but that is an optimization, not the canonical design.

---

## 9. Spec Construction and Shape Contract

The config/quantization layer constructs the spec from this unified C1 state:

```python
cfg = TileLangTQConfig.from_variant_name(variant_name, head_dim)

spec = TileLangTQAttentionSpec(
    block_size=block_size,
    num_kv_heads=num_kv_heads,
    head_size=head_dim,
    head_size_v=head_dim,
    dtype=kv_cache_torch_dtype,
    tq_slot_size=cfg.slot_size_aligned,
    tq_variant_name=variant_name,
)
```

Required fields:

| Field | Source |
|---|---|
| `block_size` | `vllm_config.cache_config.block_size` |
| `num_kv_heads` | attention layer |
| `head_size` | attention layer |
| `head_size_v` | same as `head_size` in this design |
| `dtype` | plugin-local 1-byte cache dtype, typically `torch.int8` |
| `tq_slot_size` | `cfg.slot_size_aligned` |
| `tq_variant_name` | plugin variant name |

Canonical backend shape:

```python
@staticmethod
def get_kv_cache_shape(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_size: int,
    cache_dtype_str: str = "tilelang_tq_3bit",
) -> tuple[int, int, int, int]:
    cfg = TileLangTQConfig.from_variant_name(
        get_variant_by_dtype_str(cache_dtype_str).name,
        head_size,
    )
    return (num_blocks, block_size, num_kv_heads, cfg.slot_size_aligned)
```

Shape semantics:

| Dimension | Meaning |
|---|---|
| `dim 0` | physical blocks |
| `dim 1` | tokens per block |
| `dim 2` | KV heads |
| `dim 3` | packed slot bytes |

There is no leading `2` dimension because K and V are packed into the same slot.

---

## 10. Page Size and Reshape Invariant

Canonical page-size formula:

```python
real_page_size_bytes = block_size * num_kv_heads * tq_slot_size
```

Because this spec uses no per-token-head scale side budget:

```python
page_size_bytes == real_page_size_bytes
```

So the reshape invariant becomes:

```python
math.prod(shape) == raw_tensor.numel() // element_size(spec.dtype)
```

Since the packed-cache dtype is 1 byte:

```python
num_blocks * block_size * num_kv_heads * slot_size_aligned
==
num_blocks * page_size_bytes
```

which is exactly the same formula as `real_page_size_bytes`.

---

## 11. Concrete Page Sizes

Using `block_size=16`, `num_kv_heads=8`:

| Variant | `head_dim` | `slot_size_aligned` | `page_size_bytes` |
|---|---:|---:|---:|
| `tq_3bit` | 64 | 64 | 8,192 |
| `tq_3bit` | 96 | 80 | 10,240 |
| `tq_3bit` | 128 | 112 | 14,336 |
| `tq_3bit` | 256 | 208 | 26,624 |
| `tq_4bit` | 64 | 80 | 10,240 |
| `tq_4bit` | 96 | 112 | 14,336 |
| `tq_4bit` | 128 | 144 | 18,432 |
| `tq_4bit` | 256 | 272 | 34,816 |

### D=128 comparison against BF16

BF16 baseline:

```python
2 * 16 * 8 * 128 * 2 = 65_536 bytes
```

So:

| Variant | Page bytes | Compression vs BF16 |
|---|---:|---:|
| `tq_3bit` | 14,336 | about 4.57x |
| `tq_4bit` | 18,432 | about 3.56x |

---

## 12. `vLLM`-Facing Contracts

### 12.1 Cache-dtype identifiers

Canonical plugin values:

- `tilelang_tq_3bit`
- `tilelang_tq_4bit`

Optional compatibility aliases on newer trees may include upstream TurboQuant names, but those are integration-layer details, not C1 truth.

### 12.2 Byte dtype for packed storage

The canonical packed-cache assumption is a byte-addressable buffer:

- prefer `torch.int8`
- allow `torch.uint8` if a target branch requires it

This does not change any slot or page arithmetic as long as element size remains 1 byte.

### 12.3 Profiling and store-path rules

Downstream components must honor these C1 execution rules:

- store-path row bound is `slot_mapping.shape[0]`
- `slot_mapping[i] < 0` means “do not write this row”
- all GPU-resident store/decode buffers must exist before `profile_run()`

---

## 13. Downstream Interface Contracts

For the backend/spec layer:

```python
cfg = TileLangTQConfig.from_variant_name(variant_name, head_dim)
tq_slot_size = cfg.slot_size_aligned
```

For kernels:

```python
layout = SlotLayout.from_config(cfg)
```

Both store and decode must import the same layout and treat the padding tail as unused bytes.

---

## 14. Testing Contracts

| Test | Expectation |
|---|---|
| `test_variant_registry_keys` | registry contains `tq_3bit` and `tq_4bit` |
| `test_variant_dtype_strings_are_plugin_owned` | dtype strings are `tilelang_tq_3bit` / `tilelang_tq_4bit` |
| `test_key_and_value_qjl_enabled_for_in_scope_variants` | both bools are `True` |
| `test_slot_size_aligned_multiple_of_16` | `slot_size_aligned % 16 == 0` |
| `test_slot_size_exact_values` | values match the tables above |
| `test_slot_layout_is_monotonic` | offsets increase without overlap |
| `test_slot_layout_fits_inside_aligned_size` | raw end <= aligned size |
| `test_centroid_counts` | `tq_3bit -> 4`, `tq_4bit -> 8` for both key and value |
| `test_spec_uses_plugin_local_tq_spec` | merged doc uses plugin-local `TileLangTQAttentionSpec` |
| `test_real_page_size_formula` | `page == block_size * num_kv_heads * slot_size_aligned` |
| `test_shape_product_matches_page_bytes` | reshape invariant holds |
| `test_d128_page_sizes` | `14336` for `tq_3bit`, `18432` for `tq_4bit` |
| `test_kv_cache_shape_tuple` | backend returns `(num_blocks, block_size, num_kv_heads, slot_size_aligned)` |
| `test_store_contract_uses_slot_mapping_length` | docs/tests follow padded split-KV rule |
| `test_no_post_profile_gpu_alloc_assumption` | integration checks verify buffers exist before profiling |

---

## 15. HLD Alignment Notes

This merged doc intentionally preserves the corrected design in four places:

1. values use TurboQuant MSE + QJL, not scale/zero-point quantization
2. slot alignment is 16-byte based, yielding `112 B` and `144 B` aligned slots at `head_dim=128`
3. cache-dtype identifiers stay plugin-owned for older target trees
4. the canonical KV spec is plugin-local and remains valid even when upstream TurboQuant is absent
