# High-Level Design: TileLang TurboQuant Plugin for vLLM

**Document type:** HLD (High-Level Design)  
**Scope:** Out-of-tree vLLM plugin implementing TurboQuant KV-cache compression using TileLang kernels, packaged as an installable Python plugin.  
**Algorithm authority:** `turboquant_algorithm.md` is the primary specification. Where the existing vLLM Triton implementation deviates, this HLD follows the algorithm spec.  
**Variants in scope:** `tq_3bit`, `tq_4bit` (extensible to arbitrary future variants via a unified variant registry).

---

## Table of Contents

1. [Problem Statement & Goals](#1-problem-statement--goals)
2. [Algorithm Mapping](#2-algorithm-mapping)
3. [System Architecture](#3-system-architecture)
4. [Package Structure](#4-package-structure)
5. [Component Designs](#5-component-designs)
   - [C1: Variant Registry, KV Cache Spec & Memory Layout](#c1-variant-registry-kv-cache-spec--memory-layout)
   - [C2: Offline Pre-computation](#c2-offline-pre-computation)
   - [C3: Quantization Config & KV Cache Method](#c3-quantization-config--kv-cache-method)
   - [C4: Attention Backend & Metadata Builder](#c4-attention-backend--metadata-builder)
   - [C5: Attention Implementation (Prefill/Decode Dispatch)](#c5-attention-implementation-prefilldecode-dispatch)
   - [C6: TileLang Kernels (Store & Decode)](#c6-tilelang-kernels-store--decode)
   - [C7: Plugin Registration & Entry Point](#c7-plugin-registration--entry-point)
6. [Slot Layout Reference](#6-slot-layout-reference)
7. [Global Buffer Lifecycle](#7-global-buffer-lifecycle)
8. [Extensibility Design](#8-extensibility-design)
9. [Testing Plans](#9-testing-plans)
10. [Benchmark Plan](#10-benchmark-plan)
11. [Open Questions / LLD Decisions](#11-open-questions--lld-decisions)

---

## 1. Problem Statement & Goals

### 1.1 Problem

vLLM's KV cache grows linearly with context length and batch size, consuming the majority of GPU VRAM. For a 32-layer model with 8 KV heads and `head_size=128` at bfloat16, a single block of 16 tokens costs 65,536 bytes. This limits throughput and maximum context length.

TurboQuant (Abbe, Du, Saad-Falcon et al.) addresses this by quantizing K/V vectors after applying a random orthogonal rotation, exploiting the rotated-coordinate distribution (a scaled Beta) to achieve near-optimal quantization with a precomputed Lloyd-Max codebook.

The existing vLLM implementation uses Triton kernels. This project replaces those kernels with **TileLang** kernels, which offer:
- Tile-level parallelism primitives better suited to blocked KV operations
- Cleaner expression of paged decode patterns
- A path toward future hardware (Hopper warp specialization) without rewriting host code

### 1.2 Goals

| Goal | Metric |
|------|--------|
| Correctness | Attention output within numerical tolerance of bfloat16 baseline |
| Memory reduction | ≥ 4× blocks for 3b K/V, ≥ 3× for 4b K/V vs. bfloat16 baseline |
| Decode throughput | ≥ parity with Triton implementation |
| Prefill throughput | ≤ 10% regression vs. FlashAttention baseline |
| Extensibility | Adding a new bit-width variant requires only a new `VariantSpec` entry |
| Deployment | Works as an installable `pip` package, zero vLLM source modifications |

### 1.3 Non-Goals

- Weight quantization (this is KV cache quantization only)
- Training or fine-tuning support
- Non-CUDA platforms (ROCm, XPU) in v1
- Attention types other than decoder self-attention

---

## 2. Algorithm Mapping

This section maps the `turboquant_algorithm.md` spec to the two KV cache variants. The reference `turboquant` repo uses TurboQuant for keys and group min/max quantization for values; this plugin intentionally applies TurboQuant to values as well so both halves of KV use the same rotation + Lloyd-Max + QJL residual structure.

### 2.1 TurboQuant_mse (Algorithm 1)

Used as the base quantizer inside `TurboQuant_prod` for both keys and values.

**Offline (once at server startup):**
1. Generate orthogonal rotation matrix `Π ∈ ℝ^{d×d}`. The reference repo uses QR decomposition of a seeded Gaussian matrix. For the TileLang plugin, the implementation may use a randomized Hadamard transform (`Π = H · D_sign · P_perm`) if it preserves the same spherical coordinate distribution and passes the reference accuracy tests.
2. Compute Lloyd-Max codebook `C = {c_1, ..., c_{2^b}}` for the Beta-distribution prior `f_X(x)`.

**Store (per new token):**
1. Normalize input vector: `x_unit = x / ‖x‖₂`
2. Rotate: `y = Π · x_unit`
3. Quantize: for each coordinate `j`, `idx_j = argmin_k |y_j - c_k|`
4. Store: packed `idx` array + `‖x‖₂` (norm scalar)

**Load / Dequant (during attention):**
1. Recover `y_tilde_j = c_{idx_j}` for all `j`
2. Inverse-rotate: `x_tilde = Π^T · y_tilde`
3. Rescale by stored norm: `x_tilde = x_tilde * ‖x‖₂`

### 2.2 TurboQuant_prod (Algorithm 2)

Used for keys and values in both in-scope variants. For keys, the estimator targets query-key logits. For values, the dequantized vector participates in the weighted sum `Σ_i softmax(q·k_i) v_i`; using `TurboQuant_prod` gives an unbiased vector estimate for `v_i`, so the weighted output remains unbiased conditioned on the attention weights.

**Offline (once at server startup):**
1. Instantiate `TurboQuant_mse(b=b-1)` as the base quantizer.
2. Generate random Gaussian projection matrix `S ∈ ℝ^{d×d}`, `S_{i,j} ~ N(0,1)`.

**Store (per new token):**
1. `idx = Quant_mse(x)` using (b-1) bits
2. `x_tilde = DeQuant_mse(idx)` — reconstruct approximation
3. `r = x - x_tilde` — residual vector
4. `qjl = sign(S · r)` — 1-bit QJL projection of residual
5. `γ = ‖r‖₂` — residual norm
6. Store: `(idx, qjl, γ)`

**Load / Dequant (during attention):**
1. `x_tilde_mse = DeQuant_mse(idx)` — base component
2. `x_tilde_qjl = (√(π/2) / d) · γ · S^T · qjl` — residual approximation
3. `x_tilde = x_tilde_mse + x_tilde_qjl` — combined estimate

### 2.3 Variant Mapping Table

| Variant | Key Algorithm | Value Algorithm | Total bits/dim |
|---------|--------------|-----------------|----------------|
| `tq_3bit` | `TurboQuant_prod(b=3)` = MSE(2)+QJL(1) | `TurboQuant_prod(b=3)` = MSE(2)+QJL(1) | ~6 bits/pair before metadata |
| `tq_4bit` | `TurboQuant_prod(b=4)` = MSE(3)+QJL(1) | `TurboQuant_prod(b=4)` = MSE(3)+QJL(1) | ~8 bits/pair before metadata |

> **Note on values:** The reference repo uses asymmetric per-group min/max quantization for values. This HLD deliberately differs: values are still multiplied by scalar attention weights and summed, so a vector-unbiased TurboQuant reconstruction is a valid target. The LLD must validate quality and latency against the repo's group-quantized value baseline because this choice stores extra value QJL metadata.


---

## 3. System Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                    Installed Package: tilelang-turboquant               │
│                                                                          │
│  ┌─────────────────┐    ┌──────────────────────────────────────────┐    │
│  │  plugin.py      │    │  Model Server (vLLM V1)                   │    │
│  │  register_all() │───►│  LLM(quantization="tq_3bit",             │    │
│  │  (entry point)  │    │     kv_cache_dtype="tilelang_tq_3bit",   │    │
│  └─────────────────┘    │     attention_backend="CUSTOM")          │    │
│                          └──────────────┬───────────────────────────┘    │
│                                         │                                 │
│          ┌──────────────────────────────▼───────────────────────────┐    │
│          │                   Startup Flow                             │    │
│          │  1. register_quantization_config("tq_3bit", TQConfig3)   │    │
│          │  2. register_backend(CUSTOM, TileLangTQBackend)           │    │
│          │  3. model.__init__: _init_tq_buffers() → register_buffer │    │
│          │  4. profile_run (measures buffers in peak memory)         │    │
│          │  5. available_memory = total - peak                       │    │
│          │  6. TileLangTQSpec.real_page_size_bytes → num_blocks      │    │
│          │  7. _allocate_kv_cache_tensors (int8 blob)                │    │
│          │  8. _reshape → (blocks, block_size, heads, slot_size)     │    │
│          │  9. bind_kv_cache → layer.kv_cache = shaped tensor        │    │
│          └──────────────────────────────────────────────────────────┘    │
│                                                                          │
│          ┌──────────────────────────────────────────────────────────┐    │
│          │                   Per-Step Forward Flow                   │    │
│          │                                                            │    │
│          │   unified_kv_cache_update()                                │    │
│          │    └► do_kv_cache_update(key, value, kv_cache, slots)     │    │
│          │         └► TileLang Store Kernel [C6]                     │    │
│          │              • Rotate K with Π (QR or randomized Hadamard)│    │
│          │              • Lloyd-Max quantize K → packed indices       │    │
│          │              • QJL residual encode (both variants)         │    │
│          │              • Rotate + Lloyd-Max + QJL encode V           │    │
│          │              • Scatter to kv_cache[block, pos, head, :]   │    │
│          │                                                            │    │
│          │   unified_attention_with_output()                          │    │
│          │    └► impl.forward(query, kv_cache, attn_metadata, out)   │    │
│          │         ├── Prefill → FlashAttention (first chunk)        │    │
│          │         ├── Continuation → TileLang Decode or dequant     │    │
│          │         └── Decode → TileLang Decode Kernel [C6]          │    │
│          │              • Load slot → unpack K indices + V data      │    │
│          │              • Dequant K (centroids + inverse-rotate)     │    │
│          │              • QJL residual reconstruct (both variants)   │    │
│          │              • Dequant V (centroids + inverse-rotate + QJL)│    │
│          │              • Paged softmax + weighted sum → output      │    │
│          └──────────────────────────────────────────────────────────┘    │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 4. Package Structure

```
tilelang_turboquant/
├── pyproject.toml                    # entry point: vllm.general_plugins
├── plugin.py                         # [C7] register_all() entry point
│
├── config/
│   ├── __init__.py
│   ├── variant_registry.py           # [C1] VariantSpec, VARIANT_REGISTRY
│   └── tq_config.py                  # [C1] TileLangTQConfig dataclass
│
├── memory/
│   ├── __init__.py
│   └── kv_spec.py                    # [C1] TileLangTQAttentionSpec
│
├── offline/
│   ├── __init__.py
│   ├── codebook.py                   # [C2] Lloyd-Max centroid generation
│   └── matrices.py                   # [C2] rotation builder, QJL S matrix
│
├── quantization/
│   ├── __init__.py
│   ├── quant_config.py               # [C3] TileLangTQQuantizationConfig
│   └── kv_cache_method.py            # [C3] TileLangTQKVCacheMethod
│
├── backend/
│   ├── __init__.py
│   ├── backend.py                    # [C4] TileLangTQAttentionBackend
│   ├── metadata.py                   # [C4] TileLangTQMetadata + Builder
│   └── impl.py                       # [C5] TileLangTQAttentionImpl
│
├── kernels/
│   ├── __init__.py
│   ├── tl_store.py                   # [C6] TileLang store kernel
│   └── tl_decode.py                  # [C6] TileLang decode kernel
│
└── tests/
    ├── unit/
    │   ├── test_variant_registry.py
    │   ├── test_kv_spec.py
    │   ├── test_codebook.py
    │   ├── test_matrices.py
    │   ├── test_quant_config.py
    │   ├── test_kv_cache_method.py
    │   ├── test_backend.py
    │   ├── test_metadata_builder.py
    │   ├── test_impl.py
    │   ├── test_tl_store.py
    │   └── test_tl_decode.py          # both belong to C6
    ├── integration/
    │   ├── test_plugin_registration.py
    │   ├── test_kv_cache_allocation.py
    │   ├── test_roundtrip_accuracy.py
    │   └── test_vllm_end_to_end.py
    └── benchmarks/
        ├── bench_store_kernel.py
        ├── bench_decode_kernel.py
        └── bench_model_throughput.py
```

---

## 5. Component Designs

---

### C1: Variant Registry, KV Cache Spec & Memory Layout

**Purpose:** Centralize all bit-width, quantization strategy, slot layout, and KV page-size decisions. Any new variant is one registry entry plus derived layout math, with no code changes elsewhere.

**Key Abstractions:**

```python
@dataclass(frozen=True)
class VariantSpec:
    name: str                       # e.g. "tq_3bit"
    key_quant_bits: int             # b for TurboQuant_mse
    value_quant_bits: int           # b for value TurboQuant_prod
    key_use_qjl: bool               # True → TurboQuant_prod for keys
    value_use_qjl: bool             # True → TurboQuant_prod for values
    kv_cache_dtype_str: str         # string registered with vLLM's CacheDType

VARIANT_REGISTRY: dict[str, VariantSpec] = {
    "tq_3bit": VariantSpec(
        name="tq_3bit",
        key_quant_bits=3,         # 2-bit MSE + 1-bit QJL (TurboQuant_prod)
        value_quant_bits=3,       # 2-bit MSE + 1-bit QJL (TurboQuant_prod)
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str="tilelang_tq_3bit",
    ),
    "tq_4bit": VariantSpec(
        name="tq_4bit",
        key_quant_bits=4,         # 3-bit MSE + 1-bit QJL (TurboQuant_prod)
        value_quant_bits=4,       # 3-bit MSE + 1-bit QJL (TurboQuant_prod)
        key_use_qjl=True,
        value_use_qjl=True,
        kv_cache_dtype_str="tilelang_tq_4bit",
    ),
}
```

**`TileLangTQConfig` dataclass** (derived from a `VariantSpec` + `head_dim`):

| Property | Derivation |
|----------|-----------|
| `key_mse_bits` | `key_quant_bits - 1` if `key_use_qjl` else `key_quant_bits` |
| `value_mse_bits` | `value_quant_bits - 1` if `value_use_qjl` else `value_quant_bits` |
| `key_n_centroids` | `2 ** key_mse_bits` |
| `value_n_centroids` | `2 ** value_mse_bits` |
| `key_packed_bytes` | `ceil(head_dim * key_mse_bits / 8) + 2` (indices + norm fp16) |
| `key_qjl_packed_bytes` | `ceil(head_dim / 8) + 2` (sign bits + γ fp16) if `key_use_qjl` else `0` |
| `value_packed_bytes` | `ceil(head_dim * value_mse_bits / 8) + 2` (indices + norm fp16) |
| `value_qjl_packed_bytes` | `ceil(head_dim / 8) + 2` (sign bits + γ fp16) if `value_use_qjl` else `0` |
| `slot_size` | `key_packed_bytes + key_qjl_packed_bytes + value_packed_bytes + value_qjl_packed_bytes` |
| `slot_size_aligned` | next multiple of 16 bytes (CUDA alignment) |

**Extensibility:** Future variant `"tq_2bit"` requires only appending to `VARIANT_REGISTRY`. All downstream components read from the registry.

**Testing Plan (C1 registry/config portion):**
- Unit: All `VariantSpec` fields validate (no negative bits, `slot_size_aligned` divisible by 16).
- Unit: `TileLangTQConfig.slot_size` exact values for `head_dim ∈ {64, 96, 128, 256}` at both variants.
- Unit: Memory layout invariant: `slot_size_aligned × block_size × num_kv_heads == page_size_bytes`.
- Unit: `key_mse_bits` and `value_mse_bits` = 2 for 3-bit variant, = 3 for 4-bit variant.
- Unit: `key_n_centroids` and `value_n_centroids` = 4 for 3-bit (2-bit MSE base), `= 8` for 4-bit (3-bit MSE base).
- Regression: Adding a new `VariantSpec` to `VARIANT_REGISTRY` doesn't break existing variant configs.

---

#### C1.2 KV Cache Spec & Memory Layout

**Purpose:** Tell vLLM how many bytes to allocate per KV block (page) by overriding `real_page_size_bytes`. This is the contract between the variant's slot layout and vLLM's block allocator.

**Key Class:**

```python
@dataclass(frozen=True, kw_only=True)
class TileLangTQAttentionSpec(FullAttentionSpec):
    tq_slot_size: int = 0       # slot_size_aligned from TileLangTQConfig
    tq_variant_name: str = ""   # e.g. "tq_3bit"

    @property
    def real_page_size_bytes(self) -> int:
        if self.tq_slot_size > 0:
            # No "2×" factor: K+V packed together in one slot
            return self.block_size * self.num_kv_heads * self.tq_slot_size
        return super().real_page_size_bytes

    @classmethod
    def merge(cls, specs: list["TileLangTQAttentionSpec"]):
        # All layers must agree on slot_size and variant
        assert all(s.tq_slot_size == specs[0].tq_slot_size for s in specs)
        assert all(s.tq_variant_name == specs[0].tq_variant_name for s in specs)
        merged = super().merge(specs)
        return replace(merged, tq_slot_size=specs[0].tq_slot_size,
                       tq_variant_name=specs[0].tq_variant_name)
```

**Logical KV cache tensor shape** (returned by `TileLangTQAttentionBackend.get_kv_cache_shape`):

```
(num_blocks, block_size, num_kv_heads, slot_size_aligned)   dtype=int8
```

**Memory invariant that must hold:**

```
prod(get_kv_cache_shape(...)) == KVCacheTensor.size // dtype_size(int8)
⟺  num_blocks × block_size × num_kv_heads × slot_size_aligned
     == num_blocks × real_page_size_bytes
```

No stride-order override is needed (identity order: `(num_blocks, block_size, heads, slot)` is already contiguous and what the kernels expect).

**Memory savings vs. bfloat16 (head_dim=128, 8 heads, block_size=16):**

| Variant | `slot_size_aligned` | `real_page_size_bytes` | Blocks from 16 GB / 32 layers | Compression |
|---------|--------------------|-----------------------|-------------------------------|-------------|
| bfloat16 | — | 65,536 B | ~7,629 | 1× |
| `tq_3bit` | 112 B | 14,336 B | ~34,910 | ~4.6× |
| `tq_4bit` | 144 B | 18,432 B | ~27,174 | ~3.6× |

*(Exact `slot_size_aligned` is LLD-computed from `TileLangTQConfig`.)*

**Testing Plan (C1 memory/spec portion):**
- Unit: `real_page_size_bytes` formula matches manual calculation for all valid `(head_dim, variant)` combinations.
- Unit: `merge()` raises on mismatched `tq_slot_size` or variant names.
- Unit: `page_size_bytes == real_page_size_bytes` (no per-token-head scale mode needed; scales are inside the slot).
- Integration: `_reshape_kv_cache_tensors` succeeds without `RuntimeError` (view size invariant holds).
- Integration: Reshaped tensor has shape `(num_blocks, block_size, num_kv_heads, slot_size_aligned)`.
- Integration: Two layers with the same spec can share a `KVCacheTensor` (merge idempotent).

---

### C2: Offline Pre-computation

**Purpose:** Generate the rotation matrix `Π`, the Lloyd-Max centroid codebook `C`, and the QJL projection matrix `S`. These are computed once per server startup (not per request).

#### C2.1 Rotation Matrix

- Reference mode: construct a seeded random Gaussian matrix and take its QR decomposition, matching `turboquant.rotation.generate_rotation_matrix`.
- TileLang-optimized mode: construct a randomized Hadamard transform with random sign flips and an optional permutation. A plain deterministic Hadamard matrix is not sufficient unless tests show it preserves quality for target models.
- `Π` and `Π^T` are cached per `(head_dim, device, seed)`. If using a symmetric Hadamard-only approximation, `Π^T = Π`; otherwise store or pretranspose both layouts needed by kernels.
- QR mode supports arbitrary `head_dim`. Hadamard mode requires power-of-2 dimensions or a documented padding strategy.
- Stored as `float32` on GPU (register_buffer on Attention layer).

#### C2.2 Lloyd-Max Codebook

- The prior distribution of rotated coordinates is `f_X(x) = (Γ(d/2) / (√π · Γ((d-1)/2))) (1-x²)^((d-3)/2)`.
- Lloyd-Max iteration: alternate between updating boundaries (midpoints) and centroids (conditional means under `f_X`).
- Iterations run until centroid displacement converges (< 1e-7).
- Codebook is precomputed offline for each `(head_dim, b)` combination and can be persisted to disk.
- At startup: loaded from disk or recomputed if not cached. Registered as separate `_tq_key_centroids` and `_tq_value_centroids` buffers on each `Attention` layer; they may point to the same underlying codebook when key/value MSE bit-widths match.
- Shared across layers (one copy per `(head_dim, b)` combination, but `register_buffer` stamps each layer — see buffer inventory in C3).

#### C2.3 QJL Projection Matrix `S` (both variants)

- `S ∈ ℝ^{d×d}`, entries `S_{ij} ~ N(0,1)`, not normalized.
- Used as: `qjl = sign(S · r)` where `r` is the MSE residual.
- Dequant uses: `S^T · qjl` scaled by `(√(π/2) / d) · γ`.
- `S` can be stored sparsely (random sign matrix approximation) for memory efficiency — this is a LLD decision.
- Registered as `_tq_S_matrix` buffer on each `Attention` layer (both variants).

**Testing Plan (C2):**
- Unit (codebook): Lloyd-Max centroids for known low-dim distributions converge to published values.
- Unit (codebook): For `d=128`, `b=3`: verify 8 centroids are symmetric around 0 (distribution is symmetric).
- Unit (codebook): Quantization MSE decreases monotonically with more Lloyd-Max iterations.
- Unit (rotation): `Π @ Π.T ≈ I` (orthonormality check, tolerance 1e-5).
- Unit (rotation): QR mode matches the seeded reference matrix; randomized Hadamard mode passes round-trip and model-quality thresholds against QR mode.
- Unit (QJL): `sign(S · r)` has entries ∈ `{-1, +1}` for any non-zero `r`.
- Unit (QJL): DeQuant_prod is unbiased: `E[x_tilde_qjl · query] ≈ r · query` over many random `r`.
- Property test: For `x ∈ S^{d-1}`, round-trip `DeQuant_mse(Quant_mse(x))` has MSE matching published TurboQuant figures (within 5%).

---

### C3: Quantization Config & KV Cache Method

**Purpose:** Integrate with vLLM's quantization registry so that `quantization="tq_3bit"` (or `"tq_4bit"`) is a valid engine option. Handle checkpoint-derived metadata loading (future: per-layer scales, codebook overrides).

#### C3.1 `TileLangTQQuantizationConfig` (subclasses `QuantizationConfig`)

Key methods:

| Method | Responsibility |
|--------|---------------|
| `get_name()` | Returns variant string, e.g. `"tq_3bit"` |
| `get_supported_act_dtypes()` | `[torch.float16, torch.bfloat16]` |
| `get_quant_method(layer, prefix)` | Returns `TileLangTQKVCacheMethod(self)` if layer is `Attention` |
| `get_kv_cache_spec(layer, vllm_config)` | Returns `TileLangTQAttentionSpec` with correct `tq_slot_size` |
| `get_cache_scale(checkpoint_name)` | Maps `.k_proj.output_scale` → `.attn.k_scale` (future use) |

The `get_kv_cache_spec` hook is the bridge: it's called by `Attention.get_kv_cache_spec()` when this quant config is active, producing the correct spec type for memory allocation.

#### C3.2 `TileLangTQKVCacheMethod` (subclasses `BaseKVCacheMethod`)

Follows the dual-state lifecycle from vLLM (see deep-dive Part 3):

- **Phase 0 (Attention `__init__`):** `Attention._init_kv_cache_quant` calls `set_default_quant_scales(register_buffer=True)`, which pre-creates `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale` as float32 `register_buffer`s with default value `1.0`. These buffers exist **before** `create_weights` runs. `process_weights_after_loading` later overwrites them with checkpoint values if present.
- **Phase 1 (load time):** `create_weights` registers temporary `nn.Parameter` placeholders for checkpoint-derived scalars (`k_scale`, `v_scale`; sentinel `-1.0` = unset). For **non-scalar metadata** such as per-layer codebook tensors, create dedicated `register_buffer`s here (e.g. `_tq_centroids_load`) or register them in `create_weights` directly — they cannot be loaded via `get_cache_scale` because that hook only handles scalar scale mappings. Non-scalar tensors must be populated in the general `load_weights` loop.
- **Phase 2 (post-load pass 1):** `quant_method.process_weights_after_loading(layer)` runs on every module with a quant method. For custom 3/4-bit KV quantization, **implement a full override** — do not call `super()` naively because the base class has FP8-specific branches that assume scalar sentinels and may fail. This method: validates sentinels; copies loaded values to `_k_scale`/`_v_scale`; initializes key/value centroids from precomputed codebooks (C2); initializes `_tq_rotation`, `_tq_rotation_t`, `_tq_S_matrix`, decode workspace buffers; deletes all temp Parameters. Never delete `register_buffer`s.
- **Phase 2 (post-load pass 2):** After pass 1, vLLM also calls `Attention.process_weights_after_loading(act_dtype)` on every `Attention` module, which in turn calls `impl.process_weights_after_loading(act_dtype)` on the impl. This hook is available for kernel-specific post-load preparation (e.g. pre-transposing the rotation matrix for a specific tiling strategy).

**Critical timing:** All `register_buffer` calls must complete before the memory profile run. The full vLLM load order is: `__init__` (Phase 0) → `load_weights` → `process_weights_after_loading` pass 1 + pass 2 → `profile_run`.

**Buffer inventory per Attention layer:**

| Buffer | Dtype | Shape | Variant |
|--------|-------|-------|---------|
| `_tq_key_centroids` | float32 | `(key_n_centroids,)` | both |
| `_tq_key_midpoints` | float32 | `(key_n_centroids - 1,)` | both |
| `_tq_value_centroids` | float32 | `(value_n_centroids,)` | both |
| `_tq_value_midpoints` | float32 | `(value_n_centroids - 1,)` | both |
| `_tq_rotation` | float16/float32 | `(head_dim, head_dim)` | both |
| `_tq_rotation_t` | float16/float32 | `(head_dim, head_dim)` | both if rotation is not symmetric |
| `_tq_S_matrix` | float16/float32 | `(head_dim, head_dim)` | both |
| `_k_scale` | float32 | `(1,)` | both |
| `_v_scale` | float32 | `(1,)` | both |
| `_tq_mid_o_buf` | float16 | `(max_splits, max_batch, num_heads, head_dim)` | both |
| `_tq_lse_buf` | float32 | `(max_splits, max_batch, num_heads)` | both |
| `_tq_output_buf` | float16 | `(max_batch, num_heads, head_dim)` | both |

**Testing Plan (C3):**
- Unit: `get_quant_method` returns `TileLangTQKVCacheMethod` for `Attention` layers and `None` for linear layers.
- Unit: `get_kv_cache_spec` returns `TileLangTQAttentionSpec` with `tq_slot_size` matching `TileLangTQConfig.slot_size_aligned`.
- Unit: `create_weights` → `process_weights_after_loading` lifecycle: all temp Parameters deleted, all buffers present.
- Unit: If no checkpoint scales, `_k_scale` and `_v_scale` default to `1.0`.
- Integration: `@register_quantization_config("tq_3bit")` makes `get_quantization_config("tq_3bit")` resolve correctly.
- Integration: `process_weights_after_loading` runs before `profile_run` (verify via memory trace that buffers exist at peak).
- Integration: All layer buffers (`_tq_key_centroids`, `_tq_value_centroids`, `_tq_rotation`, `_tq_S_matrix`) are on GPU after `model.cuda()`.

---

### C4: Attention Backend & Metadata Builder

**Purpose:** Implement the `AttentionBackend` interface that vLLM uses to select and configure the TileLang TQ backend. Provide the `TileLangTQMetadataBuilder` that constructs per-step metadata from the scheduler output.

#### C4.1 `TileLangTQAttentionBackend` (subclasses `AttentionBackend`)

Key class-level decisions:

| Attribute | Value | Rationale |
|-----------|-------|-----------|
| `forward_includes_kv_cache_update` | `False` | Store runs via `unified_kv_cache_update`, separately from attention |
| `accept_output_buffer` | `True` | Output written into pre-allocated buffer for CUDA graph compat |
| `supported_kv_cache_dtypes` | `["tilelang_tq_3bit", "tilelang_tq_4bit"]` | Canonical steady-state dtype literals accepted by the backend after any branch-specific normalization |
| `get_supported_kernel_block_sizes()` | `[16, 32, 64, 128]` | Matches TileLang tile sizes; no virtual subdivision needed for 16-token blocks |

`get_kv_cache_shape` reads `TileLangTQConfig` from the active vllm config context:
```
(num_blocks, block_size, num_kv_heads, slot_size_aligned)
```
No stride-order override → identity permutation → contiguous int8 tensor.

`validate_configuration` is a `@classmethod` that returns a list of human-readable failure strings (empty = OK). `CudaPlatform.get_attn_backend_cls` calls it before accepting the backend; a non-empty list raises `ValueError` with **no fallback**. Checks include:
- `dtype in supported_dtypes` (fp16/bf16)
- normalized `kv_cache_dtype in supported_kv_cache_dtypes`
- `block_size` compatible with `get_supported_kernel_block_sizes()`
- `head_dim` is supported by the selected rotation mode (QR supports arbitrary dimensions; randomized Hadamard requires power-of-2 or padding)
- Variant name is in `VARIANT_REGISTRY`
- Compute capability is adequate
- Capability classmethods: `supports_attn_type` (decoder only), `is_mla` (False), `supports_per_head_quant_scales` (False), `supports_sink`, `supports_combination`

> **Important:** `AttentionBackendEnum.CUSTOM` is **not** in vLLM's auto-backend selection list. Users must explicitly pass `--attention-backend CUSTOM`. There is no automatic promotion to CUSTOM based on `kv_cache_dtype`, and older target branches may require plugin-owned admission/dispatch shims before backend selection.

#### C4.2 `TileLangTQMetadata` (dataclass, subclasses `AttentionMetadata`)

Fields matching what the decode and prefill kernels need:

| Field | Type | Description |
|-------|------|-------------|
| `seq_lens` | `Tensor[num_reqs]` | Total context length per request |
| `slot_mapping` | `Tensor[num_tokens]` | Cache slot index per new token |
| `block_table` | `Tensor[num_reqs, max_blocks]` | Physical block IDs for each request |
| `query_start_loc` | `Tensor[num_reqs + 1]` int32 GPU | Cumulative query token start indices (for varlen attention) |
| `query_start_loc_cpu` | Python list (CPU copy) | Host-side copy for per-request dispatch without GPU→CPU sync per request |
| `num_actual_tokens` | `int` | Attention-path token count; may be padded in CUDA-graph/full-padding modes, so store logic must not treat it as the authoritative row bound |
| `max_query_len` | `int` | Longest query in batch |
| `max_seq_len` | `int` | Longest context in batch |
| `is_prefill` | `bool` | Any prefill requests in batch |
| `num_decodes` | `int` | Number of decode-only requests |
| `num_decode_tokens` | `int` | Tokens from decode requests |

#### C4.3 `TileLangTQMetadataBuilder` (subclasses `AttentionMetadataBuilder`)

- Sets `reorder_batch_threshold=1` so the model runner places decode requests first in the batch. This ensures decode tokens occupy a contiguous prefix, enabling efficient kernel dispatch.
- `build()` calls `split_decodes_and_prefills` to find the decode/prefill boundary.
- `build_for_cudagraph_capture()` fills `seq_lens` with `1` so graph capture is cheap.
- CUDA graph support: `AttentionCGSupport.UNIFORM_BATCH`.

#### C4.4 `TileLangTQAttentionImpl.process_weights_after_loading` (optional)

`AttentionImpl` exposes an optional `process_weights_after_loading(act_dtype: torch.dtype)` hook. It is called during post-load pass 2 (after `quant_method.process_weights_after_loading`). This hook is the right place for impl-level kernel preparation that depends on both the loaded weights and the final activation dtype — for example, pre-transposing or re-packing the rotation matrix into the memory layout that the TileLang decode tile expects. The HLD treats this as optional; the LLD must decide whether any such prep is needed.

**Testing Plan (C4):**
- Unit: `get_kv_cache_shape` returns correct 4-tuple for all `(head_dim, variant)` combinations.
- Unit: `supports_kv_cache_dtype` returns `True` for both variant dtype strings, `False` for `"auto"`, `"fp8"`, etc. Any alias handling belongs to branch-specific normalization before backend selection.
- Unit: `validate_configuration` fails on non-power-of-2 `head_dim`.
- Unit: `validate_configuration` fails on unknown variant name.
- Unit: `MetadataBuilder.build` correctly splits batch: decodes first, prefills second.
- Unit: `build_for_cudagraph_capture` produces `seq_lens` of all `1`.
- Integration: Backend is resolved from `attention_backend="CUSTOM"` after registration.

---

### C5: Attention Implementation (Prefill/Decode Dispatch)

**Purpose:** Orchestrate the prefill/decode execution paths, dispatching to the appropriate TileLang kernels or FlashAttention fallback.

**Class:** `TileLangTQAttentionImpl(AttentionImpl[TileLangTQMetadata])`

#### C5.0 Execution Infrastructure: ForwardContext and Ordering

Before detailing the impl methods, it is critical to understand how inputs reach the impl during a compiled forward pass. They do **not** arrive as Python arguments passed through 32 layers of `torch.compile`-d code — that would require recompilation every step.

**`ForwardContext`** (`vllm/forward_context.py`) is a per-forward-pass thread-local stash. Before each model forward, the runner calls `set_forward_context(attn_metadata, slot_mapping, ...)`, which populates a dict keyed by layer name. Inside `Attention.forward`, the call chain is:

```
Attention.forward(query, key, value)
  │
  ├─ unified_kv_cache_update(key, value, encoded_layer_name)
  │    └─ get_attention_context(layer_name)
  │         → resolves kv_cache, slot_mapping for this layer from ForwardContext
  │    └─ impl.do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
  │    └─ returns kv_cache_dummy_dep  ← empty tensor, used as ordering token
  │
  └─ unified_attention_with_output(..., kv_cache_dummy_dep=dummy)
       └─ impl.forward(layer, query, key, value, kv_cache, attn_metadata, output)
```

**Dummy tensor (`kv_cache_dummy_dep`):** `unified_kv_cache_update` returns a zero-element tensor typed to `kv_cache.dtype`. It carries no data but creates a **data-dependency edge** in the `torch.compile` computation graph, guaranteeing the store kernel executes before the attention kernel. Without this edge, the compiler has no ordering guarantee between the two ops.

**`has_separate_kv_update` flag:** The model runner sets this to `True` when any attention group has `forward_includes_kv_cache_update = False`. When True, the runner uses the **padded** token count (`num_tokens_padded`) for slot mapping tensors so the height matches the activation tensors. The padding tail is filled with `-1` via `_get_slot_mapping(slot_mapping, num_tokens_unpadded, num_tokens_padded)`.

**KV sharing:** When `kv_sharing_target_layer_name is not None` (a layer borrows another layer's KV cache), the store step is **skipped entirely** — no `do_kv_cache_update` call, no dummy dep. The sharing layer reads directly from the target's cache. The TileLang TQ backend does not need to implement special sharing logic; it only needs to not break when the store is skipped.

#### C5.1 `do_kv_cache_update` (Store Path)

```
do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
  → calls tl_turboquant_store(key, value, kv_cache, slot_mapping,
                               layer._tq_rotation, layer._tq_rotation_t,
                               layer._tq_key_midpoints,
                               layer._tq_key_centroids,
                               layer._tq_value_midpoints,
                               layer._tq_value_centroids,
                               layer._tq_S_matrix,
                               config=self.tq_config)
```

- Uses `N = slot_mapping.shape[0]` as the authoritative row bound for the store path, matching vLLM's split-KV contract.
- Passes all necessary buffers from `layer`'s registered buffers.
- Does **not** assume `num_actual_tokens` is the non-padded store length. In CUDA-graph/full-padding paths, `key`/`value` and `slot_mapping` may be padded to the same height.
- Handles `slot_mapping[i] < 0` (padding / masked rows) inside the kernel (not Python-side).

#### C5.2 `forward` (Attention Path)

Three sub-paths:

```
forward(...)
  ├── Pure decode (is_prefill=False)
  │     └── _decode_attention(q, kv_cache, metadata, layer)
  │           └── tl_turboquant_decode_attention(...)
  │
  ├── Pure prefill (num_decodes=0)
  │     ├── First chunk (max_query_len == max_seq_len)
  │     │     └── flash_attn_varlen_func(q, k, v, ...)
  │     └── Continuation chunk
  │           ├── Small (q_len ≤ threshold) → tl_decode per-query (causal index trick)
  │           └── Large → _continuation_prefill (dequant cached + flash_attn)
  │
  └── Mixed batch (decodes first, guaranteed by reorder_batch)
        ├── Decode slice → _decode_attention
        └── Prefill slice → (sub-dispatch as above, with rebased metadata)
```

**Continuation threshold:** `_CONTINUATION_DECODE_THRESHOLD = 128` (matching existing behavior; tunable as LLD constant).

**`_continuation_prefill`:** Dequantizes cached K/V from TQ format into fp16, concatenates with current chunk's raw K/V, runs FlashAttention with causal mask.

**`_decode_attention`:** Passes pre-allocated `mid_o_buf`, `lse_buf`, `output_buf` from layer to the TileLang decode kernel to avoid per-step allocation.

**Device residency rule:** `centroids`, rotation matrices, `S_matrix`, and decode scratch must already be materialized on their target device **before** `profile_run()`. The impl may validate device placement or reuse existing buffers at runtime, but it must not perform first-use GPU allocations in `forward` or `do_kv_cache_update`, because those allocations would be invisible to vLLM's memory profiler and could over-commit KV capacity.

**Mixed-batch `max_seq_len` rebasing (critical correctness detail):** When processing the prefill slice of a mixed batch, `max_seq_len` in the sub-metadata **must** be computed from the prefill requests only — not the full-batch `max_seq_len`. FlashAttention's first-chunk fast path triggers when `max_query_len == max_seq_len` (meaning all K/V are in the current chunk). If the full-batch `max_seq_len` is used (inflated by long-context decode requests), this condition never triggers, causing a correctness regression where prefill requests that should use the fast path fall into the slower continuation path.

**Input tensor notes (from model layer):** By the time `do_kv_cache_update` is called, `key` has had **RoPE applied** (positional encoding) and `value` has **not** (values are content-only). Both are shaped `(num_tokens, num_kv_heads, head_dim)` — no implicit `.contiguous()` should be called unnecessarily in the hot path.

**Testing Plan (C5):**
- Unit: `do_kv_cache_update` with known inputs produces expected slot writes (verify via `kv_cache[block_idx, pos, head, :]`).
- Unit: `forward` with `attn_metadata=None` returns zero output (edge case).
- Unit: Pure decode path is triggered when `is_prefill=False`.
- Unit: Pure prefill path is triggered when `num_decodes=0` and `max_query_len == max_seq_len`.
- Unit: Continuation path uses decode kernel for `q_len ≤ 128` and dequant+flash for larger.
- Integration: Mixed batch produces correct attention output (compare vs. split decode+prefill reference).
- Integration: Round-trip store→decode on a single request matches plain SDPA within tolerance.
- Integration: CUDA graph capture succeeds with `build_for_cudagraph_capture` metadata.

---

### C6: TileLang Kernels (Store & Decode)

**Purpose:** Two fused GPU kernels that together implement the full TurboQuant runtime data path: `tl_turboquant_store` writes compressed K/V into the paged cache; `tl_turboquant_decode_attention` reads and dequantizes that cache to compute attention. Both kernels share the same slot layout contract (see §6) and are parameterized by the same `TileLangTQConfig`, which makes them natural to design, test, and version together.

---

#### C6.1 Store Kernel (`tl_store.py`)

**Entry point:**
```python
tl_turboquant_store(
    key:           Tensor[N, Hk, D],
    value:         Tensor[N, Hk, D],
    kv_cache:      Tensor[num_blocks, block_size, Hk, slot_size],  # int8
    slot_mapping:  Tensor[N],                                        # int64
    rotation:      Tensor[D, D],         # Π rotation matrix (float32)
    rotation_t:    Tensor[D, D],         # Π^T inverse rotation matrix (float32)
    key_midpoints: Tensor[n_key_c - 1],  # key Lloyd-Max decision boundaries (float32)
    key_centroids: Tensor[n_key_c],      # key Lloyd-Max centroids (float32)
    value_midpoints: Tensor[n_val_c - 1],# value Lloyd-Max decision boundaries (float32)
    value_centroids: Tensor[n_val_c],    # value Lloyd-Max centroids (float32)
    S_matrix:      Tensor[D, D],          # QJL matrix (float32), required for both variants
    config:        TileLangTQConfig,
)
```

**Kernel structure (per-tile logic):**

Each TileLang tile handles one `(token, head)` pair:

1. **Load:** Read `key[token, head, :]` (D fp16 elements) and `value[token, head, :]` (D fp16 elements).
2. **Check slot:** If `slot_mapping[token] < 0`, skip (dummy row).
3. **Key path — rotation:**
   - Cast key to fp32.
   - Compute `‖key‖₂` (norm scalar).
   - Normalize: `k_unit = key / ‖key‖₂`.
   - Matrix multiply: `y = Π @ k_unit` (D×D rotation in fp32).
4. **Key path — MSE quantization:**
   - For each element `y[j]`, binary search on `midpoints` to find centroid index `idx[j]`.
   - Pack `D` indices (each `key_mse_bits` wide) into `key_mse_bits × D / 8` bytes.
   - Store packed indices + norm (`float16`) into `kv_cache[block_idx, pos, head, 0:key_packed_bytes]`.
5. **Key path — QJL residual (both variants):**
   - Reconstruct `y_tilde[j] = centroids[idx[j]]`.
   - Inverse-rotate: `k_tilde_unit = Π^T @ y_tilde`.
   - Rescale the MSE approximation: `k_tilde = k_tilde_unit * ‖key‖₂`.
   - Residual: `r = key - k_tilde` (original vector scale, matching the reference repo).
   - Project: `s = S @ r` (D fp32 elements).
   - Sign-pack: `qjl = sign(s)` packed as 1 bit per element → `D/8` bytes.
   - Compute `γ = ‖r‖₂`.
   - Store: packed `qjl` + `γ` (float16) at slot offset `key_packed_bytes`.
6. **Value path — TurboQuant MSE base:**
   - Cast value to fp32.
   - Compute `‖value‖₂` (norm scalar).
   - Normalize: `v_unit = value / ‖value‖₂`.
   - Rotate: `z = Π @ v_unit`.
   - For each element `z[j]`, binary search on `value_midpoints` to find centroid index `v_idx[j]`.
   - Pack `D` indices (each `value_mse_bits` wide) into `value_mse_bits × D / 8` bytes.
   - Store packed value indices + value norm (`float16`) at the value MSE slot offset.
7. **Value path — QJL residual (both variants):**
   - Reconstruct `z_tilde[j] = value_centroids[v_idx[j]]`.
   - Inverse-rotate: `v_tilde_unit = Π^T @ z_tilde`.
   - Rescale the MSE approximation: `v_tilde = v_tilde_unit * ‖value‖₂`.
   - Residual: `r_v = value - v_tilde`.
   - Project: `s_v = S @ r_v`.
   - Sign-pack: `v_qjl = sign(s_v)` packed as 1 bit per element → `D/8` bytes.
   - Compute `γ_v = ‖r_v‖₂`.
   - Store packed `v_qjl` + `γ_v` at the value QJL slot offset.

**Scatter:** `block_idx = slot_mapping[token] // block_size`, `pos = slot_mapping[token] % block_size`. Write the assembled slot to `kv_cache[block_idx, pos, head, :]`.

**TileLang tiling strategy:** Tile over `(token, head)` pairs. Each tile owns one `(token, head)` slot. Parallelism: `num_tokens × num_kv_heads` tiles. The rotation matrix is loaded into shared memory once per tile block when the tile shape benefits from it.

---

#### C6.2 Decode Kernel (`tl_decode.py`)

**Entry point:**
```python
tl_turboquant_decode_attention(
    query:         Tensor[B, Hq, D],
    kv_cache:      Tensor[num_blocks, block_size, Hk, slot_size],  # int8
    block_table:   Tensor[B, max_blocks],
    seq_lens:      Tensor[B],
    rotation:      Tensor[D, D],         # Π rotation matrix (float32), for direct-score kernels
    rotation_t:    Tensor[D, D],         # Π^T inverse rotation matrix (float32), for dequant kernels
    key_centroids: Tensor[n_key_c],      # key Lloyd-Max centroids (float32)
    value_centroids: Tensor[n_val_c],    # value Lloyd-Max centroids (float32)
    scale:         float,                # softmax temperature = 1/√D
    config:        TileLangTQConfig,
    S_matrix:      Tensor[D, D],         # QJL projection (float32)
    mid_o_buf:     Tensor[max_splits, B, Hq, D],  # scratch
    lse_buf:       Tensor[max_splits, B, Hq],      # log-sum-exp scratch
    output_buf:    Tensor[B, Hq, D],               # final output
    max_num_kv_splits: int,
) → Tensor[B, Hq, D]
```

**Kernel structure (KV-split / flash-decoding style):**

Split the KV context across `NUM_KV_SPLITS` partial tiles. Each split tile computes partial attention over a subset of KV blocks.

Per split tile — loads KV blocks for its assigned range:
1. **Load KV slot:** `kv_cache[block_id, pos, head, :]` (int8 bytes).
2. **Dequant key:**
   - Extract packed indices from `[0:key_mse_bytes]`.
   - Gather centroids: `y_tilde[j] = key_centroids[idx[j]]`.
   - Inverse-rotate: `k_tilde_unit = Π^T @ y_tilde`.
   - Extract stored norm: fp16 at offset `key_mse_bytes`.
   - Scale: `k = k_tilde_unit * norm`.
   - **QJL residual (both variants):**
     - Extract `qjl` bits from slot.
     - Extract `γ` (fp16).
     - `k_residual = (√(π/2) / D) · γ · S^T · qjl`.
     - `k = k + k_residual`.
3. **Dequant value:**
   - Extract packed value MSE indices.
   - Gather centroids: `z_tilde[j] = value_centroids[v_idx[j]]`.
   - Inverse-rotate: `v_tilde_unit = Π^T @ z_tilde`.
   - Extract stored value norm and rescale: `v = v_tilde_unit * value_norm`.
   - Extract value QJL bits and `γ_v`.
   - `v_residual = (√(π/2) / D) · γ_v · S^T · v_qjl`.
   - `v = v + v_residual`.
4. **Attention score:** `score = scale × (query[b, h_q, :] · k)` (dot product after GQA expansion).
5. **Partial softmax:** Online softmax accumulation over KV blocks in this split.
6. **Partial output:** `partial_out += score_normalized × v`.

**Direct-score optimization:** The mathematically equivalent repo path avoids materializing `k` by rotating/sketching the query once per decode step: `q_rot = Π · q` for the MSE term and `q_sketched = S · q` for the QJL term. TileLang kernels may use either direct-score or dequant-then-dot form, but direct-score should be preferred if it reduces register pressure and memory traffic.

After all splits: merge partial results using log-sum-exp reduction into `output_buf` (flash-decoding merge step). The merge can reuse the existing `triton_merge_attn_states` or be fused into the TileLang kernel.

**TileLang tiling strategy:**
- Outer tile: `(batch_request, query_head)`.
- Inner tile: KV blocks (iterate over assigned split range).
- Block-level parallelism: `B × Hq × NUM_KV_SPLITS`.
- GQA: `Hq / Hk` query heads share one KV head.

---

#### C6.3 Shared Slot Layout Descriptor

Both kernels import a `SlotLayout` dataclass generated from `TileLangTQConfig` that encodes every byte offset as named constants. This is the single source of truth for the slot layout — neither kernel hardcodes offsets independently.

```python
@dataclass(frozen=True)
class SlotLayout:
    key_mse_offset: int      # 0
    key_norm_offset: int     # key_mse_bytes
    key_qjl_offset: int      # key_packed_bytes (0 if no key QJL)
    key_qjl_gamma_offset: int # key_qjl_offset + key_qjl_bits_bytes
    value_mse_offset: int    # start of value packed MSE indices
    value_norm_offset: int   # value_mse_offset + value_mse_bytes
    value_qjl_offset: int    # value_mse_offset + value_packed_bytes
    value_qjl_gamma_offset: int # value_qjl_offset + value_qjl_bits_bytes
    slot_size_aligned: int   # total slot bytes
```

---

#### C6.4 Testing Plan (C6)

**Store kernel (`tl_store`):**
- Unit: Store to slot index 0, read back raw int8 bytes, verify expected packed encoding.
- Unit: `slot_mapping[i] = -1` → corresponding slot in `kv_cache` is NOT modified.
- Unit: For a known key vector, verify stored MSE indices match Python reference (Lloyd-Max argmin).
- Unit: Verify packed norm `‖key‖₂` is stored correctly at `SlotLayout.key_norm_offset`.
- Unit: Stored `qjl` bits match `sign(S @ r)` from Python reference for both variants.
- Unit: Stored `γ` (float16) is within fp16 rounding of `‖r‖₂` for both variants.
- Unit: Value TurboQuant packing correctness: MSE indices, norm, QJL bits, and `γ_v` match Python reference within fp16 rounding.
- Edge case: `head_dim = 64`, `head_dim = 256` (non-standard sizes).
- Stress: Batch of 512 tokens × 32 heads stores without race conditions (verify with `torch.cuda.synchronize` + reread).

**Decode kernel (`tl_decode`):**
- Unit: Dequant a single slot (written by store test above) and verify key/value match reference (within fp16 precision).
- Unit: Decode attention with `seq_len=1` (single token in cache) matches `Q @ K^T × scale × V` manually.
- Unit: GQA decode correct: 8 query heads with 1 KV head gives all 8 query heads the same attention distribution.
- Unit: `seq_lens` masking: tokens beyond `seq_lens[b]` contribute 0 to the output.
- Integration: `KV-splits > 1` produces same result as `KV-splits = 1` (merge correctness).
- Integration: CUDA graph capture and replay produces bit-identical output.
- Edge case: `seq_len = block_size` (exactly one block), `seq_len = block_size + 1` (straddles two blocks).
- Stress: Batch of 256 decode requests with context lengths uniformly in [1, 8192] — no OOB access.

**Joint store→decode round-trip:**
- Numerical (3-bit): Store 1000 random unit K vectors, decode with 1 query → mean MSE ≤ published TurboQuant figures for same `(head_dim, b=3)`.
- Numerical (4-bit): Same, MSE must be lower than 3-bit result.
- Integration (3-bit): Store 100 tokens, decode with 1 query → output within 5% MSE of bfloat16 FlashAttention reference.
- Integration (4-bit): Same as above, MSE should be lower than 3-bit.

---

### C7: Plugin Registration & Entry Point

**Purpose:** Ensure that in **every vLLM process** (API server, engine core, GPU workers), the TileLang TQ backend, quantization configs, and any required branch-compatibility shims are registered before any attention layer is constructed.

**`pyproject.toml`:**
```toml
[project.entry-points."vllm.general_plugins"]
tilelang_turboquant = "tilelang_turboquant.plugin:register_all"
```

**`plugin.py`:**
```python
def register_all() -> None:
    # 1. Register quantization configs (triggers @register_quantization_config)
    from tilelang_turboquant.quantization.quant_config import (
        TileLangTQ3BitConfig,
        TileLangTQ4BitConfig,
    )
    # Decorator already ran at import time — this import is sufficient.

    # 2. Apply branch adapters for cache-dtype admission / KV-spec dispatch
    from tilelang_turboquant.compat import apply_branch_adapters
    apply_branch_adapters()

    # 3. Register attention backend
    from vllm.v1.attention.backends.registry import (
        register_backend, AttentionBackendEnum,
    )
    register_backend(
        AttentionBackendEnum.CUSTOM,
        "tilelang_turboquant.backend.backend.TileLangTQAttentionBackend",
    )
```

`register_all` is idempotent (calling it twice is safe due to vLLM's guard in `load_general_plugins`).

**User-facing invocation:**
```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="tq_3bit",          # triggers TileLangTQ3BitConfig
    kv_cache_dtype="tilelang_tq_3bit",
    attention_backend="CUSTOM",
    gpu_memory_utilization=0.9,
)
```

**Testing Plan (C7):**
- Unit: `register_all()` is idempotent (call twice, no exception, same registry state).
- Integration: After `register_all()`, `get_quantization_config("tq_3bit")` resolves to `TileLangTQ3BitConfig`.
- Integration: After `register_all()`, `AttentionBackendEnum.CUSTOM.get_path()` returns the correct class string.
- Integration (multi-process): Simulate worker subprocess: import vllm fresh, run `load_general_plugins()`, verify registry populated correctly.
- Integration: `LLM(quantization="tq_3bit", ...)` initializes without errors on a real GPU (smoke test).
- Integration: `LLM(quantization="tq_4bit", ...)` initializes without errors on a real GPU (smoke test).
- Negative: Without plugin installed, `quantization="tq_3bit"` raises `ValueError`.

---

## 6. Slot Layout Reference

This section defines the exact byte layout within each `slot_size_aligned`-byte slot. The store kernel writes this layout; the decode kernel reads it. Both must agree exactly.

### 6.1 3-bit Variant (`tq_3bit`, K/V `quant_bits=3` → 2-bit MSE + 1-bit QJL)

For `head_dim = D`:

```
Offset  | Size (bytes)          | Content
--------|-----------------------|------------------------------------------------
0       | ceil(D×2/8)           | Packed 2-bit MSE indices for KEY  (D indices)
+       | 2                     | KEY vector norm ‖k‖₂  (float16)
+       | ceil(D/8)             | Packed QJL sign bits for KEY residual (D bits)
+       | 2                     | KEY QJL residual norm γ_k  (float16)
+       | ceil(D×2/8)           | Packed 2-bit MSE indices for VALUE (D indices)
+       | 2                     | VALUE vector norm ‖v‖₂  (float16)
+       | ceil(D/8)             | Packed QJL sign bits for VALUE residual (D bits)
+       | 2                     | VALUE QJL residual norm γ_v  (float16)
+       | padding               | Zero-fill to slot_size_aligned
```

Example for `D=128`:
- `key_mse_bytes = ceil(128×2/8) = 32`
- `key_norm_bytes = 2`
- `key_qjl_bits_bytes = ceil(128/8) = 16`, `key_qjl_gamma = 2`
- `value_mse_bytes = 32`, `value_norm_bytes = 2`
- `value_qjl_bits_bytes = 16`, `value_qjl_gamma = 2`
- `slot_size = 32+2+16+2+32+2+16+2 = 104` → `slot_size_aligned = 112` (next multiple of 16)

### 6.2 4-bit Variant (`tq_4bit`, K/V `quant_bits=4` → 3-bit MSE + 1-bit QJL)

```
Offset  | Size (bytes)          | Content
--------|-----------------------|------------------------------------------------
0       | ceil(D×3/8)           | Packed 3-bit MSE indices for KEY  (D indices)
+       | 2                     | KEY vector norm ‖k‖₂  (float16)
+       | ceil(D/8)             | Packed QJL sign bits for KEY residual (D bits)
+       | 2                     | KEY QJL residual norm γ_k  (float16)
+       | ceil(D×3/8)           | Packed 3-bit MSE indices for VALUE (D indices)
+       | 2                     | VALUE vector norm ‖v‖₂  (float16)
+       | ceil(D/8)             | Packed QJL sign bits for VALUE residual (D bits)
+       | 2                     | VALUE QJL residual norm γ_v  (float16)
+       | padding               | Zero-fill to slot_size_aligned
```

Example for `D=128`:
- `key_mse_bytes = 48`, `key_norm = 2`
- `key_qjl_bits_bytes = ceil(128/8) = 16`, `key_qjl_gamma = 2`
- `value_mse_bytes = 48`, `value_norm_bytes = 2`
- `value_qjl_bits_bytes = 16`, `value_qjl_gamma = 2`
- `slot_size = 48+2+16+2+48+2+16+2 = 136` → `slot_size_aligned = 144` (next multiple of 16)

### 6.3 Layout is an LLD Contract

The exact byte offsets must match between `tl_store.py` and `tl_decode.py`. Both components should import a shared `SlotLayout` descriptor generated from `TileLangTQConfig` to avoid offset bugs.

---

## 7. Global Buffer Lifecycle

The following diagram shows when global buffers (rotation matrix, centroids, decode workspace) are created and consumed relative to the vLLM memory profiling run.

```
Process start
     │
     ▼
load_general_plugins()      ← plugin registers backend + quant config
     │
     ▼
LLM.__init__() / AsyncLLMEngine()
     │
     ▼
model.__init__()
  └► For each Attention layer:
       quant_method.create_weights(layer)
         → layer.k_scale = Parameter(-1.0)   ← temp load-time slots
     │
     ▼
load_weights(checkpoint)
  └► layer.k_scale.copy_(loaded_value)        ← checkpoint fills temp params
     │
     ▼
process_weights_after_loading(model)
  └► For each Attention layer:
       quant_method.process_weights_after_loading(layer)
         → layer._tq_key_centroids   = register_buffer(precomputed)
         → layer._tq_key_midpoints   = register_buffer(precomputed)
         → layer._tq_value_centroids = register_buffer(precomputed)
         → layer._tq_value_midpoints = register_buffer(precomputed)
         → layer._tq_rotation   = register_buffer(Π rotation matrix)
         → layer._tq_rotation_t = register_buffer(Π^T inverse rotation matrix, if needed)
         → layer._tq_S_matrix   = register_buffer(QJL matrix)
         → layer._tq_mid_o_buf  = register_buffer(zeros, decode scratch)
         → layer._tq_lse_buf    = register_buffer(zeros, decode scratch)
         → layer._tq_output_buf = register_buffer(zeros, decode scratch)
         → del layer.k_scale, layer.v_scale  ← clean up temp params
     │
     ▼
━━━━ profile_run() ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     (peak GPU memory includes all register_buffer'd tensors above)
     available_memory = total - peak
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
     │
     ▼
_allocate_kv_cache_tensors()    ← only uses available_memory (accurate)
     │
     ▼
bind_kv_cache()                 ← layer.kv_cache = shaped tensor
     │
     ▼
Per-step forward()              ← all buffers ready, no first-use GPU alloc
```

**Key invariant:** No inference-critical buffer creation or first-use device materialization after `profile_run()`. If a tensor is needed by store or decode on GPU, it must already exist on that device before the profile run.

---

## 8. Extensibility Design

Adding a new variant (e.g., `tq_2bit`) requires changes only to:

1. **`variant_registry.py`:** Add one `VariantSpec` entry with `key_quant_bits=2`, `value_quant_bits=2`, `key_use_qjl=False` / `value_use_qjl=False` (pure MSE) or `True` (1-bit MSE + 1-bit QJL), etc.
2. **`quant_config.py`:** Add `@register_quantization_config("tq_2bit")` class (copy-paste of 4-bit config, change variant name).
3. **Codebook:** Pre-generate centroids for the chosen key/value MSE bit-depths (`n_centroids=4` for 2-bit MSE, `n_centroids=2` for 1-bit MSE). Key and value codebooks can share storage when their MSE bit-depths match.

The following components require **no changes** for a new variant:
- `kv_spec.py` — reads `tq_slot_size` from config
- `backend.py` — only needs the new canonical dtype literal admitted through the same normalized path
- `impl.py` — reads `config.key_use_qjl` to branch
- `tl_store.py` (C6.1) — controlled by `config` argument passed at call time
- `tl_decode.py` (C6.2) — same; `SlotLayout` descriptor auto-derives new offsets from config

**Design principle:** The only source of truth is `TileLangTQConfig` (derived from `VariantSpec`). Kernels are parameterized by config, not hardcoded for specific bit-widths. TileLang's `constexpr` propagation handles compile-time specialization per bit-width.

---

## 9. Testing Plans

### 9.1 Test Taxonomy

| Level | When | What |
|-------|------|------|
| Unit | Per-function, CPU-only where possible | Correctness of math, shape contracts, registry state |
| Functional | GPU required | Kernel correctness, round-trip accuracy, attention output quality |
| Integration | Full vLLM stack | Allocation, binding, forward pass, CUDA graph |
| Regression | CI on every PR | Perplexity delta, no performance cliff |

### 9.2 Cross-Component Tests

Beyond per-component tests listed in §5:

**Round-trip correctness (`tests/integration/test_roundtrip_accuracy.py`):**
- For each variant × `head_dim ∈ {64, 96, 128, 256}`:
  - Generate 512 random unit vectors for K, 512 random vectors for V.
  - Store all via `do_kv_cache_update` into a TQ cache.
  - Decode with identity query (`Q=K[0]`) and verify attention output.
  - MSE vs. bfloat16 reference ≤ published TurboQuant bounds for same `(d, b)`.

**Perplexity regression (`tests/integration/test_roundtrip_accuracy.py`):**
- Run Llama-3-8B on WikiText-2 (128 samples, max_len=512).
- Compare perplexity: `tq_3bit` ≤ bfloat16 + 25%, `tq_4bit` ≤ bfloat16 + 15%.
- This is a regression gate, not a quality benchmark.

**KV cache allocation integrity (`tests/integration/test_kv_cache_allocation.py`):**
- Verify `num_blocks × slot_size_aligned × block_size × num_kv_heads == allocated_bytes`.
- Verify no double-allocation: two layers sharing a `KVCacheTensor` point to the same storage.
- Verify `TileLangTQAttentionSpec.merge()` is idempotent.

**End-to-end vLLM (`tests/integration/test_vllm_end_to_end.py`):**
- Greedy decode of a simple prompt with both variants.
- Output tokens are non-empty and non-garbage.
- Server does not OOM at 90% GPU utilization.
- Chunked prefill works (multi-step prompt encoding).

### 9.3 CI Strategy

- Unit and functional tests run on every PR (requires 1× A10G or equivalent).
- Perplexity regression runs nightly.
- Benchmark suite runs weekly (avoids CI time pressure).

---

## 10. Benchmark Plan

### 10.1 Kernel Benchmarks

**File:** `tests/benchmarks/bench_store_kernel.py` and `bench_decode_kernel.py` (both benchmark C6)

#### Store Kernel (`tl_store`)

| Sweep | Values |
|-------|--------|
| `batch_size` (new tokens) | 1, 8, 32, 128, 512, 2048 |
| `num_kv_heads` | 8, 16, 32 |
| `head_dim` | 64, 128, 256 |
| `variant` | `tq_3bit`, `tq_4bit` |

**Metrics per configuration:**
- Throughput: tokens/second
- Memory bandwidth utilization: GB/s (compare to device peak)
- Kernel latency: µs (p50, p99)
- Compare against: existing Triton `triton_turboquant_store`

#### Decode Kernel (`tl_decode`)

| Sweep | Values |
|-------|--------|
| `batch_size` | 1, 4, 16, 64, 256 |
| `context_lengths` | 128, 512, 2K, 8K, 32K |
| `num_kv_heads` | 8, 16, 32 |
| `head_dim` | 64, 128, 256 |
| `num_kv_splits` | 1, 2, 4, 8, 16 |
| `variant` | `tq_3bit`, `tq_4bit` |

**Metrics per configuration:**
- Attention latency: µs (p50, p99)
- Throughput: tokens generated/second
- Arithmetic intensity vs. roofline
- Compare against: existing Triton `triton_turboquant_decode_attention`, FlashAttention (bfloat16 baseline)

#### Baseline comparisons:

| Baseline | Purpose |
|----------|---------|
| `triton_turboquant_store` | Direct TileLang vs. Triton comparison |
| `triton_turboquant_decode_attention` | Direct TileLang vs. Triton comparison |
| FlashAttention bfloat16 | Quality/speed tradeoff reference |
| FP8 KV cache (if available) | Alternative compression baseline |

### 10.2 Model-Level Benchmarks

**File:** `tests/benchmarks/bench_model_throughput.py`

**Models:**
- Llama-3-8B (32 layers, 8 KV heads, head_dim=128)
- Llama-3-70B (80 layers, 8 KV heads, head_dim=128) — if GPU budget allows
- A model with head_dim=64 (e.g., Mistral-7B-v0.3 style)
- A model with GQA ratio ≥ 8 (to stress GQA decode path)

**Scenarios:**

| Scenario | Batch | Prompt len | Decode len | Purpose |
|----------|-------|-----------|------------|---------|
| Decode-heavy | 64 | 512 | 256 | Primary serving workload |
| Prefill-heavy | 8 | 8192 | 32 | Document processing |
| Long context | 4 | 32768 | 128 | RAG / long document |
| Mixed | 32 | variable | variable | Realistic serving simulation |
| Max throughput | sweep | — | — | Find optimal batch for each variant |

**Metrics:**
- Throughput: tokens/second (total = prompt + generated)
- Time to first token (TTFT): ms
- Time per output token (TPOT): ms p50/p95/p99
- GPU memory used: GB (compare to bfloat16 to confirm compression ratio)
- Effective context capacity: max tokens before OOM

**Hardware:**
- Primary: NVIDIA A100 80GB SXM4 (or H100)
- Secondary: NVIDIA A10G 24GB (realistic deployment target)

**Reporting format:** All benchmarks produce CSV + summary table. Regression threshold: ≤ 5% latency increase for decode, ≤ 10% for prefill, vs. Triton baseline.

---

## 11. Open Questions / LLD Decisions

The following items are explicitly deferred to the LLD phase. The LLD author will have access to the full vLLM source and these deep-dive documents.

| # | Question | LLD Component |
|---|----------|--------------|
| 1 | Exact padding strategy for `slot_size_aligned`: multiple of 8, 16, or 32 bytes? Affects alignment for TileLang memory access patterns. | C1 |
| 2 | Should `S_matrix` (QJL projection) be a structured random matrix (e.g., random sign × Hadamard) for faster `S^T · qjl` computation? | C2, C6 |
| 3 | Codebook persistence: load from a bundled `.npz` file, generate at startup, or derive from a closed-form approximation? Cache invalidation policy. | C2, C3 |
| 4 | `max_num_kv_splits` optimal value for CUDA graph: should it be static (fixed at startup) or adapt based on `max_seq_len`? | C5, C6 |
| 5 | `_continuation_prefill`: threshold of 128 tokens — validate or tune based on TileLang kernel profiling. | C5 |
| 6 | For `S_matrix`: store per-layer or share across all layers? Sharing saves memory but loses per-layer independence. | C3 |
| 7 | Rotation mode: use QR to match the reference repo exactly, or randomized Hadamard for kernel speed after quality validation? | C2, C6 |
| 8 | GQA expand: should the decode kernel expand KV heads to match query heads inside the kernel tile, or pre-expand in Python before calling the kernel? | C6.2 |
| 9 | Value TurboQuant validation: measure quality/latency against the reference repo's group-quantized value baseline, since this HLD intentionally stores value QJL metadata. | C1, C6 |
| 10 | Sliding window attention support: `TileLangTQAttentionSpec` should define how to handle `sliding_window` models (skip cache writes beyond window). | C1, C5 |
| 11 | Should boundary-skip layers (first/last N transformer layers) get a separate full-precision spec automatically? If so, how does the spec grouping work? | C1, C4 |
| 12 | If randomized Hadamard is selected, how should non-power-of-2 `head_dim` values (e.g., `head_dim=96`) be handled: zero-pad before rotation or fall back to QR mode? | C2, C6.1 |
