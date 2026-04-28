# LLD: C3 — Quantization Config & KV Cache Method

**Document type:** LLD (Low-Level Design)  
**Component:** C3 — `quantization/quant_config.py` + `quantization/kv_cache_method.py`  
**HLD reference:** §5 C4, §7 Global Buffer Lifecycle  
**Depends on:** C1 (`TileLangTQConfig`, `TileLangTQAttentionSpec`), C2 (offline artifacts)  
**Status:** Canonical plugin-owned integration design; valid for `vLLM` trees older than the current local tree

---

## 1. Purpose & Scope

C3 is the integration layer between:

- the plugin’s variant/config model from C1
- the plugin-local KV spec from C1
- the offline artifact builders from C2
- `vLLM`’s quantization and weight-loading lifecycle

It owns:

- quantization config registration for `quantization="tq_3bit"` / `quantization="tq_4bit"`
- branch-specific admission of plugin KV-cache dtype literals
- construction of the per-layer KV cache method
- post-load registration of all runtime buffers before `profile_run()`

It does not own:

- slot arithmetic
- page-size math
- offline artifact generation math
- attention backend execution

---

## 2. Two Separate Identifiers

The design uses two distinct identifier families:

| Concern | Canonical identifier |
|---|---|
| quantization registry key | `tq_3bit`, `tq_4bit` |
| KV-cache dtype literal | `tilelang_tq_3bit`, `tilelang_tq_4bit` |

These must not be conflated.

Rules:

- `quantization="tq_3bit"` selects the plugin quantization config class.
- `kv_cache_dtype="tilelang_tq_3bit"` selects the plugin KV slot format.
- the quantization config and KV-cache dtype must agree on the variant
- `auto` is not a valid steady-state cache-dtype choice for this plugin

Optional user-facing aliases such as `kv_cache_dtype="tq_3bit"` may be accepted by the plugin entrypoint, but they must be normalized to the canonical plugin-owned dtype literal before layer construction.

Canonical normalization:

```python
CACHE_DTYPE_ALIASES = {
    "tilelang_tq_3bit": "tilelang_tq_3bit",
    "tilelang_tq_4bit": "tilelang_tq_4bit",
    "tq_3bit": "tilelang_tq_3bit",   # optional user alias
    "tq_4bit": "tilelang_tq_4bit",   # optional user alias
}

VARIANT_BY_CACHE_DTYPE = {
    "tilelang_tq_3bit": "tq_3bit",
    "tilelang_tq_4bit": "tq_4bit",
}
```

---

## 3. Compatibility Surface Across `vLLM` Versions

Older target branches will not expose exactly the same hooks. So C3 is defined in terms of required capabilities, not one single upstream implementation shape.

### 3.1 Required capabilities

The adapter must provide all of the following:

| Capability | Preferred hook | Acceptable fallback |
|---|---|---|
| register quantization config classes | `@register_quantization_config(...)` | direct registry insertion on older branch |
| admit plugin KV-cache dtype literals | branch cache-config / dtype-map hook | monkey patch validator / mapper |
| map plugin cache dtype to 1-byte torch dtype | built-in dtype-map extension | monkey patch mapping helper |
| choose plugin-local KV spec | `Attention.get_kv_cache_spec` override path | monkey patch `Attention.get_kv_cache_spec` |
| create temporary scale parameters | `BaseKVCacheMethod.create_weights` | inline equivalent implementation |
| register permanent runtime buffers before profiling | `process_weights_after_loading(layer)` | earliest branch-specific post-load hook before memory profiling |

### 3.2 Non-negotiable runtime invariants

Regardless of branch shape:

- all required C2 artifacts must be registered on-device before `profile_run()`
- no lazy `_ensure_on_device` path is allowed
- no first-use GPU allocation is allowed in `forward` or `do_kv_cache_update`
- the KV spec selected for a TQ layer must be the plugin-local C1 spec, not the default `FullAttentionSpec`

---

## 4. `TileLangTQQuantizationConfig`

**File:** `quantization/quant_config.py`

The canonical structure is one shared base class plus two concrete registry classes.

```python
class TileLangTQQuantizationConfig(QuantizationConfig):
    variant_name: str = ""

    @classmethod
    def get_name(cls) -> str:
        return cls.variant_name

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 80

    @staticmethod
    def get_config_filenames() -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "TileLangTQQuantizationConfig":
        return cls()
```

Concrete registry classes:

```python
@register_quantization_config("tq_3bit")
class TileLangTQ3BitConfig(TileLangTQQuantizationConfig):
    variant_name = "tq_3bit"


@register_quantization_config("tq_4bit")
class TileLangTQ4BitConfig(TileLangTQQuantizationConfig):
    variant_name = "tq_4bit"
```

Rationale:

- the variant is selected by `quantization=...`, not by a JSON sidecar file
- separate concrete classes keep registry behavior simple on older branches
- future variants remain a copy-on-register pattern, not a special-case code path

### 4.1 `get_quant_method`

The quantization config returns a KV-cache method only for supported attention layers.

Canonical rule:

```python
def get_quant_method(
    self, layer: torch.nn.Module, prefix: str
) -> QuantizeMethodBase | None:
    if isinstance(layer, Attention):
        return TileLangTQKVCacheMethod(self)
    return None
```

If a target branch uses additional attention layer classes that own KV cache, the adapter may extend this check. But the default contract is decoder attention only.

### 4.2 `get_cache_scale`

This hook is only for checkpoint-derived scalar scales.

```python
def get_cache_scale(self, name: str) -> str | None:
    if name.endswith(".output_scale") and ".k_proj" in name:
        return name.replace(".k_proj.output_scale", ".attn.k_scale")
    if name.endswith(".output_scale") and ".v_proj" in name:
        return name.replace(".v_proj.output_scale", ".attn.v_scale")
    return None
```

Rules:

- only scalar `k_scale` / `v_scale` may flow through this hook
- non-scalar artifacts such as codebooks, rotations, or `S` must not use `get_cache_scale`
- future checkpoint overrides for non-scalar artifacts require explicit buffer-loading logic, not scale remapping

### 4.3 `get_kv_cache_spec` helper

Even if the base `QuantizationConfig` class on a given branch does not define this method, the plugin config class should.

```python
def get_kv_cache_spec(
    self, layer: torch.nn.Module, vllm_config: "VllmConfig"
) -> "KVCacheSpec | None":
    if not isinstance(layer, Attention):
        return None

    normalized = normalize_cache_dtype(layer.kv_cache_dtype)
    expected_variant = VARIANT_BY_CACHE_DTYPE[normalized]
    if expected_variant != self.variant_name:
        raise ValueError(
            f"Quantization variant {self.variant_name} is incompatible with "
            f"kv_cache_dtype={layer.kv_cache_dtype}"
        )

    cfg = TileLangTQConfig.from_variant_name(self.variant_name, layer.head_size)
    return TileLangTQAttentionSpec(
        block_size=vllm_config.cache_config.block_size,
        num_kv_heads=layer.num_kv_heads,
        head_size=layer.head_size,
        head_size_v=layer.head_size,
        dtype=layer.kv_cache_torch_dtype,
        tq_slot_size=cfg.slot_size_aligned,
        tq_variant_name=self.variant_name,
    )
```

This helper is the canonical place where the variant/slot config and the plugin-local KV spec meet.

---

## 5. KV-Cache Dtype Admission Layer

This is the part the old draft was too optimistic about.

Because the plugin keeps canonical plugin-owned literals:

- `tilelang_tq_3bit`
- `tilelang_tq_4bit`

older `vLLM` branches will usually reject them unless the plugin installs a compatibility shim.

### 5.1 What the shim must accomplish

The adapter must ensure that the normalized plugin literal:

1. survives config parsing
2. is treated as a valid KV-cache dtype during backend selection
3. maps to a 1-byte torch dtype for raw cache tensor allocation
4. reaches `Attention.get_kv_cache_spec()` unchanged so the plugin can choose the right C1 spec

### 5.2 Current-tree patch points

On the current local tree, the relevant surfaces are:

- cache-dtype validation in [cache.py](/Users/sasmitdatta/Desktop/vllm/vllm/config/cache.py:18)
- dtype-string to torch-dtype mapping in [torch_utils.py](/Users/sasmitdatta/Desktop/vllm/vllm/utils/torch_utils.py:25)
- backend selector validation in [selector.py](/Users/sasmitdatta/Desktop/vllm/vllm/v1/attention/selector.py:64)

Older branches may place these checks elsewhere, but the required behavior is the same.

### 5.3 Torch dtype rule

The plugin cache tensor must be byte-addressable, matching the C1 contract.

Canonical rule:

- map the plugin cache dtype literal to a 1-byte torch dtype
- `torch.int8` is the preferred canonical mapping
- `torch.uint8` is acceptable if the target branch internally assumes unsigned byte cache tensors

This choice does not change the C1 page-size arithmetic as long as the element size remains 1 byte.

---

## 6. `TileLangTQKVCacheMethod`

**File:** `quantization/kv_cache_method.py`

This is the layer-scoped object that bridges checkpoint loading with runtime buffers.

### 6.1 `create_weights`

Preferred implementation:

- inherit from `BaseKVCacheMethod`
- reuse `create_weights(layer)` if it only creates the temporary scalar sentinels:
  - `q_scale`
  - `k_scale`
  - `v_scale`
  - `prob_scale`

Canonical temporary state:

```python
layer.q_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
layer.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
layer.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
layer.prob_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
```

If a branch’s base helper does something more exotic, the plugin should inline this exact sentinel allocation instead of inheriting accidental FP8 behavior.

### 6.2 `process_weights_after_loading`

This method must be fully overridden. Do not call `super().process_weights_after_loading(layer)`.

Why:

- upstream base logic is FP8-oriented
- it assumes built-in quantized-cache semantics
- it may mutate q/prob scales in ways irrelevant to this plugin
- it does not register the C2 artifacts or decode workspace

Canonical flow:

1. early-return if temp params were already deleted
2. validate variant agreement between quantization config and `layer.kv_cache_dtype`
3. copy checkpoint `k_scale` / `v_scale` into permanent `_k_scale` / `_v_scale`, defaulting to `1.0`
4. keep `_q_scale` and `_prob_scale` at `1.0`
5. fetch C2 artifacts on CPU
6. move them to the final device and register runtime buffers
7. preallocate decode scratch buffers with `persistent=False`
8. delete the temporary Parameters

### 6.3 Canonical registration sketch

```python
def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
    if not hasattr(layer, "k_scale"):
        return

    cfg = TileLangTQConfig.from_variant_name(
        self.quant_config.variant_name,
        layer.head_size,
    )
    _validate_variant_match(layer.kv_cache_dtype, self.quant_config.variant_name)

    k_scale = _extract_optional_scale(layer.k_scale, default=1.0)
    v_scale = _extract_optional_scale(layer.v_scale, default=1.0)
    layer._k_scale.copy_(k_scale)
    layer._v_scale.copy_(v_scale)
    layer._k_scale_float = float(k_scale)
    layer._v_scale_float = float(v_scale)

    device = layer._k_scale.device
    runtime_dtype = _select_runtime_matrix_dtype(layer)

    rot_spec = RotationSpec(
        head_dim=cfg.head_dim,
        mode=_rotation_mode_from_runtime_config(),
        seed=_rotation_seed_from_runtime_config(),
        allow_hadamard_padding=_allow_hadamard_padding(),
    )
    rotation, rotation_t, is_symmetric = get_rotation(rot_spec)

    key_centroids, key_midpoints = get_codebook(
        CodebookSpec(cfg.head_dim, cfg.key_mse_bits)
    )
    value_centroids, value_midpoints = get_codebook(
        CodebookSpec(cfg.head_dim, cfg.value_mse_bits)
    )
    s_matrix = get_s_matrix(QJLSpec(cfg.head_dim, seed=_qjl_seed()))

    layer.register_buffer(
        "_tq_key_centroids",
        key_centroids.to(device=device, dtype=torch.float32),
    )
    layer.register_buffer(
        "_tq_key_midpoints",
        key_midpoints.to(device=device, dtype=torch.float32),
    )
    layer.register_buffer(
        "_tq_value_centroids",
        value_centroids.to(device=device, dtype=torch.float32),
    )
    layer.register_buffer(
        "_tq_value_midpoints",
        value_midpoints.to(device=device, dtype=torch.float32),
    )
    layer.register_buffer(
        "_tq_rotation",
        rotation.to(device=device, dtype=runtime_dtype),
    )
    if not is_symmetric:
        layer.register_buffer(
            "_tq_rotation_t",
            rotation_t.to(device=device, dtype=runtime_dtype),
        )
    layer.register_buffer(
        "_tq_S_matrix",
        s_matrix.to(device=device, dtype=runtime_dtype),
    )

    _register_decode_scratch_buffers(layer, cfg, device)

    del layer.k_scale
    del layer.v_scale
    del layer.q_scale
    del layer.prob_scale
```

### 6.4 Scale extraction rule

TQ does not use FP8-style q/prob checkpoint scales.

So:

- if `k_scale` or `v_scale` temp params were never populated, use `1.0`
- q/prob temp params are ignored except for cleanup
- do not emit FP8-specific warnings in the TQ path

Recommended helper:

```python
def _extract_optional_scale(param: torch.nn.Parameter, default: float) -> float:
    val = float(param.item())
    if val < 0.0:
        return default
    if val == 0.0:
        raise ValueError("Loaded zero kv scale for TileLang TQ layer")
    return val
```

---

## 7. Buffer Inventory

After `process_weights_after_loading`, each supported attention layer must expose:

| Buffer | Persistent | Dtype | Shape |
|---|---|---|---|
| `_k_scale` | True | `float32` | `(1,)` |
| `_v_scale` | True | `float32` | `(1,)` |
| `_q_scale` | True | `float32` | `(1,)` |
| `_prob_scale` | True | `float32` | `(1,)` |
| `_tq_key_centroids` | True | `float32` | `(2^key_mse_bits,)` |
| `_tq_key_midpoints` | True | `float32` | `(2^key_mse_bits - 1,)` |
| `_tq_value_centroids` | True | `float32` | `(2^value_mse_bits,)` |
| `_tq_value_midpoints` | True | `float32` | `(2^value_mse_bits - 1,)` |
| `_tq_rotation` | True | validated runtime dtype | `(D, D)` |
| `_tq_rotation_t` | True | validated runtime dtype | `(D, D)` when nonsymmetric |
| `_tq_S_matrix` | True | validated runtime dtype | `(D, D)` |
| `_tq_mid_o_buf` | False | runtime workspace dtype | impl-defined |
| `_tq_lse_buf` | False | `float32` | impl-defined |
| `_tq_output_buf` | False | runtime workspace dtype | impl-defined |

Rules:

- key and value codebooks are logically separate even when they have the same bit-width
- scratch buffers must use `persistent=False`
- all persistent buffers must exist before `profile_run()`

---

## 8. Decode Scratch Buffer Registration

Scratch buffer sizing belongs to the decode path, but C3 owns the preallocation timing.

Canonical rule:

- allocate decode/store workspace in the same post-load phase that registers C2 artifacts
- size from runtime config visible on the current branch
- if a newer branch exposes a dedicated split-KV or CUDA-graph tuning knob, use it
- if an older branch lacks that knob, fall back to a plugin-owned default

The exact shapes are kernel-dependent, but the HLD invariant is not:

- these buffers must exist before `profile_run()`
- they must not first appear inside `forward`

---

## 9. KV-Spec Selection Hook

The old draft assumed a modern built-in extension point. That is not safe across versions.

The compatibility-safe contract is:

- the quant config exposes a plugin helper `get_kv_cache_spec(...)`
- the branch adapter ensures `Attention.get_kv_cache_spec(...)` consults that helper before default dispatch

Preferred monkey patch shape:

```python
def _patched_get_kv_cache_spec(self, vllm_config):
    if self.quant_config is not None and hasattr(self.quant_config, "get_kv_cache_spec"):
        custom = self.quant_config.get_kv_cache_spec(self, vllm_config)
        if custom is not None:
            return custom
    return _orig_get_kv_cache_spec(self, vllm_config)
```

This patch must be installed before any model is constructed.

If a branch has model-specific attention classes that override `get_kv_cache_spec`, the adapter must patch those classes too or patch the common base class they call into.

---

## 10. `register_all()` Responsibilities

The plugin entrypoint must complete registration in every process that may construct a model.

Canonical sequence:

1. import the quant config module so the decorators fire
2. install cache-dtype admission shims
3. install the KV-spec dispatch shim
4. register the CUSTOM attention backend

Sketch:

```python
def register_all() -> None:
    from tilelang_turboquant.quantization.quant_config import (
        TileLangTQ3BitConfig,
        TileLangTQ4BitConfig,
    )

    _patch_cache_dtype_admission()
    _patch_attention_get_kv_cache_spec()
    _register_attention_backend()
```

Requirements:

- idempotent if called more than once
- must run before `LLM(...)`
- must not assume upstream TurboQuant is already present

---

## 11. Lifecycle Summary

Canonical lifecycle:

1. `register_all()` runs
2. `LLM(..., quantization="tq_3bit", kv_cache_dtype="tilelang_tq_3bit", attention_backend="CUSTOM")`
3. attention layers create permanent default `_k_scale/_v_scale/_q_scale/_prob_scale` buffers
4. `create_weights()` creates temporary scalar Parameters for checkpoint loading
5. checkpoint load may populate `k_scale` / `v_scale`
6. post-load hook calls `TileLangTQKVCacheMethod.process_weights_after_loading`
7. C2 artifacts and scratch buffers are registered on device
8. temporary Parameters are deleted
9. `profile_run()` sees the full memory footprint
10. KV cache allocation and reshape use the plugin-local C1 spec

No inference-critical buffer creation is allowed after step 9.

---

## 12. Testing Contracts

### Unit Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_quant_config_name_3bit` | instantiate `TileLangTQ3BitConfig` | name is `tq_3bit` |
| `test_quant_config_name_4bit` | instantiate `TileLangTQ4BitConfig` | name is `tq_4bit` |
| `test_quant_method_only_for_attention` | pass attention vs non-attention layer | method only for attention |
| `test_cache_dtype_normalization` | alias input `tq_3bit` | normalizes to `tilelang_tq_3bit` |
| `test_variant_mismatch_rejected` | quantization `tq_3bit`, cache dtype `tilelang_tq_4bit` | raises |
| `test_get_cache_scale_maps_k_proj` | checkpoint K scale name | maps to `.attn.k_scale` |
| `test_get_cache_scale_maps_v_proj` | checkpoint V scale name | maps to `.attn.v_scale` |
| `test_process_weights_defaults_scales` | temp params untouched | `_k_scale == _v_scale == 1.0` |
| `test_process_weights_registers_separate_key_value_codebooks` | matching bit-width case | both logical buffer pairs exist |
| `test_process_weights_registers_rotation_and_s` | supported config | runtime matrices present |
| `test_process_weights_deletes_temp_params` | after finalize | temp params removed |
| `test_scratch_buffers_nonpersistent` | inspect state dict | scratch buffers excluded |

### Integration Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_register_all_registers_quant_configs` | call entrypoint | `tq_3bit` / `tq_4bit` resolve |
| `test_register_all_admits_plugin_cache_dtype_literals` | construct cache config | plugin literals accepted |
| `test_attention_get_kv_cache_spec_returns_plugin_spec` | TQ layer | returns `TileLangTQAttentionSpec` |
| `test_runtime_buffers_exist_before_profile_run` | post-load then profile | buffers visible at peak |
| `test_no_lazy_gpu_materialization` | first forward after load | no first-use allocation path |
| `test_plugin_boots_on_branch_without_upstream_turboquant` | disable upstream TQ assumptions | initialization succeeds |

### Negative Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_auto_cache_dtype_rejected_for_tq_quantization` | `quantization="tq_3bit"`, `kv_cache_dtype="auto"` | explicit error |
| `test_missing_register_all_fails_cleanly` | no plugin registration | informative failure |

---

## 13. User-Facing Invocation

Canonical invocation:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="tq_3bit",
    kv_cache_dtype="tilelang_tq_3bit",
    attention_backend="CUSTOM",
)
```

If the plugin chooses to accept `kv_cache_dtype="tq_3bit"` as a user alias, it must normalize that string before `vLLM` cache-dtype validation runs.
