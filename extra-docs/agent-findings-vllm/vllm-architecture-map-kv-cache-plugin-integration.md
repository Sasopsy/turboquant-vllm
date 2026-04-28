# vLLM Architecture Map: KV Cache Plugin Integration

**Purpose:** First-principles, code-level reference for building an **out-of-tree** plugin that integrates custom KV-cache compression (e.g. **TurboQuant**-style ~3-bit caches) with custom **Triton** store and **paged decode** kernels in [vLLM](https://github.com/vllm-project/vllm).

**Codebase scope:** Paths below are relative to the vLLM Python package root (`vllm/vllm/` in a source checkout). This document targets the **V1** engine (`v1/worker/gpu_model_runner.py`, `v1/attention/`, `v1/core/kv_cache_utils.py`). A legacy `worker/model_runner.py` may not exist in all branches; treat **GPU model runner + V1 attention** as canonical.

**Related in-tree reference:** vLLM already ships a **TurboQuant** attention backend (`v1/attention/backends/turboquant_attn.py`) with Triton store (`v1/attention/ops/triton_turboquant_store.py`) and decode (`v1/attention/ops/triton_turboquant_decode.py`). Use it as an end-to-end template for non-standard cache layout and split **KV write / attention** execution.

---

## Table of contents

1. [Model execution pipeline](#1-model-execution-pipeline)
2. [Cache management (standard & quantized)](#2-cache-management-standard--quantized)
3. [Attention dispatch & registry](#3-attention-dispatch--registry)
4. [KV cache writing (`reshape_and_cache` & equivalents)](#4-kv-cache-writing-reshape_and_cache--equivalents)
5. [External plugin system](#5-external-plugin-system)
6. [Appendix: environment variables & file index](#appendix-environment-variables--file-index)

---

## 1. Model execution pipeline

### 1.1 High-level flow (decoder-only, e.g. LLaMA)

The following is the **actual** call chain for a standard transformer block with self-attention.

| Step | Component | File | What happens |
|------|-----------|------|----------------|
| 1 | `LlamaDecoderLayer.forward` | `model_executor/models/llama.py` | Routes to attention + MLP |
| 2 | `LlamaAttention.forward` | same | **QKV projection** → split → RoPE → attention |
| 3 | `QKVParallelLinear` (`qkv_proj`) | `model_executor/layers/linear.py` (via constructor) | Fused Q+K+V matmul; output shape `[*, q_size + kv_size + kv_size]` |
| 4 | `q, k, v = qkv.split(...)` | `llama.py` | **QKV projection completes here** — activations are full-rank tensors |
| 5 | `rotary_emb(positions, q, k)` | model-specific | RoPE on Q and K only |
| 6 | `self.attn(q, k, v)` | `Attention.forward` | Cache write + attention (see §1.2) |
| 7 | `o_proj` | `llama.py` | Output projection |

**Ordering fact:** The **KV cache is not written inside `qkv_proj`**. Writing happens inside the **`Attention`** module, which receives **already-projected** `k` and `v` (after RoPE on `k`).

### 1.2 `Attention.forward` — where KV meets the scheduler

File: `model_executor/layers/attention/attention.py`

1. **Optional dynamic scales (deprecated path for FP8):**  
   If `calculate_kv_scales` is true, `torch.ops.vllm.maybe_calc_kv_scales` may run once to populate `_k_scale` / `_v_scale` from activations.

2. **Optional Q-quantization for FP8 attention:**  
   When `query_quant` is set, queries are quantized before attention (helps `torch.compile` fuse ops).

3. **Shape prep (outside heavy custom ops):**  
   - `query`: `[num_tokens, num_heads * head_size_v]` → view `[num_tokens, num_heads, head_size]`  
   - `key` / `value`: `[num_tokens, num_kv_heads, head_size]` (after view)

4. **Two execution modes:**  
   - **`use_direct_call` true:** call Python helpers `unified_kv_cache_update` then `unified_attention_with_output`.  
   - **Else:** `torch.ops.vllm.unified_kv_cache_update` and `torch.ops.vllm.unified_attention_with_output` (opaque ops for compilation).

5. **KV sharing:** If `kv_sharing_target_layer_name` is set, this layer **does not** write KV; it reads shared cache in the backend.

### 1.3 `forward_includes_kv_cache_update` — ordering contract

`AttentionBackend` declares (`v1/attention/backend.py`):

- `forward_includes_kv_cache_update: bool` — default `True` for some backends, **`False` for FlashAttention and TurboQuant**.

When **`False`**:

- `Attention.forward` runs **`unified_kv_cache_update` first** (calls `impl.do_kv_cache_update`), **then** `unified_attention_with_output` (calls `impl.forward`).
- Rationale: FlashAttention’s CUDA kernel expects K/V **already in paged cache** for decode; separate “write” and “read/compute” phases match FlashAttention’s API.

When **`True`**: the backend’s single `forward` is responsible for both (less common in V1 GPU Flash path).

**For TurboQuant-style plugins:** implement **`do_kv_cache_update`** with your **store** kernel and **`forward`** with your **paged attention** kernel; set **`forward_includes_kv_cache_update = False`**.

### 1.4 Custom op glue: `unified_kv_cache_update` / `unified_attention_with_output`

File: `model_executor/layers/attention/attention.py`

**`get_attention_context(layer_name)`** returns:

- `attn_metadata` (from `ForwardContext`)
- The **`Attention`** module from `forward_context.no_compile_layers[layer_name]`
- **`attn_layer.kv_cache`** — physical tensor bound by runner
- **`slot_mapping[layer_name]`** — per-token slot indices

**`unified_kv_cache_update`:**

- Resolves layer, requires `impl.do_kv_cache_update`, calls:  
  `attn_layer.impl.do_kv_cache_update(attn_layer, key, value, kv_cache, layer_slot_mapping)`
- Returns a **dummy** scalar tensor to create a **data dependency** for `torch.compile` (ordering vs attention).

**`unified_attention_with_output`:**

- Calls `self.impl.forward(self, query, key, value, kv_cache, attn_metadata, output=..., ...)`.

### 1.5 Runner: `GPUModelRunner.execute_model` and forward context

File: `v1/worker/gpu_model_runner.py`

Rough sequence:

1. **`_update_states(scheduler_output)`** — sync batch / block tables / lengths with scheduler.
2. **`_prepare_inputs`** — logits indices, spec decode metadata.
3. **`_determine_batch_execution_and_padding`** — CUDA graph mode, padding, microbatches.
4. **`_get_slot_mappings`** — per KV-cache group, per layer: **`slot_mapping`** tensors (may be **padded** when `pad_attn` or `has_separate_kv_update`).
5. **`_build_attention_metadata`** — builds **per-layer** metadata + **`CommonAttentionMetadata`** (block tables, `query_start_loc`, `seq_lens`, etc.).
6. **`_preprocess`** — input ids, positions, embeddings.
7. **`set_forward_context(...)`** (`forward_context.py`) — installs global **`ForwardContext`** with:
   - `attn_metadata` (dict per layer or list for DBO)
   - `slot_mapping` (dict layer → tensor)
   - `cudagraph_runtime_mode`, `batch_descriptor`, `ubatch_slices`, etc.
8. **`_model_forward(...)`** — compiled or eager model forward.

**`ForwardContext`** (`forward_context.py`) holds:

- `no_compile_layers` — from `vllm_config.compilation_config.static_forward_context` (populated at model init with **every `Attention` instance** keyed by layer name).
- `attn_metadata`, `slot_mapping` — **per forward** tensors.

### 1.6 Prefill vs decode (what changes)

vLLM does **not** use separate Python entrypoints named “prefill” vs “decode” at the `Attention` layer. Instead:

- The **scheduler** decides how many tokens each request runs this step (`num_scheduled_tokens`).
- **Prefill:** many tokens per request in one batch (large `max_query_len`).
- **Decode:** typically one token per request (often `max_query_len == 1`), but batched across requests.

The **same** `Attention.forward` and **`impl.forward`** run; backends inspect metadata:

- **`query_start_loc`** (cu-seqlens for queries),
- **`seq_lens`** / **`max_seq_len`**,
- **`block_table`** (paged KV),
- **`is_prefilling`** (in `CommonAttentionMetadata`) for some backends (e.g. Mamba).

TurboQuant explicitly branches on **`is_prefill`** vs decode inside `TurboQuantAttentionImpl.forward` and uses FlashAttention varlen for many prefill cases.

### 1.7 Data structures (summary)

| Tensor / object | Typical shape / role |
|-----------------|----------------------|
| `query` | `[num_tokens, num_heads, head_dim]` (after view) |
| `key`, `value` | `[num_tokens, num_kv_heads, head_dim]` |
| `kv_cache` | Backend-specific; see §2 / §4 |
| `slot_mapping[layer]` | `[num_tokens]` or padded length; `int64` slot index per token |
| `block_table` | `[num_reqs, max_num_blocks_per_seq]` — physical block IDs for paged attention |
| `attn_metadata[layer]` | Backend-specific (`FlashAttentionMetadata`, `TurboQuantMetadata`, …) |

**Padding:** When CUDA graphs or separate KV-update backends require it, **`num_tokens_padded`** and **`num_reqs_padded`** can exceed the “logical” unpadded counts; **`num_actual_tokens`** in metadata tells kernels how many entries are real.

### 1.8 Integration hooks (pipeline)

| Goal | Where to hook |
|------|----------------|
| Replace **store** kernel | `AttentionImpl.do_kv_cache_update` |
| Replace **attention** (prefill/decode) | `AttentionImpl.forward` |
| Change **metadata** (block table layout, seq info) | `AttentionMetadataBuilder` for your backend |
| Change **tensor shapes** for allocation | `AttentionBackend.get_kv_cache_shape`, `get_kv_cache_stride_order` |
| Ensure runner passes your metadata | `GPUModelRunner._build_attention_metadata` uses per-backend builders — keep builder registered via backend class |

---

## 2. Cache management (standard & quantized)

### 2.1 From spec to GPU bytes

KV memory is **not** guessed by a standalone profiler class. The pipeline is:

1. **Per-layer `KVCacheSpec`** — from `Attention.get_kv_cache_spec(vllm_config)` (`attention.py`).
2. **Group layers** — `get_kv_cache_groups` (`v1/core/kv_cache_utils.py`) merges layers that share block tables / compatible specs.
3. **Compute `KVCacheConfig`** — `get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory)`:
   - Determines **`num_blocks`**
   - Emits **`KVCacheTensor`** entries: each has **`size` in bytes** and **`shared_by`** layer names.

4. **Worker profiling** (`v1/worker/gpu_worker.py`):  
   - Runs **`profile_run`** / dummy forward to measure **weights + activations + torch peak**.  
   - Derives **free memory** after accounting for **`non_kv_cache_memory`**.  
   - Passes **`available_memory`** into KV cache config generation (exact call path via engine `_initialize_kv_caches`).

5. **`GPUModelRunner.initialize_kv_cache_tensors`**:  
   - Allocates raw byte buffers per `KVCacheTensor`  
   - **`_reshape_kv_cache_tensors`**: views as the dtype/shape expected by **`AttentionBackend.get_kv_cache_shape`** and permutes strides via **`get_kv_cache_stride_order`**.

6. **`bind_kv_cache`** (`v1/worker/utils.py`):  
   - Fills **`runner.kv_caches`** (list order by layer index)  
   - Sets **`forward_context[layer_name].kv_cache = tensor`** for each `Attention` module.

### 2.2 Bytes per block (“page size”)

Core type: `KVCacheSpec` (`v1/kv_cache_interface.py`).

- Property **`page_size_bytes`** is the **logical** size of one **block** (one scheduler block of `block_size` tokens) for that layer/spec.
- For **`AttentionSpec`**:
  - **`real_page_size_bytes`** ≈ `2 * block_size * num_kv_heads * head_size * dtype_size` (K and V).
  - If **`kv_quant_mode.is_per_token_head`**, add **`2 * block_size * num_kv_heads * sizeof(float32)`** for **separate** K/V scale storage (budgeted even though stored outside the main tensor in some backends).
  - Optional **`page_size_padded`** for alignment.

**TurboQuant:** `TQFullAttentionSpec` overrides **`real_page_size_bytes`** to  
`block_size * num_kv_heads * tq_slot_size` (packed slot, not `2 * head_dim * dtype`).

### 2.3 Computing `num_blocks`

From `get_kv_cache_config_from_groups` (`v1/core/kv_cache_utils.py`):

- **Uniform multi-layer groups (general case):**  
  `page_size = get_uniform_page_size(kv_cache_groups)`  
  `num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)`  
  where `get_num_blocks` does:  
  `num_blocks = available_memory // page_size // num_layers_in_group_pattern`  
  (see function — **`group_size`** is the number of “slots” in the repeating pattern.)

- **UniformTypeKVCacheSpecs** (same attention type, different hidden sizes per layer): special-case uses **`page_size_bytes`** of the combined spec directly.

- **`may_override_num_blocks`**: `cache_config.num_gpu_blocks_override` **wins** for tests or fixed sizing.

### 2.4 Quantization: `CacheConfig` vs `QuantizationConfig`

**`CacheConfig`** (`config/cache.py`):

- **`cache_dtype`**: storage dtype for KV (`CacheDType` literal: `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, per-token-head modes, **`turboquant_*`**, …).
- **`calculate_kv_scales`**: legacy path for dynamic K/V scales (deprecated; interacts with CUDA graph disabling in runner).
- **`kv_cache_dtype_skip_layers`**: per-layer or attention-type skips — forces **`auto`** cache for those layers.
- **`kv_cache_memory_bytes`**: optional override of total KV bytes (ignores `gpu_memory_utilization` when set).

**`QuantizationConfig`** (model weights + optional KV schemes):

- **`Attention.__init__`** calls **`_init_kv_cache_quant`**:  
  - `quant_config.get_quant_method(self, prefix=...)`  
  - If method is **`BaseKVCacheMethod`**, **`create_weights`** adds **`q_scale`, `k_scale`, `v_scale`, `prob_scale`** as **parameters** for checkpoint loading.
- **`kv_cache_scheme`** (some configs): can force **`cache_dtype`** to **`fp8`** and align with compressed-tensors / llm-compressor workflows.
- **`BaseKVCacheMethod.process_weights_after_loading`** (`model_executor/layers/quantization/kv_cache.py`):  
  - Copies checkpoint scales to **`_k_scale` / `_v_scale`** buffers used by kernels.  
  - **Per-token-head** modes: removes checkpoint scales and relies on kernel-computed scales.

### 2.5 Parallel tensors for scales / metadata

- **FP8 per-tensor:** scales live on the **`Attention`** layer as **`_k_scale` / `_v_scale`** (buffers or params), passed into **`reshape_and_cache_flash`** and FlashAttention as descales.
- **Per-token-head:** extra **page size** is reserved in **`AttentionSpec.page_size_bytes`**; backends may create **separate GPU tensors** (e.g. `_k_scale_cache` on impl — see cleanup in `gpu_model_runner._cleanup_profiling_kv_cache`).
- **TurboQuant:** rotation signs, centroids, midpoints, decode buffers — initialized in **`Attention._init_turboquant_buffers`** and **`TurboQuantAttentionImpl._ensure_on_device`** (`turboquant_attn.py`). Buffers are registered so **`model.to(device)`** runs **before** KV profiling (avoids OOM on first decode).

### 2.6 Integration hooks (cache)

| Task | Mechanism |
|------|-----------|
| New physical layout / bytes per block | New or extended **`KVCacheSpec`** + correct **`page_size_bytes`** |
| New `cache_dtype` string | Add to **`CacheDType`** in `config/cache.py` (upstream) or reuse existing **`turboquant_*`** |
| Checkpoint-loaded scales | **`BaseKVCacheMethod`** + **`register_quantization_config`** |
| Allocator sees your layout | **`AttentionBackend.get_kv_cache_shape`** + **`get_kv_cache_stride_order`** consistent with **`page_size_bytes`** |

---

## 3. Attention dispatch & registry

### 3.1 Selection API

**`get_attn_backend(...)`** — `v1/attention/selector.py`

- Reads **`get_current_vllm_config()`**.
- Builds **`AttentionSelectorConfig`**: `head_size`, `dtype`, **`kv_cache_dtype`**, optional **`block_size`** (if user specified block size in cache config), MLA/sparse/MM-prefix flags, etc.
- Calls **`_cached_get_attn_backend(backend=vllm_config.attention_config.backend, ...)`** — cached on full config tuple.

**`_cached_get_attn_backend`:**

1. **`current_platform.get_attn_backend_cls(selected_backend, attn_selector_config, num_heads)`**  
   - Returns a **string** (fully qualified class path), not an instance.
2. **`resolve_obj_by_qualname(attention_cls)`** — imports class.
3. If **`backend.get_required_kv_cache_layout()`** is not `None`, calls **`set_kv_cache_layout(...)`** so NHD/HND matches the backend.

### 3.2 CUDA platform logic (typical GPU)

File: `platforms/cuda.py` — **`CudaPlatform.get_attn_backend_cls`** / **`get_valid_backends`**

- If **`attention_config.backend`** is set: **only** that `AttentionBackendEnum` is validated via **`validate_configuration`**; failure raises **`ValueError`**.
- If auto-selection: iterates **priority list** from **`_get_backend_priorities`** (MLA, dtype, heads, etc.).
- **TurboQuant shortcut:** if `kv_cache_dtype.startswith("turboquant_")`, returns **`[(AttentionBackendEnum.TURBOQUANT, 0)]`** immediately.

### 3.3 `AttentionBackendEnum` and overrides

File: `v1/attention/backends/registry.py`

- Each enum member’s **value** is the **default** import path (e.g. `FLASH_ATTN = "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend"`).
- **`register_backend(enum, class_path)`** or **`@register_backend(enum)`** populates **`_ATTN_OVERRIDES`** / **`_MAMBA_ATTN_OVERRIDES`**.
- At runtime, **`enum.get_path()`** returns override or default; **`get_class()`** resolves the class.

**`CUSTOM`:** enum value is `None`; **must** be registered before use.

### 3.4 Abstract backend interface (must implement)

From `v1/attention/backend.py` (class **`AttentionBackend`**):

**Class-level:**

- `get_name() -> str` — must match **`AttentionBackendEnum`** name where applicable (e.g. `"FLASH_ATTN"`).
- `get_impl_cls() -> AttentionImpl`
- `get_builder_cls()` — returns **`AttentionMetadataBuilder`** subclass
- `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str=...) -> tuple[int, ...]`
- Optional: `get_kv_cache_stride_order`, `get_kv_cache_block_dim`, `get_supported_kernel_block_sizes`, `supports_kv_cache_dtype`, `validate_configuration`, etc.

**AttentionImpl** (see `AttentionImpl` / `AttentionImplBase` in same file family):

- `forward(layer, query, key, value, kv_cache, attn_metadata, output=..., ...)`
- If `forward_includes_kv_cache_update == False`: implement **`do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)`**

**AttentionMetadataBuilder:**

- Builds backend-specific metadata from **`CommonAttentionMetadata`** and scheduler state; supports CUDA graph capture hooks where applicable (`build_for_cudagraph_capture`, etc.).

### 3.5 Model-side registration

`Attention.__init__` (`attention.py`):

- **`attn_backend = get_attn_backend(...)`** unless explicitly passed.
- **`impl_cls = self.attn_backend.get_impl_cls()`** → **`self.impl = impl_cls(...)`**
- **`self.backend = AttentionBackendEnum[self.attn_backend.get_name()]`**

Exposes **`get_attn_backend()`** for runner utilities (`gpu_model_runner`, spec decode, KV connectors).

### 3.6 Integration hooks (dispatch)

| Approach | How |
|----------|-----|
| Swap Flash implementation | `@register_backend(AttentionBackendEnum.FLASH_ATTN) class MyFlash: ...` |
| Third-party backend | `register_backend(AttentionBackendEnum.CUSTOM, "pkg.MyBackend")` and select **`CUSTOM`** from config (if exposed) |
| TurboQuant-like routing | Ensure **`get_valid_backends`** or user-selected backend includes your backend when `cache_dtype` matches |
| MLA / Mamba | Separate enums: **`MambaAttentionBackendEnum`**, **`register_backend(..., is_mamba=True)`** |

---

## 4. KV cache writing (`reshape_and_cache` & equivalents)

### 4.1 Naming: multiple “store” entry points

vLLM uses **several** APIs depending on backend and layout:

| API | Definition | Typical caller |
|-----|------------|----------------|
| `reshape_and_cache` | `_custom_ops.reshape_and_cache` → `torch.ops._C_cache_ops.reshape_and_cache` | Older / non-flash layouts |
| `reshape_and_cache_flash` | `_custom_ops.reshape_and_cache_flash` → C++ `reshape_and_cache_flash` | **FlashAttention** GPU path |
| Triton `triton_reshape_and_cache_flash` | `v1/attention/ops/triton_reshape_and_cache_flash.py` | ROCm / Triton / FP8 variants |
| `triton_turboquant_store` | `v1/attention/ops/triton_turboquant_store.py` | TurboQuant combined cache |
| `concat_and_cache_mla` | C++ op | MLA caches |

### 4.2 FlashAttention path — shapes

**`FlashAttentionImpl.do_kv_cache_update`** (`v1/attention/backends/flash_attn.py`):

- `key_cache, value_cache = kv_cache.unbind(0)`  
  → **`kv_cache` shape** `[2, num_blocks, block_size, num_kv_heads, head_size]`.
- Calls **`reshape_and_cache_flash(key, value, key_cache, value_cache, slot_mapping, kv_cache_dtype, k_scale, v_scale)`**.

**Triton kernel documentation** (`triton_reshape_and_cache_flash.py`):

- Keys/values in: `[num_tokens, num_heads, head_size]`
- Cache: `[num_blocks, block_size, num_heads, head_size]` (per K and V tensor)
- `slot_mapping`: `[num_tokens]`, entries **`>= 0`** active, **`< 0`** ignored (padding)

Mapping: `block_idx = slot_idx // block_size`, `block_offset = slot_idx % block_size`.

### 4.3 Quantized KV (FP8) in store

- Same **`reshape_and_cache_flash`** entry; **`kv_cache_dtype`** string selects FP8 behavior.
- Triton path uses **`k_scale` / `v_scale`** when storing from BF16/FP16 to FP8.
- **Per-token-head** variants use dedicated Triton entry points (e.g. `triton_reshape_and_cache_flash_per_token_head_quant` in the same module).

### 4.4 TurboQuant store (reference for custom packed cache)

**`TurboQuantAttentionImpl.do_kv_cache_update`:**

- Slices tokens: `N = slot_mapping.shape[0]`; `key[:N]`, `value[:N]`.
- Calls **`triton_turboquant_store(key, value, kv_cache, slot_mapping, ...)`**.

**`kv_cache`:** `[num_blocks, block_size, num_kv_heads, slot_size_aligned]` — **no** leading `2` dimension.

### 4.5 Integration hooks (store)

| Need | Action |
|------|--------|
| Replace Flash store | Monkey-patch / register backend that overrides **`do_kv_cache_update`** or swap **`FLASH_ATTN`** class via **`register_backend`** |
| New packed layout | New **`get_kv_cache_shape`**, **`do_kv_cache_update`** implementation, and matching **decode** kernel |
| Keep CUDA graphs safe | Match **`num_actual_tokens`** / padded **`key`/`value`** with **`slot_mapping`** length (see `execute_model` flags) |

---

## 5. External plugin system

### 5.1 Entry point group

File: `plugins/__init__.py`

- **`DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"`**
- Also: `vllm.io_processor_plugins`, `vllm.platform_plugins`, `vllm.stat_logger_plugins` (different lifecycles).

### 5.2 Load sequence

1. **`load_plugins_by_group("vllm.general_plugins")`**  
   - Uses **`importlib.metadata.entry_points(group="vllm.general_plugins")`**.  
   - If **`envs.VLLM_PLUGINS`** is **`None`**: load **all** plugins.  
   - Else: load only entry points whose **name** is in the comma-separated allowlist.

2. For each selected entry: **`plugin.load()`** returns a **callable** (no arguments).

3. **`load_general_plugins()`** invokes **each callable once**, then sets **`plugins_loaded = True`** so repeat calls in the same process are no-ops.

### 5.3 Where `load_general_plugins()` runs (multi-process)

Plugins execute in **every process** that imports vLLM initialization — typical call sites:

| Location | File |
|----------|------|
| V1 engine core | `v1/engine/core.py` — `EngineCore.__init__` |
| Worker bootstrap | `v1/worker/worker_base.py` — `init_worker` |
| CLI / args | `engine/arg_utils.py` (multiple paths) |
| Model registry subprocess | `model_executor/models/registry.py` — `_run` |
| Lazy imports | Other modules may pull `load_general_plugins` indirectly |

**Design constraint (from docstring):** plugins must tolerate **multiple loads** across processes (idempotent registration).

### 5.4 Registering a `QuantizationConfig`

File: `model_executor/layers/quantization/__init__.py`

- **`@register_quantization_config("my_method")`** adds **`my_method`** to **`QUANTIZATION_METHODS`** and maps it to your **`QuantizationConfig`** subclass in **`_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`**.
- **`get_quantization_config("my_method")`** lazy-imports your class.

### 5.5 Registering an `AttentionBackend`

File: `v1/attention/backends/registry.py`

- **`register_backend(AttentionBackendEnum.X, "module.QualName")`** **or** decorator **`@register_backend(AttentionBackendEnum.X)`** on your class.

### 5.6 Example `pyproject.toml` snippet

```toml
[project.entry-points."vllm.general_plugins"]
my_turboquant_plugin = "my_pkg.vllm_plugin:init_vllm_plugin"

[tool.setuptools.packages.find]
where = ["src"]
```

```python
# my_pkg/vllm_plugin.py
def init_vllm_plugin() -> None:
    from vllm.v1.attention.backends.registry import (
        register_backend,
        AttentionBackendEnum,
    )
    from vllm.model_executor.layers.quantization import register_quantization_config

    register_backend(AttentionBackendEnum.CUSTOM, "my_pkg.attn.MyBackend")
    # Optional: register_quantization_config("my_quant")(MyQuantConfig)
```

Install package in the same environment as vLLM (`pip install -e .`).

### 5.7 Custom torch ops (optional)

`vllm/_oink_ops.py` documents that external code may register **`torch.ops.oink.*`** under **`VLLM_USE_OINK_OPS`** — pattern for **CUDA** custom ops without merging into vLLM. Same **general_plugins** load order applies.

### 5.8 Integration checklist (out-of-tree)

1. **Entry point** under **`vllm.general_plugins`** calling **`register_backend`** / **`register_quantization_config`**.
2. **`VLLM_PLUGINS=my_turboquant_plugin`** if you need to restrict loads in shared environments (`envs.py`: **`VLLM_PLUGINS`**).
3. **`CacheDType`** extended upstream if you need a **new** `--kv-cache-dtype` string; or reuse **`turboquant_*`** if semantically compatible.
4. Implement **`AttentionBackend` + AttentionImpl + MetadataBuilder`**; wire **`validate_configuration`** for your dtype/head dim.
5. Test **engine + worker** paths (plugins load in **both** processes).

---

## Appendix: environment variables & file index

### A.1 Selected environment variables

| Variable | Role |
|----------|------|
| **`VLLM_PLUGINS`** | Comma-separated list of **entry point names** to load from `vllm.general_plugins`; if unset, all plugins load. |
| **`VLLM_USE_OINK_OPS`** | Gate for optional external **`torch.ops.oink`** kernels (`_oink_ops.py`). |

(See `envs.py` for full list.)

### A.2 File index (primary)

| Topic | Path |
|-------|------|
| LLaMA attention block | `model_executor/models/llama.py` |
| Attention layer | `model_executor/layers/attention/attention.py` |
| Forward context | `forward_context.py` |
| V1 GPU runner | `v1/worker/gpu_model_runner.py` |
| Bind KV tensors | `v1/worker/utils.py` — `bind_kv_cache` |
| KV config + blocks | `v1/core/kv_cache_utils.py` — `get_kv_cache_config_from_groups`, `get_num_blocks` |
| KV specs | `v1/kv_cache_interface.py` |
| Cache config | `config/cache.py` |
| Attention selector | `v1/attention/selector.py` |
| Backend registry | `v1/attention/backends/registry.py` |
| Flash backend | `v1/attention/backends/flash_attn.py` |
| TurboQuant backend | `v1/attention/backends/turboquant_attn.py` |
| Triton reshape cache | `v1/attention/ops/triton_reshape_and_cache_flash.py` |
| Python ops shim | `_custom_ops.py` — `reshape_and_cache*`, `concat_and_cache_mla` |
| KV quant method | `model_executor/layers/quantization/kv_cache.py` |
| Quant registration | `model_executor/layers/quantization/__init__.py` |
| Plugins | `plugins/__init__.py` |
| CUDA platform | `platforms/cuda.py` |

---

## Document history

- **2026-04-17:** Expanded from an internal architecture map; aligned with vLLM V1 paths under `vllm/vllm/`.

---

*This document is a structural map for engineering integration; it is not an official vLLM project doc. Verify against your exact vLLM commit when implementing.*
