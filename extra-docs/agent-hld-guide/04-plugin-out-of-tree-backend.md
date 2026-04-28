# Agent reference: out-of-tree plugin & `AttentionBackend` contract

**Audience:** LLM drafting HLD/LLD for CUSTOM attention backends (TileLang/TQ-style) without forking vLLM.

---

## Multi-process rule

- Parent `register_backend(...)` does **not** populate workers/engine subprocesses.
- **`vllm.general_plugins` entry point** → `load_general_plugins()` (`vllm/plugins/__init__.py`) runs your callable **once per process** (guarded by `plugins_loaded`).
- `load_plugins_by_group` discovers entry points; **`load_general_plugins` executes** callables. Failed plugin load logged, skipped.

**`pyproject.toml`**

```toml
[project.entry-points."vllm.general_plugins"]
your_plugin = "your_pkg.plugin:register"
```

**`register()`:** import vLLM inside function; `register_backend(AttentionBackendEnum.CUSTOM, "your_pkg.backend.YourBackend")`. **Idempotent, deterministic** string; no per-rank conditional class paths.

---

## Registry (`vllm/v1/attention/backends/registry.py`)

- `AttentionBackendEnum.CUSTOM = None` — **no default path**; must register or `get_path()` raises.
- `_ATTN_OVERRIDES[enum] = "module.Class"`; `get_class()` dynamic import.

**Forms:** `@register_backend(CUSTOM)` decorator on class, or `register_backend(CUSTOM, "fqn")` (preferred for plugins).

---

## Resolution chain

1. CLI/config: `AttentionConfig.backend` — `"auto"` → `None`; else `AttentionBackendEnum[name.upper()]`.
2. `get_attn_backend` → `_cached_get_attn_backend` (`@cache` on `(backend, AttentionSelectorConfig, num_heads)`).
3. `current_platform.get_attn_backend_cls`: for explicit backend, `get_class()` → **`validate_configuration`** → on failure **`ValueError`**, no fallback.
4. Returns FQN string → `resolve_obj_by_qualname` → `type[AttentionBackend]`.
5. Optional: `get_required_kv_cache_layout() != None` → `set_kv_cache_layout` (global).

**Auto backend:** `CUSTOM` **not** in auto list — must pass `--attention-backend CUSTOM`.

---

## `AttentionBackend` minimum surface

**Abstract staticmethods**

- `get_name() -> str`
- `get_impl_cls() -> AttentionImpl`
- `get_builder_cls() -> AttentionMetadataBuilder`
- `get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str) -> tuple[int, ...]`

**Class vars (typical custom KV)**

- `supported_dtypes = [fp16, bf16]`
- `supported_kv_cache_dtypes`: must include every `CacheDType` you pass (see `vllm/config/cache.py`), often `"auto"` + your literal e.g. `"turboquant_3bit_nc"`.
- `forward_includes_kv_cache_update = False` for split store.

**Overrides when needed**

- `get_kv_cache_stride_order`, `get_required_kv_cache_layout`, `get_supported_kernel_block_sizes`, `get_supported_head_sizes`
- Capability classmethods: `is_mla`, `supports_sink`, `supports_mm_prefix`, `is_sparse`, `supports_per_head_quant_scales`, `supports_non_causal`, `supports_attn_type`, `supports_compute_capability`, `supports_combination`

---

## `validate_configuration` gates (`attention/backend.py`)

Checks include: head size list, `dtype in supported_dtypes`, **`kv_cache_dtype in supported_kv_cache_dtypes`**, block size vs `get_supported_kernel_block_sizes`, MLA/sink/sparse/mm_prefix/per-head scales flags vs model, compute capability, attn type, non-causal, combination hook. Fail → list of reasons → `ValueError` in `CudaPlatform.get_attn_backend_cls`.

---

## `AttentionImpl`

- `__init__(num_heads, head_size, scale, num_kv_heads=..., kv_cache_dtype=..., ...)`
- `forward(layer, query, key, value, kv_cache, attn_metadata, output, ...)` — **in-place `output`**, return it.
- If split store: **`do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)`** — no `attn_metadata` arg here.
- Optional: `process_weights_after_loading(act_dtype)` for kernel-specific prep.

---

## `AttentionMetadataBuilder`

- `__init__(kv_cache_spec, layer_names, vllm_config, device)` — call `super`.
- `build(common_prefix_len, common_attn_metadata, fast_build=False) -> MetadataDataclass`
- Optional: `_cudagraph_support`, `reorder_batch_threshold`, `build_for_cudagraph_capture`, spec-decoding hooks.

---

## End-to-end launch

```bash
vllm serve MODEL --attention-backend CUSTOM --kv-cache-dtype your_dtype
# quantization="your_method" if separate quant config drives spec/buffers
```

Package **installed** so entry points visible to `importlib.metadata`.

---

## Ordered implementation checklist

1. `pyproject.toml` entry point → `register()` → `register_backend(CUSTOM, fqn)`.
2. `AttentionBackend` + `supported_kv_cache_dtypes` + `get_kv_cache_shape` consistent with **Part 1** invariant.
3. `AttentionImpl`: `do_kv_cache_update` + `forward`.
4. `AttentionMetadataBuilder` + metadata dataclass.
5. `QuantizationConfig` + `BaseKVCacheMethod` if checkpoint metadata (Part 3).
6. `pip install -e .`; verify `entry_points(group='vllm.general_plugins')`.

---

## Key files

| Topic | Path |
|-------|------|
| Plugins | `vllm/plugins/__init__.py` |
| Registry | `vllm/v1/attention/backends/registry.py` |
| Selector | `vllm/v1/attention/selector.py` |
| Backend base + validate | `vllm/v1/attention/backend.py` |
| Attention config | `vllm/config/attention.py` |
| Cache dtypes | `vllm/config/cache.py` |
| Platform gate | `vllm/platforms/cuda.py` |
| Unified KV / attention | `vllm/model_executor/layers/attention/attention.py` |

---

## Doc map (this folder)

| File | Deep-dive source |
|------|------------------|
| `01-kv-allocation-and-memory-illusion.md` | `01-the-memory-illusion-kv-allocation.md` + `turboquant_algorithm.md` (math summary) |
| `02-split-store-forward-and-metadata-flow.md` | `02-split-execution-the-read-write-barrier.md` |
| `03-checkpoint-to-attention-buffers.md` | `03-metadata-and-quantization-injection.md` |
| `04-plugin-out-of-tree-backend.md` | `04-the-plugin-heist-hijacking-vllm.md` |
