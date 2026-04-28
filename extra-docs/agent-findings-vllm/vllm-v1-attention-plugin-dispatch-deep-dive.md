# vLLM V1 Attention Dispatch Registry & General Plugins — Technical Report

**Scope:** `vllm.plugins`, `vllm.v1.attention.selector`, `vllm.v1.attention.backends.registry`, `vllm.v1.attention.backend` (as present in the analyzed tree).  
**Goal:** Map plugin initialization, `AttentionBackendEnum.CUSTOM` registration, and the concrete interface contract for an out-of-tree attention backend (e.g. Tile Lang–based ~3-bit KV) **without editing vLLM source**.

---

## 1. Plugin lifecycle: `load_plugins_by_group("vllm.general_plugins")` and `load_general_plugins()`

### 1.1 Entry points and constants

From `vllm/plugins/__init__.py`:

- **`DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"`** — documented as loaded in all processes (engine core, workers, etc.).
- **`load_plugins_by_group(group)`** uses `importlib.metadata.entry_points(group=group)` to discover plugins, optionally filters by `envs.VLLM_PLUGINS` (allowlist of plugin **names**; `None` loads all).
- For each selected entry point, **`plugin.load()`** returns a callable; that callable is stored in a dict `plugins[name] -> func`. **Exceptions during load are logged; failed plugins are skipped** (no crash of the whole group).

### 1.2 `load_general_plugins()` execution sequence

```text
load_general_plugins():
  if plugins_loaded: return          # process-local guard
  plugins_loaded = True
  plugins = load_plugins_by_group(DEFAULT_PLUGINS_GROUP)
  for func in plugins.values():
      func()                         # side-effect registration runs here
```

**Important semantics:**

1. **Per-process singleton:** `plugins_loaded` is a **module global**. Within one Python process, **`load_general_plugins()` only executes the loaded callables once**, even if invoked from multiple call sites.
2. **Multi-process / distributed:** Each worker, engine process, or subprocess has its **own** interpreter → **`plugins_loaded` resets** → **each process runs the plugin entry functions exactly once** (on first `load_general_plugins()` in that process). This matches the documented expectation that plugins run everywhere vLLM runs.
3. **Call sites (non-exhaustive but relevant):** `EngineCore.__init__`, `WorkerBase.init_worker`, `model_executor/models/registry.py`, and CLI paths in `engine/arg_utils.py` all invoke `load_general_plugins()`. Order varies by launch path, but **registration must be valid whenever `get_attn_backend` first resolves the backend** in that process.

### 1.3 Relationship to `load_plugins_by_group` alone

`load_plugins_by_group` **does not** set `plugins_loaded`. It only discovers and imports entry points and returns the dict of callables. **`load_general_plugins` is the thin wrapper that (a) deduplicates execution per process and (b) invokes each callable.**

### 1.4 Idempotency and multi-process safety

**Official guidance** (docstring on `load_general_plugins`): plugins may be loaded multiple times in different processes; they should tolerate **multiple loads without corruption**.

**Mechanisms:**

| Concern | Behavior |
|--------|----------|
| **Duplicate `load_general_plugins()` in same process** | Second+ call is a no-op (`plugins_loaded`); plugin callables **not** re-run. |
| **Fresh process (worker bootstrap)** | `plugins_loaded` is False → plugins run **once** in that process. |
| **`register_backend(...)`** | Writes into module-level `_ATTN_OVERRIDES` / `_MAMBA_ATTN_OVERRIDES`. **Re-assigning the same `AttentionBackendEnum` key is idempotent** if the class path is stable. **Last registration wins** if code paths disagree. |
| **Model registry / other registration** | Same pattern: use **guards** (e.g. “if not already registered”) for registries that are not overwrite-safe; vLLM’s own docs recommend **re-entrant** plugin functions. |

**Practical recipe for a Tile Lang KV plugin:**

- Put **`register_backend(AttentionBackendEnum.CUSTOM, ...)`** inside the general plugin entry function so it runs on every process’s first `load_general_plugins()`.
- Prefer **deterministic** registration (same class path every time). Avoid conditional registration that could diverge across ranks.
- If you also register models or other globals, follow the same **test-before-register** pattern as in the official plugin examples.

---

## 2. The `CUSTOM` registry hook: `register_backend` and routing to `get_attn_backend`

### 2.1 Enum and override maps

In `vllm/v1/attention/backends/registry.py`:

- **`AttentionBackendEnum.CUSTOM = None`** — intentional: avoids colliding with backends that use empty string; **`CUSTOM` has no default class path**.
- **`_ATTN_OVERRIDES: dict[AttentionBackendEnum, str]`** holds runtime overrides (fully qualified class names as strings).

**`AttentionBackendEnum.get_path()`** returns `_ATTN_OVERRIDES.get(self, self.value)`. For `CUSTOM`, `self.value` is falsy, so **without an override, `get_path()` raises** `ValueError` instructing use of `register_backend`.

**`get_class()`** calls `resolve_obj_by_qualname(self.get_path())` — dynamic import of the backend class.

### 2.2 `register_backend` API (exact behavior)

```python
def register_backend(
    backend: AttentionBackendEnum | MambaAttentionBackendEnum,
    class_path: str | None = None,
    is_mamba: bool = False,
) -> Callable[[type], type] | ...
```

**Two forms:**

1. **Decorator (no `class_path`):**  
   `@register_backend(AttentionBackendEnum.CUSTOM)`  
   → sets `_ATTN_OVERRIDES[backend] = f"{cls.__module__}.{cls.__qualname__}"`.

2. **Direct string registration:**  
   `register_backend(AttentionBackendEnum.CUSTOM, "my_pkg.backend.TileLangKVBackend")`  
   → sets override to that string; returns a no-op lambda (for API symmetry).

**Also supported:** overriding **existing** enum members (e.g. `FLASH_ATTN`) to point at a different implementation — same override map.

**Mamba:** `is_mamba=True` uses `_MAMBA_ATTN_OVERRIDES` and `MambaAttentionBackendEnum` (includes its own `CUSTOM`).

### 2.3 Wiring config → selector → platform → resolved class

**Config:** `AttentionConfig.backend: AttentionBackendEnum | None` (`vllm/config/attention.py`).  
Strings from CLI/config are normalized: `"auto"` → `None`; otherwise **`AttentionBackendEnum[value.upper()]`**, so the user selects **`CUSTOM`** explicitly.

**Selector:** `get_attn_backend(...)` builds `AttentionSelectorConfig`, reads `vllm_config.attention_config.backend`, and calls **`_cached_get_attn_backend(backend=..., attn_selector_config=..., num_heads=...)`**.

**Caching:** `_cached_get_attn_backend` is decorated with **`@cache`** (functools). The resolved **`type[AttentionBackend]`** is cached keyed by `(backend enum, attn_selector_config, num_heads)`. **Change `AttentionSelectorConfig` fields → new cache entry.**

**Platform resolution:** `_cached_get_attn_backend` calls:

```python
attention_cls = current_platform.get_attn_backend_cls(
    backend,
    attn_selector_config=attn_selector_config,
    num_heads=num_heads,
)
# ...
backend = resolve_obj_by_qualname(attention_cls)  # attention_cls is a STRING (FQN)
```

On **CUDA** (`platforms/cuda.py`), when `selected_backend is not None`:

1. **`selected_backend.get_class()`** — imports and obtains the **class** (uses your `register_backend` path for `CUSTOM`).
2. **`validate_configuration(...)`** — if any invalid reasons, **`ValueError`** with reasons; otherwise:
3. **`return selected_backend.get_path()`** — returns the **string FQN** (not the class object) for `resolve_obj_by_qualname` in the selector.

So for **`CUSTOM`** to route correctly:

1. **`register_backend(AttentionBackendEnum.CUSTOM, ...)` must run before** the first `get_attn_backend` that needs `CUSTOM` in that process.
2. **`AttentionConfig.backend` must be `AttentionBackendEnum.CUSTOM`** (e.g. `--attention-backend CUSTOM`).
3. Your **`AttentionBackend` subclass must pass `validate_configuration`** for the given `AttentionSelectorConfig` (dtype, head size, KV dtype, MLA flags, etc.). Failures surface as **`ValueError`** from `get_attn_backend_cls`, not as silent fallback.

**Auto-selection path:** If `backend is None`, the platform enumerates priorities and picks a valid backend; **`CUSTOM` is not auto-selected** unless your platform’s priority list includes it (standard CUDA stack does not depend on `CUSTOM` for auto mode). **Out-of-tree backends normally require explicit `CUSTOM` + registration.**

### 2.4 KV cache layout side effect

After resolving the class, the selector calls **`backend.get_required_kv_cache_layout()`**. If non-`None`, it invokes **`set_kv_cache_layout(required_layout)`** (`vllm.v1.attention.backends.utils`). Custom backends should override **`get_required_kv_cache_layout`** when layout must differ from default.

---

## 3. Backend interface contract: `AttentionBackend`, `AttentionImpl`, `AttentionMetadataBuilder`

Below, **“must implement”** means **abstract or required for the framework to call your code on the standard decoder path**. Optional hooks are noted.

### 3.1 `AttentionBackend` (abstract base)

**Declared `@abstractmethod` (must implement):**

| Member | Role |
|--------|------|
| **`get_name() -> str`** | Static method; human-readable / logging name. |
| **`get_impl_cls() -> type[AttentionImplBase]`** | Returns the **`AttentionImpl`** (or compatible) class used at runtime. |
| **`get_builder_cls()`** | Returns **`AttentionMetadataBuilder`** subclass for this backend. |
| **`get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto") -> tuple[int, ...]`** | Logical KV tensor shape for allocation/planning. |

**Strongly expected for correct integration (defaults exist but often wrong for exotic KV):**

- **`supported_dtypes` / `supported_kv_cache_dtypes`** class vars — used by **`validate_configuration`**.
- **`get_supported_kernel_block_sizes()`** — default `[MultipleOf(1)]`; backends with stricter alignment should override (see FlashAttention).
- **`forward_includes_kv_cache_update`** — whether Q/K/V forward path includes cache write (affects layering).

**Override when your backend differs:**

- **`get_kv_cache_stride_order(...)`** — if physical layout ≠ logical shape; default raises `NotImplementedError` and framework assumes logical = physical.
- **`get_required_kv_cache_layout()`** — return a layout enum value if the global layout must be switched.
- Capability flags: **`is_mla`**, **`supports_sink`**, **`supports_mm_prefix`**, **`is_sparse`**, **`supports_per_head_quant_scales`**, **`supports_non_causal`**, **`supports_attn_type`**, **`supports_compute_capability`**, **`supports_combination`** — must align with what you actually implement; **`validate_configuration`** aggregates these.

**Not abstract but part of contract for “odd” dtypes:** extend **`supported_kv_cache_dtypes`** and any **`supports_*`** checks so your ~3-bit / quantized KV dtype is accepted.

### 3.2 `AttentionImpl` (standard decoder attention — non-MLA)

Subclass **`AttentionImpl[T]`** where `T` is your metadata type.

**Declared `@abstractmethod`:**

| Member | Role |
|--------|------|
| **`__init__(..., num_heads, head_size, scale, num_kv_heads=..., alibi_slopes=..., sliding_window=..., kv_cache_dtype=..., logits_soft_cap=..., attn_type=..., kv_sharing_target_layer_name=...)`** | Construction; call `super().__init__` behavior is via **`AttentionImplBase.__new__`** (sets DCP/PCP ranks). |
| **`forward(layer, query, key, value, kv_cache, attn_metadata, output, output_scale=..., output_block_scale=...)`** | Main compute; writes into **`output`**. |

**Class / instance attributes expected by base:**

- **`kv_cache_dtype: str`**
- **`num_heads`, `head_size`, `scale`** (set in `__init__`)

**Optional hooks (have defaults):**

- **`process_weights_after_loading`**, **`fused_output_quant_supported`**, **`fused_rope_kvcache_supported`**, **`do_rope_and_kv_cache_update`** — only if you integrate fusion / RoPE paths.

**MLA / sparse MLA paths:** use **`MLAAttentionImpl`** or **`SparseMLAAttentionImpl`** instead — different abstract methods (`forward_mha` / `forward_mqa`). This report assumes a standard **`AttentionImpl`** path for a custom KV–focused backend.

### 3.3 `AttentionMetadataBuilder[M]`

Subclass **`AttentionMetadataBuilder[M]`** with **`M`** your metadata dataclass/type.

**Declared `@abstractmethod`:**

| Member | Role |
|--------|------|
| **`__init__(kv_cache_spec, layer_names, vllm_config, device)`** | Must assign `kv_cache_spec`, `layer_names`, `vllm_config`, `device` (base documents this). |
| **`build(common_prefix_len, common_attn_metadata, fast_build=False) -> M`** | Build per-layer / per-call metadata from **`CommonAttentionMetadata`**. |

**Often overridden:**

- **`build_for_cudagraph_capture`**, **`build_for_drafting`**, **`update_block_table`** (only if **`supports_update_block_table`**).
- Class vars: **`_cudagraph_support`**, **`reorder_batch_threshold`**, **`supports_update_block_table`**.

### 3.4 `AttentionMetadata`

Typically a **backend-specific `@dataclass` or typed struct** referenced as generic **`T`** in **`AttentionImpl[T]`** and **`AttentionMetadataBuilder[T]`**. The base **`AttentionMetadata`** class is an empty stub — real metadata is per-backend.

---

## 4. End-to-end checklist (out-of-tree, no vLLM fork)

1. **Packaging:** Declare **`[project.entry-points."vllm.general_plugins"]`** (or setuptools equivalent) mapping a name → **`your_package.register`** (callable).
2. **`register()`:** Call **`register_backend(AttentionBackendEnum.CUSTOM, "your_package.backend.YourBackend")`** or use the **decorator** on the class. Ensure this runs **before** attention resolution in every process (guaranteed if only invoked from this entry point and **`load_general_plugins()`** runs first in that process).
3. **Runtime config:** Set **`attention_config.backend`** to **`CUSTOM`** (CLI: `--attention-backend CUSTOM`).
4. **Implement:** `AttentionBackend` + `AttentionImpl` + `AttentionMetadataBuilder` + metadata type, satisfying abstract methods and **`validate_configuration`** for your target **`kv_cache_dtype`** and flags.
5. **Idempotency:** Keep registration **deterministic**; use guards for any registry that is not naturally overwrite-idempotent.
6. **Tests:** In multi-process tests, reset **`plugins_loaded`** only in controlled test harnesses (see vLLM’s plugin tests); production code should not rely on resetting.

---

## 5. File reference map

| File | Responsibility |
|------|----------------|
| `vllm/plugins/__init__.py` | `load_plugins_by_group`, `load_general_plugins`, `VLLM_PLUGINS`, `plugins_loaded` |
| `vllm/v1/attention/backends/registry.py` | `AttentionBackendEnum`, `_ATTN_OVERRIDES`, `register_backend`, `get_path` / `get_class` |
| `vllm/v1/attention/selector.py` | `get_attn_backend`, `_cached_get_attn_backend`, `AttentionSelectorConfig`, `resolve_obj_by_qualname`, KV layout hook |
| `vllm/v1/attention/backend.py` | `AttentionBackend`, `AttentionImpl`, `AttentionImplBase`, `AttentionMetadataBuilder`, `CommonAttentionMetadata` |
| `vllm/config/attention.py` | `AttentionConfig.backend` parsing (`auto` → `None`, else enum by name) |
| `vllm/platforms/cuda.py` (and siblings) | `get_attn_backend_cls` — validation + `get_path()` return for explicit backend |

---

*Generated from source inspection of the vLLM tree at analysis time; line-level behavior should be re-verified when upgrading vLLM versions.*
