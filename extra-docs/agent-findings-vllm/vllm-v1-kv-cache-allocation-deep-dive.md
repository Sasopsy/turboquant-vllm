# vLLM V1: KV cache allocation, page sizing, and tensor reshape

Technical reference for overriding standard KV cache allocation (e.g. TurboQuant-style / bit-packed caches with custom kernels). Maps **scheduler byte budgets** (`KVCacheSpec.page_size_bytes`), **worker raw allocation** (`KVCacheTensor` → `torch.zeros`), and **runner-side views** (`AttentionBackend.get_kv_cache_shape` / `get_kv_cache_stride_order`).

---

## 1. `page_size_bytes`, `real_page_size_bytes`, and `num_blocks`

### 1.1 Where “page size” is defined

**Class:** `AttentionSpec` — `vllm/v1/kv_cache_interface.py`

- **`real_page_size_bytes`** (concrete on subclasses). Base `AttentionSpec` uses the standard K/V formula:

```python
@property
def real_page_size_bytes(self) -> int:
    return (
        2
        * self.block_size
        * self.num_kv_heads
        * self.head_size
        * get_dtype_size(self.dtype)
    )
```

- **`page_size_bytes`** — starts from `real_page_size_bytes`, then:

  - If `kv_quant_mode.is_per_token_head`: adds  
    `2 * block_size * num_kv_heads * get_dtype_size(torch.float32)`  
    (budget for K/V per-token-head scale storage carved from / packed with the cache; see §2).

  - If `page_size_padded is not None`: returns `page_size_padded` (must be ≥ computed size).

**TurboQuant / non–`dtype_size` slots:** `TQFullAttentionSpec(FullAttentionSpec)` overrides **`real_page_size_bytes`** so the byte budget is **`block_size * num_kv_heads * tq_slot_size`** (bit-packed slot per head per token), not `head_size * get_dtype_size(dtype)`:

```python
@property
def real_page_size_bytes(self) -> int:
    if self.tq_slot_size > 0:
        return self.block_size * self.num_kv_heads * self.tq_slot_size
    return super().real_page_size_bytes
```

**Wiring from the model:** `Attention.get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec` returns `TQFullAttentionSpec(..., tq_slot_size=tq_config.slot_size_aligned)` when `kv_cache_dtype.startswith("turboquant_")`, where `slot_size_aligned` comes from `TurboQuantConfig` (packed key + value + alignment) — the same quantity the backend uses for tensor shape (§3).

### 1.2 How `num_blocks` is computed from available memory

**Module:** `vllm/v1/core/kv_cache_utils.py`

- **`get_num_blocks(vllm_config: VllmConfig, num_layers: int, available_memory: int, page_size: int) -> int`**

  - `num_blocks = int(available_memory // page_size // num_layers)`
  - `num_blocks = max(num_blocks, 0)`
  - **`may_override_num_blocks(vllm_config, num_blocks)`** — if `CacheConfig.num_gpu_blocks_override` is set, replaces `num_blocks`.

- **`get_uniform_page_size(kv_cache_specs: Iterable[KVCacheSpec]) -> int`** — asserts all merged group specs share one **`page_size_bytes`**.

**Config assembly:** **`get_kv_cache_config_from_groups(vllm_config, kv_cache_groups, available_memory) -> KVCacheConfig`**

- **No groups (attention-free):** `num_blocks=1`, empty tensors.

- **`UniformTypeKVCacheSpecs` (one group, per-layer different hidden size but same “type”):**  
  `num_blocks = available_memory // kv_cache_groups[0].kv_cache_spec.page_size_bytes`  
  where `UniformTypeKVCacheSpecs.page_size_bytes` is **`sum(spec.page_size_bytes for each layer)`** — then each `KVCacheTensor.size = per_layer_spec.page_size_bytes * num_blocks`.

- **General case (uniform physical page per layer across groups):**  
  `group_size = max(len(g.layer_names) for g in kv_cache_groups)`  
  `page_size = get_uniform_page_size([group.kv_cache_spec for group in kv_cache_groups])`  
  `num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)`  
  For each slot `i` in `0 .. group_size-1`, builds **`KVCacheTensor(size=page_size * num_blocks, shared_by=[...])`**.

**Cross-worker:** **`get_kv_cache_configs(...)`** builds per-worker configs, then sets every worker’s **`num_blocks`** to **`min(...)`** and shrinks each **`KVCacheTensor.size`** proportionally.

**Related config:** `CacheConfig` — `vllm/config/cache.py` — **`num_gpu_blocks_override`**, **`kv_cache_memory_bytes`** (bypasses utilization-based KV sizing on the worker), **`block_size`**, **`cache_dtype`**, etc.

---

## 2. Metadata allocation: scales, zero-points, TurboQuant side buffers

### 2.1 Budgeted in `page_size_bytes` (same raw `KVCacheTensor` blob)

| Mechanism | Where budgeted | Where it lives physically |
|-----------|------------------|---------------------------|
| **Per-token-head K/V scales** (`int8_per_token_head`, `fp8_per_token_head`) | `AttentionSpec.page_size_bytes` adds `2 × block × heads × 4` bytes | Backend-specific: e.g. Triton carves **`float32` strided views** from a **padded head dimension** inside the reshaped KV tensor (`TritonAttentionImpl._ensure_scale_caches`), aligned with the extra bytes in the spec. |
| **TurboQuant packed value scale + zero** | Included in **`TurboQuantConfig.value_packed_size`** (`+ 4` bytes: two `fp16`), hence in **`slot_size` / `slot_size_aligned`**, hence in **`TQFullAttentionSpec.real_page_size_bytes`** | Inside each **slot** as part of the packed layout; not a separate parallel tensor at the allocator level. |
| **Optional padding** | `AttentionSpec.page_size_padded` / `MambaSpec.page_size_padded` | Extra bytes in the same raw allocation. |

### 2.2 Not in `page_size_bytes` — separate module buffers

**`Attention._init_turboquant_buffers(self, cache_dtype: str, head_size: int, prefix: str) -> None`** — `vllm/model_executor/layers/attention/attention.py`

- Registers **`_tq_signs`**, **`_tq_centroids`**, **`_tq_config`**.
- Pre-allocates decode workspace buffers **`_tq_mid_o_buf`**, **`_tq_output_buf`**, **`_tq_lse_buf`** so memory profiling reserves space **before** KV blocks consume all free memory.

These are **not** part of **`KVCacheTensor`** / block pool sizing.

**Implication for a Tile Lang / TurboQuant-style plugin**

- **Bit-packed KV payload:** fold into **`real_page_size_bytes`** (pattern: `TQFullAttentionSpec` + matching **`get_kv_cache_shape`**).
- **Per-block or per-slot metadata** that must be contiguous with KV for kernels: include in **slot or `page_size_padded`**.
- **Global centroids / rotation / scratch:** use **`Attention`** buffers or explicit non–KV-cache allocations, not **`page_size_bytes`**, unless you intentionally want the block allocator to account for them.

---

## 3. Raw bytes → shaped tensors: `GPUModelRunner` and backends

### 3.1 Worker entry

**`Worker.initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None`** — `vllm/v1/worker/gpu_worker.py`

- Sets **`self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks`**.
- Calls **`self.model_runner.initialize_kv_cache(kv_cache_config)`** (optionally under CuMem pool `tag="kv_cache"`).

### 3.2 `GPUModelRunner.initialize_kv_cache`

**`initialize_kv_cache(self, kv_cache_config: KVCacheConfig, is_profiling: bool = False) -> None`** — `vllm/v1/worker/gpu_model_runner.py`

1. Deep-copies config; may inject encoder-only / KV-sharing groups.
2. **`initialize_attn_backend(kv_cache_config, ...)`** — builds **`AttentionGroup`** list (backend class per group).
3. **`kernel_block_sizes = prepare_kernel_block_sizes(kv_cache_config, self.attn_groups)`** — `vllm/v1/worker/utils.py` — may **subdivide** scheduler `block_size` to a **kernel** `kernel_block_size` supported by all backends in that group (`select_common_block_size`).
4. **`initialize_metadata_builders`**, optional input-batch refresh.
5. **`kv_caches = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)`**.
6. KV transfer registration, etc.

### 3.3 Raw allocation: `_allocate_kv_cache_tensors`

**`_allocate_kv_cache_tensors(self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]`**

- For each **`KVCacheTensor`**, one **`torch.zeros(size, dtype=torch.int8, device=self.device)`**.
- Every layer in **`shared_by`** gets a **reference to the same tensor** in the dict.

The engine allocates **opaque byte storage** first; logical dtype is applied when **viewing**.

### 3.4 Reshape: `_reshape_kv_cache_tensors`

**`_reshape_kv_cache_tensors(self, kv_cache_config, kv_cache_raw_tensors, kernel_block_sizes) -> dict[str, torch.Tensor]`** — for each attention layer:

1. `assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0`
2. `num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes`
3. `num_blocks_per_kv_block = kv_cache_spec.block_size // kernel_block_size`
4. `kernel_num_blocks = num_blocks * num_blocks_per_kv_block` — virtual split so each scheduler block maps to multiple kernel blocks.
5. `kv_cache_shape = attn_backend.get_kv_cache_shape(kernel_num_blocks, kernel_block_size, num_kv_heads, head_size, cache_dtype_str=self.cache_config.cache_dtype)`
6. `dtype = kv_cache_spec.dtype`
7. Stride order: try **`attn_backend.get_kv_cache_stride_order()`**; on failure use identity **`tuple(range(len(kv_cache_shape)))`**.
8. Build view: **`raw_tensor.view(dtype).view(physical_shape).permute(*inv_order)`** so the exposed tensor has the **backend’s logical shape** while **physical** layout follows stride order when implemented.

**`TurboQuantAttentionBackend.get_kv_cache_shape`** — `vllm/v1/attention/backends/turboquant_attn.py` — returns  
`(num_blocks, block_size, num_kv_heads, tq_config.slot_size_aligned)` — **no leading `2` for K/V**, matching **`TQFullAttentionSpec`** byte count. It does **not** override **`get_kv_cache_stride_order`**, so layout matches logical shape.

**`AttentionBackend.get_kv_cache_block_dim`** — `vllm/v1/attention/backend.py` — discovers which dimension is **`num_blocks`** via a sentinel shape — used for hybrid layout fixes and KV zeroing.

### 3.5 Binding to modules

**`bind_kv_cache(kv_caches, forward_context, runner_kv_caches, num_attn_module=1)`** — `vllm/v1/worker/utils.py`

- Fills **`runner_kv_caches`** in **layer index order**.
- Sets **`forward_context[layer_name].kv_cache = kv_cache`** for each **`Attention`** module.

---

## 4. Checklist for an out-of-tree TurboQuant-style plugin

1. **`KVCacheSpec`:** Implement **`page_size_bytes`** / **`real_page_size_bytes`** (and **`max_memory_usage_bytes`**) so scheduler **`num_blocks`** and **`KVCacheTensor.size`** match your packed layout.
2. **`Attention.get_kv_cache_spec`:** Return your spec with **`block_size`** from **`vllm_config.cache_config.block_size`**.
3. **`AttentionBackend.get_kv_cache_shape`:** Same **`num_blocks × block × heads × …`** element count as **`page_size_bytes`** under **`kv_cache_spec.dtype`** (after **`view(dtype)`**).
4. **`get_kv_cache_stride_order`:** Only if physical dim order must differ from logical (see **`FlashAttentionBackend`** pattern in the codebase).
5. **Metadata:** Put **block-scoped** bytes in **page/slot**; put **static or scratch** data in **`Attention` buffers** or separate allocations, following TurboQuant’s split.

---

## 5. Key source files

| Concern | Path |
|--------|------|
| Block counts, `KVCacheTensor`, grouping | `vllm/v1/core/kv_cache_utils.py` |
| Specs, `page_size_bytes`, `TQFullAttentionSpec` | `vllm/v1/kv_cache_interface.py` |
| Worker init, `initialize_from_config` | `vllm/v1/worker/gpu_worker.py` |
| `bind_kv_cache`, `prepare_kernel_block_sizes` | `vllm/v1/worker/utils.py` |
| `CacheConfig` | `vllm/config/cache.py` |
| Allocate + reshape KV | `vllm/v1/worker/gpu_model_runner.py` (`_allocate_kv_cache_tensors`, `_reshape_kv_cache_tensors`, `initialize_kv_cache_tensors`) |
| Backend shape / stride contract | `vllm/v1/attention/backend.py` |
| TurboQuant backend example | `vllm/v1/attention/backends/turboquant_attn.py` |
| `Attention` KV spec + TQ buffers | `vllm/model_executor/layers/attention/attention.py` |
| TQ slot math | `vllm/model_executor/layers/quantization/turboquant/config.py` |
