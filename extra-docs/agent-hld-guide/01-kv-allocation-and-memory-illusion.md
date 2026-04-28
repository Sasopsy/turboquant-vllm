# Agent reference: KV allocation & memory illusion (vLLM V1)

**Audience:** LLM drafting HLD/LLD for TurboQuant-style KV extensions. **Scope:** How VRAM is budgeted, grouped, allocated as opaque bytes, then `.view()`’d into backend shapes.

---

## Mental model

- vLLM = paged KV “OS”: logical seq KV ↔ physical `block_size`-token blocks; `block_table` maps seq→physical block IDs; one large prealloc avoids fragmentation, unpredictable OOM, and CUDA-graph address churn.
- **Memory illusion:** allocate flat `torch.int8` blobs; reshape via `dtype` view + `get_kv_cache_shape` + optional stride permutation. Same bytes, backend-defined interpretation.

---

## Spec hierarchy (`vllm/v1/kv_cache_interface.py`)

| Type | Role |
|------|------|
| `KVCacheSpec` | Abstract: `block_size`, `page_size_bytes`, `max_memory_usage_bytes` |
| `AttentionSpec` | `num_kv_heads`, `head_size`, `dtype`, `kv_quant_mode`, optional `page_size_padded`. `real_page_size_bytes` = K+V raw; `page_size_bytes` adds per-token-head scale storage if `KVQuantMode.is_per_token_head`, else padding |
| `FullAttentionSpec` | Asymmetric K/V: `head_size_v`; `real_page_size_bytes` = `block_size * num_kv_heads * (head_size + head_size_v) * dtype_size` |
| `TQFullAttentionSpec` | `tq_slot_size` (bytes/token/head packed K+V+metadata): `real_page_size_bytes` = `block_size * num_kv_heads * tq_slot_size` — **no `2*`** (slot packs both) |
| `KVQuantMode` | `NONE`, `FP8_PER_TENSOR`, `INT8_PER_TOKEN_HEAD`, `FP8_PER_TOKEN_HEAD`; `>=2` ⇒ extra float32 scales in page budget |

---

## Profiler → available KV bytes

- `GPUWorker.determine_available_memory` (`vllm/v1/worker/gpu_worker.py`): load model, dummy forward (`profile_run`), peak alloc, optional cudagraph memory; derive `available_memory` after weights/overhead; respect `gpu_memory_utilization` or `CacheConfig.kv_cache_memory_bytes`.
- **TQ/global buffers:** anything not in the block pool (centroids, signs, decode scratch) must be **`register_buffer` before profile** so peak includes them; else KV allocator over-commits → inference OOM.

---

## Groups & `KVCacheConfig` (`vllm/v1/core/kv_cache_utils.py`)

- `get_kv_cache_groups()`: layers with same spec + shared block table → one group; hybrid models → multiple groups (e.g. full-attn vs sliding-window).
- `get_kv_cache_config_from_groups`:
  - `group_size = max(len(g.layer_names) for g in kv_cache_groups)` = **number of raw `KVCacheTensor` blobs**.
  - `page_size = get_uniform_page_size(...)` — all groups share one `page_size_bytes`.
  - `num_blocks = get_num_blocks(vllm_config, group_size, available_memory, page_size)` = `available_memory // page_size // group_size` (then `may_override_num_blocks`).
- **Tensor ↔ layers:** `kv_cache_tensors[i]` has `shared_by` = one layer name from **each** group at index `i` (transpose of groups×layers). Hybrid example: 10 blobs, each shared by 3 layers from 3 groups — separate block tables per group so no index collision.
- Standard 32-layer single group: `group_size=32` → 32 blobs, each `shared_by` one layer.

---

## `KVCacheTensor`

- `size = page_size_bytes * num_blocks` (bytes as int8 element count).
- `shared_by`: layer names referencing **same** raw tensor object.

---

## TQ vs dense page math (illustrative)

- Dense bf16 Llama-8B-ish: `real_page_size` ∝ `2 * block * heads * head_dim * 2`.
- TQ: `block * heads * tq_slot`; smaller page ⇒ **more blocks** for same VRAM ⇒ longer context / more concurrent seqs. Total KV bytes ≈ `available_memory` cap either way; gain is **block count**, not “free” memory.

---

## Lifecycle (allocation → bind)

1. **Raw alloc:** `_allocate_kv_cache_tensors` (`gpu_model_runner.py`): `torch.zeros(size, int8, device)` per `KVCacheTensor`; duplicate refs in `kv_cache_raw_tensors[layer_name]`.
2. **Kernel block split:** `prepare_kernel_block_sizes`: may subdivide scheduler `block_size` → `kernel_block_size`, `kernel_num_blocks = num_blocks * (block_size // kernel_block_size)`. TQ default: `MultipleOf(1)` → no split.
3. **Reshape:** `_reshape_kv_cache_tensors`:
   - `num_blocks = raw.numel() // kv_cache_spec.page_size_bytes`
   - `kv_cache_shape = attn_backend.get_kv_cache_shape(kernel_num_blocks, kernel_block_size, num_kv_heads, head_size, cache_dtype_str)`
   - Optional `get_kv_cache_stride_order` → permute dims (Flash: swap leading `2` vs `num_blocks`; TQ: identity).
   - Chain: `raw.view(dtype).view(physical_shape).permute(inv_order)`.
4. **Bind:** `bind_kv_cache` (`vllm/v1/worker/utils.py`): `forward_context[layer].kv_cache = tensor`; append to `runner_kv_caches` by layer index.

---

## Invariant (must hold)

`prod(get_kv_cache_shape(...)) == raw_tensor.numel() // element_size(spec.dtype)`

- TQ pattern: `spec.dtype = int8`, shape `(num_blocks, block_size, num_kv_heads, slot_size_aligned)`; `page_size_bytes = block_size * num_kv_heads * slot_size`.

---

## Backend shape contrast

| Backend | Typical `get_kv_cache_shape` | Stride order |
|---------|------------------------------|--------------|
| FlashAttention | `(2, num_blocks, block_size, num_kv_heads, head_dim)` | `(1,0,2,3,4)` — physical `num_blocks` outer |
| TurboQuant | `(num_blocks, block_size, num_kv_heads, slot_size)` | identity (default) |

`TurboQuantAttentionBackend.get_kv_cache_shape` reads `tq_config.slot_size_aligned` from `get_current_vllm_config().quant_config.tq_config`.

---

## Metadata placement

| Lifetime | Placement | Mechanism |
|----------|-----------|-----------|
| Per token/head (scales, packed bits) | Inside slot / page | `tq_slot_size` / `page_size_bytes` |
| Per-token-head scales (built-in modes) | Extra bytes in page | `kv_quant_mode.is_per_token_head` |
| Global codebook, signs, workspaces | Outside pool | `Attention._init_turboquant_buffers` / `register_buffer` — **before profile** |

---

## Store kernel addressing (for LLD)

- `slot_mapping[i]`: linear physical slot; `slot < 0` ⇒ skip write.
- `block_idx = slot // block_size`, `offset = slot % block_size` (conceptually; actual strides depend on layout).

---

## Key files

| Area | Path |
|------|------|
| Specs | `vllm/v1/kv_cache_interface.py` |
| `num_blocks`, groups | `vllm/v1/core/kv_cache_utils.py` |
| Alloc/reshape | `vllm/v1/worker/gpu_model_runner.py` |
| Bind, kernel sizes | `vllm/v1/worker/utils.py` |
| TQ backend shape | `vllm/v1/attention/backends/turboquant_attn.py` |
| TQ slot config | `vllm/model_executor/layers/quantization/turboquant/config.py` |
| Attention buffers | `vllm/model_executor/layers/attention/attention.py` |

---

## TurboQuant algorithm (offline / math — from `turboquant_algorithm.md`)

- **Induced coord density** on `[-1,1]` for unit vectors after random orthogonal rotation; **Lloyd–Max** codebook `C` for `b`-bit scalars minimizing MSE against that density.
- **Algorithm 1 (`TurboQuant_mse`):** random orthogonal `Π`; codebook `C_mse`; quant: `y=Πx`, per-dim nearest centroid index; dequant: inverse map + `Πᵀ`.
- **Algorithm 2 (`TurboQuant_prod`):** `(b-1)`-bit MSE quant + 1-bit QJL on residual: `idx = Quant_mse(x)`, `r = x - DeQuant_mse(idx)`, `qjl = sign(S r)`, `γ = ||r||`; dequant combines MSE reconstruction + scaled `Sᵀ qjl` for inner-product-oriented error.

---

## Debug quick refs

- View error: mismatch `prod(shape)` vs bytes/`dtype_size`.
- Inference OOM: global buffers registered after profile.
- Garbage attn: shape matches but store/read layout disagree — use sentinels in slots.

---

## Plugin checklist (allocation side)

- Custom spec: override `real_page_size_bytes` for packed slot; `dtype=int8` if slot is byte-granular.
- `get_kv_cache_shape` must match `page_size_bytes` and `num_blocks` math.
- `merge()` on spec: consistent `slot_size` across layers or split groups.
