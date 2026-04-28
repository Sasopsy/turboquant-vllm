# Agent reference: split store vs forward, batching, metadata

**Audience:** LLM drafting HLD/LLD for TQ/vLLM attention plugins. **Scope:** Why store and attention are split; tensors and guards; `ForwardContext`; decode/prefill; mixed batches.

---

## Why split execution

1. **Correctness:** Same `kv_cache` buffer read during attention while another warp could be writing new K/V → race. Framework runs **full batch store** then **read/compute**.
2. **CUDA graphs:** Static launch sequences; split `unified_kv_cache_update` → `unified_attention_with_output` matches capture/replay boundaries.

---

## Backend flag

- `forward_includes_kv_cache_update: bool`
  - **`False` (Flash, TQ):** framework calls `do_kv_cache_update` **before** `impl.forward`. `forward` treats KV as read-only.
  - **`True`:** fused path; framework skips separate store.

---

## `Attention.forward` order (`attention.py`)

1. Optional KV scale / query quant.
2. Reshape: `query (T,Hq,D)`, `key/value (T,Hkv,D)` / `D_v`.
3. If `not forward_includes_kv_cache_update` and not KV-sharing and `key/value` present:
   - **Direct path:** `kv_cache_dummy_dep = unified_kv_cache_update(key, value, layer_name)`
   - **Compiled path:** `torch.ops.vllm.unified_kv_cache_update(key, value, encoded_layer)`
4. `unified_attention_with_output(..., kv_cache_dummy_dep=...)` → `impl.forward(..., attn_metadata, output)`.

**KV sharing:** `kv_sharing_target_layer_name is not None` → skip store + dummy dep; read target’s cache.

---

## `unified_kv_cache_update`

- Resolves `get_attention_context(layer_name)` → `attn_layer`, `kv_cache`, `layer_slot_mapping`.
- If `layer_slot_mapping is not None`: assert `do_kv_cache_update` exists; call it.
- Returns `torch.empty(0, device=kv_cache.device, dtype=kv_cache.dtype)` — **dummy tensor** for `torch.compile` **ordering** (`kv_cache_dummy_dep`); compute op `del`’s it but keeps data dependency.

---

## `do_kv_cache_update` contract

```text
(layer, key[T,Hkv,D], value[T,Hkv,Dv], kv_cache, slot_mapping[T]) -> None
```

- **`slot_mapping`:** `int64`, one entry per **row** of K/V after reshape. **`slot < 0` ⇒ must not write** (padding tail, CP pad, spec-decode masks, etc.).
- **Row bound:** use `N = slot_mapping.shape[0]` aligned with runner; padded path: still length `num_tokens_padded`, tail `-1`.
- **`has_separate_kv_update`:** true if any non-encoder group has a backend with `forward_includes_kv_cache_update == False` → runner uses **padded** token count for slot mappings when needed so height matches activations (`gpu_model_runner.py`).

**Padding tail:** `_get_slot_mapping`: fill `slot_mapping[num_tokens_unpadded:num_tokens_padded] = -1`.

---

## Flash vs TQ `kv_cache` layout

- **Flash:** `(2, num_blocks, block_size, num_kv_heads, head_dim)`; `unbind(0)` → K/V; store via `reshape_and_cache_flash` (CUDA/Triton).
- **TQ:** `(num_blocks, block_size, num_kv_heads, slot_size)` fused slot; custom store kernel quantizes/packs.

---

## `ForwardContext` (`forward_context.py`)

- Per-forward stash: `attn_metadata[layer]`, `slot_mapping[layer]`, `no_compile_layers` (layer name → `Attention` module).
- Set by runner: `set_forward_context(..., slot_mapping=slot_mappings, ...)` around full model forward.
- `get_attention_context(layer)` used inside unified KV/attention ops.

---

## `CommonAttentionMetadata` (fields for kernels)

| Field | Shape / type | Role |
|-------|----------------|------|
| `query_start_loc` | `(num_reqs+1,)` int32 GPU | Cumulative Q row starts (prefill / varlen) |
| `query_start_loc_cpu` | CPU copy | Host-side dispatch |
| `seq_lens` | `(num_reqs,)` int32 | Total context len per req (includes just-stored token at decode) |
| `num_reqs`, `num_actual_tokens`, `max_query_len`, `max_seq_len` | scalars | Batch geometry; `num_actual_tokens` may include padding |
| `block_table_tensor` | `(num_reqs, max_blocks)` int32 | Physical block IDs per req |
| `slot_mapping` | `(num_tokens_padded,)` int64 | Store path slots (also in common meta) |

**Decode navigation:** `block_table`, `seq_lens`. **Prefill:** `query_start_loc`, causal over Q rows. **Paged read:** `page_idx = t // block_size`, `off = t % block_size`, `phys = block_table[r, page_idx]`.

---

## Prefill vs decode (TQ-specific behavior)

- **Detect prefill:** e.g. `max_query_len > 1` → `is_prefill`.
- **Pure decode:** Triton `triton_turboquant_decode_attention` on compressed cache.
- **Prefill (`turboquant_attn.py`):** multiple strategies — whole-batch Flash varlen when first chunk; per-request loops; **continuation** (`q_len < seq_len`): cache holds prior tokens → small chunks may reuse **decode Triton** with synthetic `seq_lens`/`block_table`; large → dequant + Flash varlen / SDPA.
- **Mixed batch:** `split_decodes_and_prefills` (CPU); decodes first in Q; run decode on `q[:num_decode_tokens]`, then prefill on remainder with **rebased** `query_start_loc` and **prefill-local** `max_seq_len` (not full-batch max — Flash path constraint).

---

## Metadata builder pattern

- `AttentionMetadataBuilder.build(common_prefix_len, common_attn_metadata, fast_build)` → backend-specific dataclass (e.g. TQ: `seq_lens`, `block_table`, `query_start_loc`, `is_prefill`, decode/prefill counts).
- Optional: `reorder_batch_threshold`, `_cudagraph_support`, `build_for_cudagraph_capture`, spec-decoder hooks.

---

## Implementation checklist (execution)

**Store**

- Mask all writes: `slot >= 0`.
- `slot_mapping` int64 in Triton loads.
- Inputs already 3D; RoPE on **K** only before store; **V** unrotated (see Part 3 for QKV path).

**Forward (decode)**

- Slice `query` per actual/decode rows as metadata requires.
- Read KV via `block_table` + `seq_lens`; do not exceed `seq_lens[r]`; avoid NULL blocks.
- Write **only** to `output`; do not write `kv_cache` in forward (TQ design).

**Forward (prefill)**

- Branch on `is_prefill` / mixed batch like TQ reference; handle continuation and metadata rebasing.

**Backend**

- `forward_includes_kv_cache_update = False` + implement both `do_kv_cache_update` and `forward`.

---

## Key files

| Path | Symbols |
|------|---------|
| `vllm/model_executor/layers/attention/attention.py` | `Attention.forward`, `unified_kv_cache_update`, `unified_attention_with_output`, reshape |
| `vllm/forward_context.py` | `ForwardContext`, `set_forward_context`, `get_attention_context` |
| `vllm/v1/worker/gpu_model_runner.py` | `has_separate_kv_update`, `_get_slot_mapping`, metadata build |
| `vllm/v1/attention/backend.py` | `CommonAttentionMetadata`, builder/impl bases |
| `vllm/v1/attention/backends/flash_attn.py` | Reference store via `reshape_and_cache_flash` |
| `vllm/v1/attention/backends/turboquant_attn.py` | TQ store, decode, prefill, mixed |
| `vllm/v1/worker/utils.py` | `split_decodes_and_prefills` |
