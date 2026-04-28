# LLD: C4 + C5 — Attention Backend, Metadata Builder & Attention Implementation

**Document type:** LLD (Low-Level Design)  
**Components:** C4 — `backend/backend.py` + `backend/metadata.py`, C5 — `backend/impl.py`  
**HLD reference:** §5 C5, §5 C6, §7 Global Buffer Lifecycle, §11 Open Questions #4, #5  
**Depends on:** C1 (`TileLangTQConfig`, `TileLangTQAttentionSpec`), C2 (offline artifacts), C3 (quant config and post-load buffers)  
**Status:** Canonical plugin-owned backend/runtime contract; valid for pre-TurboQuant `vLLM` trees

---

## 1. Purpose & Scope

C4 defines how the plugin exposes itself as an attention backend and how per-step metadata is built from scheduler output.

C5 defines how the backend implementation:

- stores new K/V rows into the packed TQ cache
- dispatches decode / prefill / mixed batches
- consumes the buffers created in C3

These two components are documented together because:

- C4 metadata fields exist only to serve C5 branches
- C5 correctness depends on the exact batch partitioning guarantees that C4 establishes

---

## 2. Compatibility Rules

This document must remain valid on older `vLLM` branches. Therefore:

- do not assume upstream TurboQuant backend classes exist
- do not assume the exact current-tree `AttentionBackend` or `AttentionMetadataBuilder` hook surface is identical everywhere
- define the canonical behavior in terms of required capabilities, and let the plugin adapter map that behavior onto each branch

Non-negotiable runtime invariants:

- the backend must treat `tilelang_tq_3bit` and `tilelang_tq_4bit` as the canonical cache-dtype literals after normalization
- the backend must use the plugin-local spec and packed-cache shape defined in C1
- store and attention run as separate steps: `forward_includes_kv_cache_update = False`
- `slot_mapping.shape[0]` is the authoritative row bound for store
- `slot_mapping[i] < 0` means “skip store for this row”
- no first-use GPU allocation is allowed in `do_kv_cache_update` or `forward`

---

## 3. Open Questions Resolved

From HLD §11:

| Question | Decision |
|---|---|
| `max_num_kv_splits` static or dynamic? | Static per impl instance, derived from runtime config at init; use a plugin-owned default if a branch lacks the newer knob |
| continuation threshold | Keep `_CONTINUATION_DECODE_THRESHOLD = 128` unless later profiling proves a better universal cutoff |

---

## 4. C4 — `TileLangTQAttentionBackend`

**File:** `backend/backend.py`

The canonical backend class advertises a split-KV-update attention backend that consumes the packed cache layout from C1.

### 4.1 Core class attributes

Canonical values:

| Attribute | Value | Why |
|---|---|---|
| `accept_output_buffer` | `True` | lets the runner supply a preallocated output tensor; important for CUDA graph and no-allocation execution |
| `forward_includes_kv_cache_update` | `False` | store happens in a separate step before attention |
| `supported_dtypes` | `[torch.float16, torch.bfloat16]` | query/key/value activations arrive as fp16/bf16 |
| `supported_kv_cache_dtypes` | `["tilelang_tq_3bit", "tilelang_tq_4bit"]` | canonical normalized plugin literals only |

`"auto"` is intentionally not a supported steady-state cache dtype for this backend. If the plugin accepts a user alias or shorthand, C3 must normalize it before backend selection.

### 4.2 Required backend methods

Canonical backend responsibilities:

```python
class TileLangTQAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str: ...

    @staticmethod
    def get_impl_cls() -> type["TileLangTQAttentionImpl"]: ...

    @staticmethod
    def get_builder_cls() -> type["TileLangTQMetadataBuilder"]: ...

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]: ...

    @staticmethod
    def get_kv_cache_shape(...) -> tuple[int, int, int, int]: ...
```

### 4.3 `get_name`

Canonical backend name:

```python
@staticmethod
def get_name() -> str:
    return "TILELANG_TQ"
```

The user still selects the backend via `attention_backend="CUSTOM"` or the branch-equivalent registration handle. The backend’s internal name is only for diagnostics and registry identity.

### 4.4 `get_supported_kernel_block_sizes`

Preferred supported kernel block sizes:

```python
[16, 32, 64, 128]
```

This keeps the design aligned with the HLD and with common paged-attention scheduling assumptions. If a target branch or kernel implementation only supports a subset, the adapter must narrow the list rather than silently accepting unsupported block sizes.

### 4.5 `get_kv_cache_shape`

The backend must delegate shape semantics to the C1 contract:

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

There is no leading `2` dimension because K and V are packed into the same slot.

### 4.6 Capability methods

Canonical answers:

| Method | Value |
|---|---|
| `supports_attn_type` | decoder-only |
| `supports_per_head_quant_scales` | `False` |
| `is_mla` | `False` |
| `supports_sink` | `False` |
| `supports_mm_prefix` | `False` |
| `supports_non_causal` | `False` unless explicitly implemented later |

### 4.7 `supports_kv_cache_dtype`

The canonical backend should accept only normalized plugin literals:

```python
@classmethod
def supports_kv_cache_dtype(cls, kv_cache_dtype: str | None) -> bool:
    if kv_cache_dtype is None:
        return False
    return normalize_cache_dtype(kv_cache_dtype) in cls.supported_kv_cache_dtypes
```

If a branch’s base class assumes `None` means “don’t care”, the adapter may keep that behavior at the outer selection layer, but the TQ backend itself should still reject non-TQ cache dtypes.

### 4.8 Validation contract

The backend must reject unsupported configurations before execution.

Preferred hook:

```python
@classmethod
def validate_configuration(...) -> list[str]:
    ...
```

If a branch lacks this exact method, the adapter must enforce the same checks in the branch’s equivalent selection path.

Minimum validation set:

- activation dtype is supported
- cache dtype resolves to one of the plugin literals
- block size is compatible with `get_supported_kernel_block_sizes()`
- variant exists in the plugin registry
- attention type is decoder
- runtime rotation mode is valid for the requested `head_size`
- target device capability meets the TileLang kernel requirement

Important correction relative to the old draft:

- non-power-of-two `head_size` is not automatically invalid
- it is valid when the configured rotation mode is `qr`
- it is valid for `randomized_hadamard` only if explicit, tested padding support is enabled

So validation must be conditional on the chosen rotation mode from C2, not a blanket head-size restriction.

### 4.9 Backend selection rule

The HLD is explicit: this backend is not automatically inferred from cache dtype alone.

Canonical rule:

- user or adapter must explicitly select the custom backend registration path
- there is no required auto-promotion from generic backends to TILELANG_TQ

---

## 5. C4 — `TileLangTQMetadata`

**File:** `backend/metadata.py`

This metadata is the per-step contract C5 consumes.

Canonical fields:

| Field | Type | Meaning |
|---|---|---|
| `seq_lens` | device tensor | total context length per request |
| `slot_mapping` | device tensor | cache slot index per new token row |
| `block_table` | device tensor | physical block IDs per request |
| `query_start_loc` | device tensor | cumulative query starts for varlen prefill |
| `query_start_loc_host` | host-native sequence | CPU-side copy used for request-local slicing without per-request device sync |
| `num_actual_tokens` | `int` | attention-path token count; may be padded in full-padding / CUDA-graph modes |
| `max_query_len` | `int` | longest query length in the batch |
| `max_seq_len` | `int` | longest context length relevant to the current metadata slice |
| `is_prefill` | `bool` | whether the batch contains any prefill work |
| `num_decodes` | `int` | number of decode requests at the front of the reordered batch |
| `num_decode_tokens` | `int` | number of tokens belonging to the decode prefix |

Notes:

- `query_start_loc_host` may be a Python list or a CPU tensor depending on branch adapter preference
- `num_actual_tokens` is not the store-path row bound
- `max_seq_len` must be re-based for sub-batches when mixed decode/prefill dispatch slices metadata

---

## 6. C4 — `TileLangTQMetadataBuilder`

**File:** `backend/metadata.py`

The builder turns common runner metadata into backend-specific metadata with one critical guarantee:

- decode requests appear first in the batch
- prefill requests follow contiguously

### 6.1 Reorder rule

Preferred initialization:

```python
self._init_reorder_batch_threshold(
    reorder_batch_threshold=1,
    supports_spec_as_decode=False,
)
```

This means:

- true single-token decode requests are pulled to the front
- multi-token continuation/prefill requests remain in the tail

If a branch lacks `_init_reorder_batch_threshold`, the adapter must achieve the same effective batch partitioning by its branch-specific reorder hook.

### 6.2 CUDA graph support

Preferred support level:

```python
AttentionCGSupport.UNIFORM_BATCH
```

Compatibility note:

- if an older branch exposes a smaller enum surface, the adapter may degrade to the nearest equivalent support level
- the important behavioral rule is that graph capture is only promised for uniform-query-length batches the backend actually supports

### 6.3 `build()`

Canonical builder flow:

1. receive common attention metadata from the runner
2. compute decode/prefill split using the branch’s CPU-safe helper or equivalent logic
3. create one host-native copy of `query_start_loc`
4. populate `TileLangTQMetadata`

Preferred sketch:

```python
def build(
    self,
    common_prefix_len: int,
    common_attn_metadata: CommonAttentionMetadata,
    fast_build: bool = False,
) -> TileLangTQMetadata:
    cam = common_attn_metadata
    num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
        cam,
        decode_threshold=self.reorder_batch_threshold,
    )

    qsl_host = _to_host_query_start_loc(cam)

    return TileLangTQMetadata(
        seq_lens=cam.seq_lens,
        slot_mapping=cam.slot_mapping,
        block_table=cam.block_table_tensor,
        query_start_loc=cam.query_start_loc,
        query_start_loc_host=qsl_host,
        num_actual_tokens=cam.num_actual_tokens,
        max_query_len=cam.max_query_len,
        max_seq_len=cam.max_seq_len,
        is_prefill=(cam.max_query_len > 1),
        num_decodes=num_decodes,
        num_decode_tokens=num_decode_tokens,
    )
```

### 6.4 `build_for_cudagraph_capture()`

Canonical behavior:

- reuse `build()`
- replace `seq_lens` with a small constant value such as `1`
- do not allocate a separate structural metadata layout for capture

This matches the HLD’s intent: capture should be cheap, and replay patches the real sequence lengths into the same storage.

---

## 7. C5 — `TileLangTQAttentionImpl`

**File:** `backend/impl.py`

This class consumes C3-registered buffers and C4 metadata to implement the runtime path.

### 7.1 Initialization

Canonical impl state:

```python
class TileLangTQAttentionImpl(AttentionImpl[TileLangTQMetadata]):
    supports_quant_query_input = False

    def __init__(..., kv_cache_dtype: str, ...):
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = normalize_cache_dtype(kv_cache_dtype)
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name

        variant = get_variant_by_dtype_str(self.kv_cache_dtype).name
        self.tq_config = TileLangTQConfig.from_variant_name(variant, head_size)
        self.fa_version = _detect_flash_attention_version_if_available(head_size)
        self.max_num_kv_splits = _get_static_kv_split_count_from_runtime()
        self._continuation_decode_threshold = 128
```

Rules:

- config derivation is from normalized cache dtype, not from quantization name alone
- `max_num_kv_splits` is static for the life of the impl instance
- if a branch lacks the newer config field, the plugin must use a documented default

### 7.2 `process_weights_after_loading(act_dtype)`

Default canonical behavior: no-op.

This hook exists for impl-level kernel prep such as:

- pre-transposing rotation matrices
- repacking matrix layout for a specific TileLang kernel
- validating that registered buffers match the selected activation dtype

Any additional tensors created here must still satisfy the same lifecycle rule:

- they must exist before `profile_run()`
- they must not first appear in `forward`

### 7.3 `do_kv_cache_update()`

Canonical signature:

```python
def do_kv_cache_update(
    self,
    layer,
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    ...
```

Canonical behavior:

- read `N = slot_mapping.shape[0]`
- if `N == 0`, return immediately
- do not slice by `num_actual_tokens`
- do not filter padding rows in Python
- invoke the fused store kernel with layer buffers from C3

Preferred call shape:

```python
tl_turboquant_store(
    key=key,
    value=value,
    kv_cache=kv_cache,
    slot_mapping=slot_mapping,
    rotation=layer._tq_rotation,
    rotation_t=getattr(layer, "_tq_rotation_t", layer._tq_rotation),
    key_midpoints=layer._tq_key_midpoints,
    key_centroids=layer._tq_key_centroids,
    value_midpoints=layer._tq_value_midpoints,
    value_centroids=layer._tq_value_centroids,
    S_matrix=layer._tq_S_matrix,
    config=self.tq_config,
)
```

Store-path rules:

- `slot_mapping[i] < 0` must be handled inside the kernel
- key already has RoPE applied when it reaches this method
- value does not
- the method must not allocate device tensors or migrate buffers

### 7.4 Top-level `forward()`

Canonical responsibilities:

1. respect an optional preallocated output buffer
2. return a zeroed result when metadata is absent or no tokens are active
3. dispatch to one of:
   - pure decode
   - pure prefill
   - mixed decode+prefill

Preferred high-level structure:

```python
def forward(..., attn_metadata, output=None, ...):
    if output is None:
        output = _allocate_output_like_query(...)

    if attn_metadata is None:
        return output.zero_()

    N = attn_metadata.num_actual_tokens
    if N <= 0:
        return output.zero_()

    q = _reshape_query_prefix(query, N)

    if not attn_metadata.is_prefill:
        attn_out = self._decode_attention(...)
    elif attn_metadata.num_decodes == 0:
        attn_out = self._prefill_attention(...)
    else:
        attn_out = self._mixed_attention(...)

    return _write_back(attn_out, output, N)
```

### 7.5 Pure decode path

When `is_prefill == False`, the whole active batch is decode-only.

Canonical decode path:

- read cached K/V only
- use the TileLang decode kernel
- pass the preallocated scratch buffers from the layer

Required inputs:

- query slice
- `kv_cache`
- `block_table`
- `seq_lens`
- runtime matrices and centroids
- `mid_o_buf`, `lse_buf`, `output_buf`

### 7.6 Pure prefill path

Prefill has two subcases:

#### First-chunk prefill

When:

```python
max_query_len == max_seq_len
```

the entire K/V context is in the current chunk, so the backend should use the native varlen FlashAttention path with raw `key` and `value` instead of dequantizing from cache.

#### Continuation prefill

When cached context exists, choose between:

- small continuation: decode-style path for short query lengths
- large continuation: dequantize cached K/V, concatenate with current chunk K/V, then run FlashAttention

Canonical threshold:

```python
_CONTINUATION_DECODE_THRESHOLD = 128
```

### 7.7 Mixed decode + prefill path

Because the metadata builder guarantees decodes first:

- the first `num_decode_tokens` rows belong to the decode prefix
- the remaining active rows belong to prefill requests

Canonical mixed flow:

1. allocate or reuse one output tensor for the active token prefix
2. build decode sub-metadata by slicing the decode prefix
3. run `_decode_attention()` on the prefix
4. build prefill sub-metadata by slicing the tail
5. run `_prefill_attention()` on the tail
6. stitch both outputs into the active prefix

Critical correctness detail:

- the prefill sub-metadata must compute `max_seq_len` from the prefill slice only
- using the full-batch `max_seq_len` suppresses the FlashAttention first-chunk fast path for valid prefills

### 7.8 Metadata slicing helpers

To keep the mixed path correct and readable, C5 should define explicit helpers:

- `_slice_decode_metadata(metadata)`
- `_slice_prefill_metadata(metadata, num_decode_tokens, num_decodes)`

The prefill helper must:

- subtract the decode-token prefix from `query_start_loc`
- slice `seq_lens` and `block_table` to prefill requests only
- recompute prefill-local `max_seq_len`
- keep `num_actual_tokens` equal to the prefill active-token count

### 7.9 Device residency rule

By the time `forward()` or `do_kv_cache_update()` runs, all of these must already be resident on the target device:

- key/value centroids and midpoints
- rotation matrices
- `S_matrix`
- decode scratch buffers

The impl may validate shapes or dtypes, but it must not:

- call a lazy `_ensure_on_device`
- allocate first-use buffers
- `.to(device)` long-lived artifacts on demand

That memory had to be visible before `profile_run()`.

### 7.10 KV sharing rule

If `kv_sharing_target_layer_name is not None`, the outer attention layer may skip the store step entirely.

The impl does not need custom sharing logic beyond:

- tolerating the absence of a `do_kv_cache_update` call for that step
- reading from the provided `kv_cache` normally in `forward`

---

## 8. Testing Contracts

### C4 Unit Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_backend_supports_only_plugin_cache_dtypes` | query backend dtype support | only normalized plugin literals accepted |
| `test_backend_shape_matches_c2` | call `get_kv_cache_shape` | shape matches packed-slot contract |
| `test_validate_configuration_rejects_variant_mismatch` | unsupported dtype/variant | failure |
| `test_validate_configuration_allows_non_power_of_two_in_qr_mode` | `head_size=96`, QR rotation | success |
| `test_validate_configuration_rejects_unsupported_hadamard_shape` | `head_size=96`, Hadamard without padding support | failure |
| `test_metadata_builder_reorders_decode_prefix` | mixed batch | decodes first |
| `test_metadata_builder_build_for_cudagraph_capture` | capture path | `seq_lens` filled with small constant |

### C5 Unit Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_do_kv_cache_update_uses_slot_mapping_height` | padded rows present | store bound equals `slot_mapping.shape[0]` |
| `test_do_kv_cache_update_skips_negative_slots_in_kernel_contract` | negative slots | no write for masked rows |
| `test_forward_decode_path_selected` | `is_prefill=False` | decode path runs |
| `test_forward_first_chunk_prefill_path_selected` | `max_query_len == max_seq_len` | raw FlashAttention path runs |
| `test_forward_continuation_prefill_small_query_uses_decode_style_path` | short continuation | small continuation branch runs |
| `test_forward_continuation_prefill_large_query_uses_dequant_flash_path` | long continuation | dequant+flash branch runs |
| `test_mixed_batch_prefill_max_seq_len_rebased` | mixed batch | prefill submetadata uses prefill-local max |
| `test_forward_no_metadata_returns_zero_output` | `attn_metadata=None` | zero output |

### Integration Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_backend_registered_under_custom_attention_backend` | plugin init | backend resolves correctly |
| `test_store_then_decode_matches_reference_attention_within_tolerance` | single request round-trip | numerical agreement within target tolerance |
| `test_mixed_decode_prefill_batch_matches_split_reference` | mixed batch | output matches split execution |
| `test_no_runtime_gpu_materialization_in_store_or_forward` | memory instrumentation | no first-use long-lived allocations |
| `test_cudagraph_capture_for_supported_uniform_batch` | graph capture path | capture succeeds |

---

## 9. User-Facing Runtime Assumption

Canonical invocation still requires:

```python
LLM(
    ...,
    quantization="tq_3bit",
    kv_cache_dtype="tilelang_tq_3bit",
    attention_backend="CUSTOM",
)
```

If a branch adapter supports extra aliases or automatic normalization, that is a plugin convenience feature, not part of the canonical C4/C5 contract.
