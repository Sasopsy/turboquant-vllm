# LLD: C6 + C7 — TileLang Kernels and Plugin Registration

**Document type:** LLD (Low-Level Design)  
**Components:** C6 — `kernels/tl_store.py` + `kernels/tl_decode.py`, C7 — `plugin.py` + packaging entry point  
**HLD reference:** §5 C7, §5 C8, §6 Slot Layout Reference, §7 Global Buffer Lifecycle  
**Depends on:** C1 (`TileLangTQConfig`, `SlotLayout`, packed KV spec), C2 (offline artifacts), C3 (registered layer buffers), C4-C5 (backend and impl contracts)  
**Status:** Canonical plugin-owned kernel and registration contract for pre-TurboQuant `vLLM` trees

---

## 1. Scope

C6 defines the runtime kernel contracts for:

- writing compressed K/V into the packed cache
- reading that cache during decode / continuation paths

C7 defines how the plugin is discovered and how it registers all required shims into each `vLLM` process.

This document intentionally separates:

- canonical mathematical/runtime behavior
- branch-specific adapter work

That distinction matters because older target `vLLM` branches may not expose the same plugin, backend, or dtype-extension hooks as the current local tree.

---

## 2. C6 Shared Kernel Contract

Both kernels must agree on four things:

1. the slot layout from C1
2. the runtime buffers registered in C3
3. the corrected split-KV store semantics from C5
4. the algorithm choice that both keys and values use the same TurboQuant-style MSE+QJL structure

### 2.1 Slot layout source of truth

Neither kernel may hard-code independent byte offsets.

Canonical rule:

- import `SlotLayout` or equivalent offset constants derived from `TileLangTQConfig`
- pass resolved integer offsets into the compiled kernel as constants

The single source of truth remains C1.

### 2.2 Layer buffer contract

The kernels consume the layer buffers created in C3:

- `_tq_key_centroids`
- `_tq_key_midpoints`
- `_tq_value_centroids`
- `_tq_value_midpoints`
- `_tq_rotation`
- `_tq_rotation_t` when nonsymmetric
- `_tq_S_matrix`

They must not assume any upstream TurboQuant in-tree buffer names.

### 2.3 Store-path row bound

The store kernel contract is fixed by the corrected HLD:

- authoritative row bound is `slot_mapping.shape[0]`
- `num_actual_tokens` is not the store length
- `slot_mapping[i] < 0` means “skip write”

Python must not prefilter those rows away.

### 2.4 Value path correction

The old draft regressed values back to min/max quantization. That is not this design.

Canonical rule:

- values use the same TurboQuant decomposition style as keys
- value path uses its own value codebook (`value_mse_bits`, `value_centroids`, `value_midpoints`)
- value path also carries a QJL residual channel in the current design

So the slot contains logical subfields for:

- key MSE payload
- key norm / residual metadata
- value MSE payload
- value norm / residual metadata

not a min/max value quantizer.

---

## 3. C6.1 Store Kernel

**File:** `kernels/tl_store.py`

### 3.1 Canonical signature

The exact low-level calling convention may vary slightly by TileLang wrapper, but the logical inputs are:

```python
def tl_turboquant_store(
    key: torch.Tensor,
    value: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor | None,
    key_midpoints: torch.Tensor,
    key_centroids: torch.Tensor,
    value_midpoints: torch.Tensor,
    value_centroids: torch.Tensor,
    S_matrix: torch.Tensor,
    cfg: TileLangTQConfig,
) -> None:
    ...
```

Canonical shapes:

- `key`: `(N, Hk, D)`
- `value`: `(N, Hk, D)`
- `kv_cache`: `(num_blocks, block_size, Hk, slot_size_aligned)`
- `slot_mapping`: `(N,)`

where `N = slot_mapping.shape[0]`.

### 3.2 Per-row logical behavior

For each logical `(token_row, kv_head)` pair:

1. read `slot = slot_mapping[token_row]`
2. if `slot < 0`, skip
3. compress the key vector:
   - normalize if required by the chosen formulation
   - apply the configured rotation
   - quantize rotated coordinates using the key Lloyd-Max codebook
   - reconstruct the MSE component
   - form the residual
   - compute the QJL sign payload and residual norm/scale metadata
4. compress the value vector using the value codebook and value QJL path
5. pack both halves into the exact slot offsets from C1
6. scatter the finished slot into:
   - `block_idx = slot // block_size`
   - `pos = slot % block_size`

### 3.3 Implementation flexibility

The logical behavior above is fixed, but the implementation may choose between:

- fully fused in-kernel math
- a small amount of pre-kernel temporary preparation
- a hybrid direct-pack path

Allowed flexibility:

- temporary per-step tensors may be formed as part of the normal compute graph
- the store implementation may precompute rotated vectors or residual projections if that is measurably better

Not allowed:

- changing the logical slot contents
- changing the values path back to min/max quantization
- introducing new long-lived device buffers after `profile_run()`

### 3.4 Rotation contract

The store kernel must honor the rotation mode selected by C2/C3:

- `qr` mode must work for any supported `head_dim`
- `randomized_hadamard` mode is allowed only when the configured shape/padding rules are satisfied

The kernel must not silently substitute a deterministic plain Hadamard if the configured rotation mode requires a different transform.

### 3.5 Packing rules

Packing is variant-dependent but driven entirely by C1 config:

- number of MSE bits comes from `key_mse_bits` / `value_mse_bits`
- number of codebook entries comes from the matching centroid tensors
- QJL bit count follows `head_dim`
- all byte offsets come from `SlotLayout`

The kernel may use any internal bit-packing strategy as long as decode reads the exact same layout.

### 3.6 Tiling rule

Preferred parallel decomposition:

- one logical tile per `(token_row, kv_head)`
- flatten to a 1D launch or equivalent TileLang grid

This matches the independence of slot writes and avoids cross-tile synchronization for the store path.

---

## 4. C6.2 Decode Kernel

**File:** `kernels/tl_decode.py`

### 4.1 Canonical signature

Logical inputs:

```python
def tl_turboquant_decode_attention(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    rotation: torch.Tensor,
    rotation_t: torch.Tensor | None,
    key_centroids: torch.Tensor,
    value_centroids: torch.Tensor,
    S_matrix: torch.Tensor,
    scale: float,
    cfg: TileLangTQConfig,
    mid_o_buf: torch.Tensor,
    lse_buf: torch.Tensor,
    output_buf: torch.Tensor,
    max_num_kv_splits: int,
) -> torch.Tensor:
    ...
```

Canonical shapes:

- `query`: `(B, Hq, D)`
- `block_table`: `(B, max_blocks)`
- `seq_lens`: `(B,)`
- scratch buffers come from C3 and are already allocated

### 4.2 Split-KV decode architecture

The canonical decode architecture is KV-split flash-decoding style:

1. split the cached context into `NUM_KV_SPLITS`
2. each split computes a partial attention contribution
3. merge partial outputs with a numerically stable log-sum-exp reduction

The merge step may:

- reuse an upstream stage-2 kernel when available and interface-compatible
- or use a plugin-local merge kernel

Reusing an upstream helper is optional, not a required dependency.

### 4.3 Two valid score formulations

The HLD allows either of these logical decode formulations:

#### Direct-score preferred path

Preferred when it reduces register pressure and memory traffic:

- transform or sketch the query once per `(batch, query_head)`
- compute MSE contribution directly in the rotated/codebook space
- compute QJL correction directly from the residual-sign payload and a query-side projection

This avoids reconstructing a full key vector for every cached position if the kernel can score directly from the packed representation.

#### Dequant-then-dot fallback

Also valid:

- reconstruct key and value vectors explicitly
- compute attention scores in the original space

This is simpler but can cost more bandwidth and arithmetic.

Canonical rule:

- either formulation is allowed
- both must be mathematically consistent with the same packed slot layout
- direct-score is the preferred optimization target, not a mandatory semantic change

### 4.4 Required key/value semantics

Regardless of which score formulation is used, decode must logically recover:

- the key MSE component from the key codebook
- the key QJL correction from the sign payload plus scale metadata
- the value MSE component from the value codebook
- the value QJL correction from the value sign payload plus scale metadata

The decode path must therefore not assume:

- values are min/max quantized
- only keys carry QJL state
- upstream TurboQuant slot semantics

### 4.5 GQA handling

Grouped-query attention expansion should happen inside the decode kernel or the kernel launch decomposition, not by materializing an expanded KV cache in Python.

Canonical rule:

- map query heads to KV heads inside the runtime path
- do not create a Python-side expanded cache tensor that would destroy the packed-memory advantage

### 4.6 Launch decomposition

Preferred stage-1 grid:

- batch dimension
- query-head dimension
- KV-split dimension

Preferred stage-2 grid:

- batch dimension
- query-head dimension

The exact TileLang launch syntax may differ, but the split/decompose/merge structure is the important contract.

### 4.7 Static split count

For CUDA graph compatibility, the impl treats `max_num_kv_splits` as static for the life of the impl instance.

Rules:

- graph-compatible path uses a fixed split count
- shorter sequences may leave some work tiles effectively empty
- if a branch lacks the newer runtime knob, use the documented plugin default from C4/C5

---

## 5. Launcher Wrapper Responsibilities

The Python wrappers around TileLang kernels are responsible for:

- validating coarse tensor shape/dtype invariants
- extracting slot-layout constants from C1
- passing preallocated scratch buffers
- selecting direct-score vs dequant-first decode formulation
- avoiding unnecessary Python-side copies or per-row loops

They are not responsible for:

- first materialization of long-lived matrices or workspaces
- fixing variant mismatches that should have been rejected in C3/C4

### 5.1 Runtime allocation rule

Allowed:

- ordinary ephemeral compute-graph temporaries needed for a forward step

Not allowed:

- first-time creation of long-lived decode/store workspaces
- first-time device migration of rotation/codebook/S buffers

Those had to be completed before `profile_run()`.

---

## 6. CUDA Graph Compatibility

The kernels and wrappers must remain graph-capturable under the support level promised by C4.

Required properties:

- no Python branching based on per-token device values in the hot path
- no dynamic long-lived buffer allocation during store/decode
- static split-count configuration for the graph-supported path
- use of preallocated output/scratch buffers from C3

`slot_mapping < 0` handling remains in-kernel, which is graph-safe and avoids Python-side filtering.

---

## 7. C7 — Plugin Registration

**Files:** `plugin.py`, `pyproject.toml`

C7 is where the whole design becomes visible to `vLLM`.

### 7.1 Entry-point group

Preferred packaging declaration:

```toml
[project.entry-points."vllm.general_plugins"]
tilelang_turboquant = "tilelang_turboquant.plugin:register_all"
```

This matches the local `vLLM` general-plugin loader model and is the canonical mechanism when the target branch supports that group.

If an older branch uses a different general-plugin discovery mechanism, the adapter must supply the equivalent hook, but the plugin-owned callable remains `register_all()`.

### 7.2 `register_all()` responsibilities

Canonical registration sequence:

1. import quantization config classes so registry decorators fire
2. apply cache-dtype admission shims from C3
3. apply KV-spec dispatch shim from C3
4. register the custom attention backend

Preferred sketch:

```python
_REGISTRATION_DONE = False


def register_all() -> None:
    global _REGISTRATION_DONE
    if _REGISTRATION_DONE:
        return
    _REGISTRATION_DONE = True

    from tilelang_turboquant.quantization.quant_config import (
        TileLangTQ3BitConfig,
        TileLangTQ4BitConfig,
    )

    _patch_cache_dtype_admission()
    _patch_attention_get_kv_cache_spec()
    _register_attention_backend()
```

Important correction relative to the old draft:

- registering quant configs and the backend alone is not enough
- the cache-dtype admission and KV-spec dispatch shims are part of the required registration sequence for older branches

### 7.3 Backend registration

Preferred current-tree pattern:

```python
register_backend(
    AttentionBackendEnum.CUSTOM,
    "tilelang_turboquant.backend.backend.TileLangTQAttentionBackend",
)
```

If an older branch lacks `AttentionBackendEnum.CUSTOM` or uses a different registration mechanism, the adapter must bind the backend through that branch’s equivalent custom-backend slot.

### 7.4 Per-process idempotency

Plugin loading is per process, not global.

Canonical guarantee:

- `register_all()` is idempotent within a process
- each `vLLM` process may still execute it once for itself

That includes:

- launcher / API process
- engine-core process
- worker processes

So `_REGISTRATION_DONE` is correctly process-local.

### 7.5 `VLLM_PLUGINS` filtering

On the local `vLLM` tree, plugin filtering happens at entry-point loading time based on the entry-point name.

Canonical plugin name:

- `tilelang_turboquant`

Important correction:

- when `VLLM_PLUGINS` is unset, all general plugins are eligible to load
- when `VLLM_PLUGINS` is set, only names present in that comma-split list are loaded
- on the current local tree, `VLLM_PLUGINS=""` does not mean “load all”; it yields an empty plugin name entry and effectively disables normal named plugins

So the reliable explicit enable form is:

```bash
VLLM_PLUGINS="tilelang_turboquant"
```

### 7.6 Multi-process expectation

The plugin must assume:

- some `vLLM` processes may import plugin machinery independently
- each process needs its own in-memory registry mutations
- successful registration in one process does not register another process

This is why idempotency and per-process registration are both required.

---

## 8. User-Facing Invocation

Canonical invocation remains:

```python
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-3-8B",
    quantization="tq_3bit",
    kv_cache_dtype="tilelang_tq_3bit",
    attention_backend="CUSTOM",
)
```

If the plugin accepts user aliases such as `kv_cache_dtype="tq_3bit"`, normalization must happen before branch cache-dtype validation.

---

## 9. Testing Contracts

### C6 Unit Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_store_skips_negative_slot_rows` | padded store rows | masked rows are not written |
| `test_store_uses_slot_layout_offsets_from_c1` | inspect packed slot | offsets match C1 |
| `test_store_packs_value_path_as_tq_not_minmax` | decode stored value payload | value path follows value codebook/QJL contract |
| `test_decode_recovers_value_qjl_path` | round-trip with value residual active | value residual contributes correctly |
| `test_decode_supports_direct_score_and_fallback_equivalence` | compare formulations | numerically consistent within tolerance |
| `test_decode_gqa_mapping_no_python_cache_expand` | GQA case | no expanded KV cache materialization required |

### C6 Integration Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_store_then_decode_round_trip_matches_reference` | one request | attention output within tolerance |
| `test_mixed_batch_decode_uses_same_packed_slots` | mixed workload | packed-slot contract preserved |
| `test_no_first_use_long_lived_allocations_in_kernel_wrappers` | memory instrumentation | no late long-lived buffer creation |
| `test_cudagraph_supported_path_captures` | uniform supported batch | graph capture succeeds |

### C7 Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_entry_point_declares_general_plugin` | inspect packaging metadata | entry point exists |
| `test_register_all_is_process_local_idempotent` | call twice in one process | no duplicate failure |
| `test_register_all_applies_c4_shims_before_backend_use` | backend resolution path | shims active before model init |
| `test_vllm_plugins_filter_requires_plugin_name_when_set` | filtered load | plugin only loads when listed |
| `test_backend_registration_visible_after_plugin_load` | query backend registry | custom backend path resolves |

---

## 10. Benchmark Expectations

Benchmarking belongs to the project plan, but C6/C7 should be evaluated with at least:

- store throughput
- decode throughput across short and long contexts
- mixed-batch behavior
- graph-captured decode
- memory footprint with all pre-`profile_run()` buffers present

The benchmark harness may compare:

- direct-score decode vs dequant-then-dot fallback
- QR rotation vs validated randomized-Hadamard mode
- plugin-local stage-2 merge vs upstream merge reuse when both are available
