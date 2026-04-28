# LLD: C2 — Offline Pre-computation

**Document type:** LLD (Low-Level Design)  
**Component:** C2 — `offline/codebook.py` + `offline/matrices.py`  
**HLD reference:** §5 C3, §7 Global Buffer Lifecycle, §11 Open Questions #2, #3, #6, #7, #12  
**Depends on:** C1 LLD (`TileLangTQConfig`, `key_mse_bits`, `value_mse_bits`)  
**Status:** Canonical plugin-owned design; valid for `vLLM` trees that do not ship upstream TurboQuant support

---

## 1. Purpose & Scope

C2 defines how the plugin prepares the read-only mathematical artifacts required by TurboQuant KV-cache compression:

| Artifact | Logical meaning | Canonical shape | Canonical dtype |
|---|---|---|---|
| `Π` | rotation matrix used before MSE quantization | `(D, D)` | `float32` on CPU |
| `Π^T` | inverse / transpose view used by decode | `(D, D)` | `float32` on CPU |
| `C_key`, `M_key` | key centroids and midpoint boundaries | `(2^b_k,)`, `(2^b_k - 1,)` | `float32` on CPU |
| `C_value`, `M_value` | value centroids and midpoint boundaries | `(2^b_v,)`, `(2^b_v - 1,)` | `float32` on CPU |
| `S` | QJL projection matrix | `(D, D)` | `float32` on CPU |

Where:
- `D = head_dim`
- `b_k = key_mse_bits`
- `b_v = value_mse_bits`

C2 owns:
- mathematical definitions of these artifacts
- canonical CPU-side generation and cache keys
- optional disk persistence
- validation rules
- the contract C3 must follow when moving artifacts to GPU

C2 does not own:
- `vLLM` registration hooks
- per-layer buffer names
- device-specific packing or transposition for kernels
- decode scratch workspace sizing

Those belong to C3/C5/C6.

---

## 2. Compatibility Rules

This LLD must work on `vLLM` versions older than the current local tree. Therefore:

- Do not assume the target `vLLM` version ships `turboquant_attn.py`, `TQFullAttentionSpec`, TurboQuant cache dtypes, or any upstream TurboQuant helper.
- Treat upstream TurboQuant implementations only as references for math and validation, not as required runtime dependencies.
- Keep the canonical offline representation plugin-owned and `vLLM`-agnostic.
- Materialize canonical artifacts on CPU first. Any GPU copies, casts, layout conversions, or per-layer `register_buffer` calls are integration work done by C3.
- No artifact required by store or decode may be first allocated on GPU after `profile_run()`.

The last rule is mandatory even on older branches with slightly different load hooks.

---

## 3. Public API

**Files:** `offline/matrices.py`, `offline/codebook.py`

The offline layer exposes three logical builders plus a small cache facade:

```python
@dataclass(frozen=True)
class RotationSpec:
    head_dim: int
    mode: Literal["qr", "randomized_hadamard"]
    seed: int
    allow_hadamard_padding: bool = False


@dataclass(frozen=True)
class CodebookSpec:
    head_dim: int
    mse_bits: int
    distribution: Literal["beta"] = "beta"


@dataclass(frozen=True)
class QJLSpec:
    head_dim: int
    seed: int


def get_rotation(spec: RotationSpec) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """Return (rotation, rotation_t, is_symmetric) as CPU float32 tensors."""


def get_codebook(spec: CodebookSpec) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (centroids, midpoints) as CPU float32 tensors."""


def get_s_matrix(spec: QJLSpec) -> torch.Tensor:
    """Return S as a CPU float32 tensor."""
```

Contract:
- all return tensors are contiguous CPU `float32`
- all builders are deterministic with respect to `spec`
- cache hits return numerically identical artifacts
- disk persistence, if enabled, stores CPU canonical tensors only

---

## 4. Rotation Matrix

### 4.1 Canonical Semantics

The logical rotation is an orthonormal matrix `Π ∈ R^(D×D)` used in:

- store: `y = Π · x`
- decode: `x_tilde = Π^T · y_tilde`

The offline layer must provide both `Π` and `Π^T`, even if they alias mathematically.

### 4.2 Supported Modes

#### `qr` mode

This is the compatibility baseline.

Construction:
1. Sample seeded Gaussian `G ∈ R^(D×D)`.
2. Compute reduced QR decomposition `G = QR`.
3. Fix QR sign ambiguity deterministically by absorbing `sign(diag(R))` into `Q`.
4. Return `Π = Q`, `Π^T = Q^T`.

Properties:
- supports arbitrary `head_dim`
- matches the reference TurboQuant formulation most directly
- should be the default mode on older branches unless a faster mode is explicitly enabled and validated

#### `randomized_hadamard` mode

This is an optional optimization path, not the compatibility baseline.

Canonical requirement:
- the realized transform must be orthonormal
- it must include randomization, not a plain deterministic Hadamard alone
- it must pass the same round-trip and model-quality gates defined in testing

Acceptable realizations include one Hadamard factor plus random sign flips and optional permutation. The exact factor order may vary by implementation as long as the same logical operator is applied consistently in store and decode.

### 4.3 Non-Power-of-2 Head Dimensions

The HLD explicitly leaves this as a design concern for the implementation path. The compatibility-safe rule is:

- `qr` mode must support any `head_dim`
- `randomized_hadamard` mode may only run when one of these is true:
  - `head_dim` is a power of two
  - the implementation has an explicit, tested padding strategy enabled by configuration

If neither condition holds, the builder must reject `randomized_hadamard` and fall back to `qr` at the integration layer.

This avoids silently inventing a lower-fidelity approximation for unsupported dimensions.

### 4.4 Canonical Builder Sketch

```python
def build_qr_rotation(head_dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    gaussian = torch.randn(head_dim, head_dim, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(gaussian, mode="reduced")

    signs = torch.sign(torch.diag(r))
    signs[signs == 0] = 1
    q = q * signs.unsqueeze(0)
    return q.contiguous(), q.transpose(0, 1).contiguous()
```

The Hadamard builder may use a different internal construction, but it must satisfy the same output contract.

### 4.5 Registration Dtype

Canonical offline artifacts stay in CPU `float32`.

When C3 registers them on device:
- `float32` is the baseline-safe runtime dtype
- `float16` is permitted only as an explicitly validated runtime choice for the target kernels and target model family

The offline layer itself does not hard-code the device dtype.

---

## 5. Lloyd-Max Codebook

### 5.1 Canonical Prior

The canonical codebook prior is the induced rotated-coordinate density from the HLD:

`f_X(x) = (Gamma(d/2) / (sqrt(pi) * Gamma((d-1)/2))) * (1 - x^2)^((d-3)/2)`, for `x ∈ [-1, 1]`

This LLD does not replace that prior with the Gaussian approximation used by some reference implementations. If a compatibility branch wants an approximate solver for startup speed, it must validate it against the canonical Beta-prior output and make that tradeoff explicit.

### 5.2 Key and Value Codebooks

The plugin applies the same TurboQuant structure to both keys and values, so C2 produces:

- key codebook from `(head_dim, key_mse_bits)`
- value codebook from `(head_dim, value_mse_bits)`

If `key_mse_bits == value_mse_bits`, the offline layer may deduplicate CPU storage internally, but C3 still registers the logical buffers separately:

- `_tq_key_centroids`
- `_tq_key_midpoints`
- `_tq_value_centroids`
- `_tq_value_midpoints`

### 5.3 Solver Contract

The solver alternates between:

1. boundary update: `m_i = (c_i + c_(i+1)) / 2`
2. centroid update: conditional mean of `f_X(x)` over each Voronoi interval

The LLD fixes the output contract, not one exact numerical quadrature scheme. Any implementation is acceptable if it satisfies:

- deterministic output for fixed `(head_dim, mse_bits, algorithm_version)`
- centroids strictly increasing
- symmetry about zero within tolerance
- midpoint consistency
- convergence or bounded residual verified by tests

### 5.4 Canonical API Sketch

```python
def solve_lloyd_max_beta(
    head_dim: int,
    mse_bits: int,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return centroids and midpoints for the exact Beta prior."""
```

Notes:
- the support is `[-1, 1]`
- output tensors are sorted ascending
- `midpoints[i] = (centroids[i] + centroids[i + 1]) / 2`

### 5.5 Persistence

Codebooks may be:

- recomputed in memory
- loaded from a plugin-owned disk cache
- shipped as bundled precomputed artifacts

All three are valid. The canonical semantics are the same.

If disk persistence is used:
- store CPU `float32` tensors only
- key the artifact by `head_dim`, `mse_bits`, and `algorithm_version`
- treat cache failure as non-fatal and recompute in memory

The plugin must not require writable home-directory access in order to start.

---

## 6. QJL Projection Matrix `S`

### 6.1 Canonical Semantics

`S ∈ R^(D×D)` with entries sampled i.i.d. from `N(0, 1)`.

It is used as:
- store: `qjl = sign(S · r)`
- decode: residual estimator based on `S^T · qjl`, with the scale factor applied by runtime kernels

### 6.2 Storage Rule

The logical matrix is dense Gaussian.

The HLD allows storage optimization as an LLD decision, but the compatibility-safe contract is:

- canonical offline representation is dense CPU `float32`
- any compressed, sparse, or sign-only internal representation is an implementation detail and must remain mathematically equivalent to the same seeded dense logical matrix

This keeps C2 independent of kernel-specific shortcuts.

### 6.3 Sharing Rule

One seeded `S` per `(head_dim, seed)` is the canonical artifact.

That artifact may be shared at the CPU cache level. When it is registered onto layers:
- aliasing or cloning is a C3 ownership decision
- each layer must still observe the same numerical matrix for the same config

### 6.4 Builder Sketch

```python
def build_s_matrix(head_dim: int, seed: int) -> torch.Tensor:
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return torch.randn(head_dim, head_dim, generator=g, dtype=torch.float32).contiguous()
```

---

## 7. Artifact Cache and Persistence

### 7.1 Process-Local Memoization

C2 should maintain process-local memoization keyed by the spec dataclasses above. This avoids recomputing identical artifacts across layers.

Recommended behavior:
- memoize CPU canonical tensors
- never memoize device tensors inside C2
- keep cache entries immutable after creation

### 7.2 Optional Disk Cache

If the plugin enables disk caching, the cache key must include enough information to prevent semantic collisions.

Minimum key fields:
- artifact kind: `rotation`, `codebook`, `s_matrix`
- `head_dim`
- `mse_bits` for codebooks
- rotation `mode`
- `seed` where relevant
- `algorithm_version`

Disk cache contents:
- CPU `float32` tensors
- minimal metadata needed to validate the artifact

Disk cache rules:
- cache miss is normal
- corrupt entry triggers recompute
- read-only filesystem must not fail startup
- device placement is never serialized

---

## 8. Registration Timing and `vLLM` Lifecycle

C2 artifacts become runtime buffers in C3. The required timing is:

1. weights load completes
2. the earliest post-load hook available on that `vLLM` branch builds or loads all required C2 artifacts
3. C3 copies them to the target device, casts if needed, and registers layer buffers
4. only after that may `profile_run()` execute

Required per-layer logical buffers:
- `_tq_key_centroids`
- `_tq_key_midpoints`
- `_tq_value_centroids`
- `_tq_value_midpoints`
- `_tq_rotation`
- `_tq_rotation_t` when rotation is not symmetric
- `_tq_S_matrix`

Rules:
- no lazy `_ensure_on_device` path
- no first-use GPU allocation in `forward` or `do_kv_cache_update`
- if an older branch lacks the exact modern hook names, the adapter must still choose a hook that runs before memory profiling

This rule matches the corrected HLD memory-budget contract.

---

## 9. Validation Invariants

These checks should run once per unique artifact, not on every token step.

### 9.1 Rotation

- shape is `(D, D)`
- dtype is CPU `float32` before registration
- `Π^T` matches the transpose of `Π`
- `Π @ Π^T ≈ I` within configured tolerance
- if mode is `randomized_hadamard` and `head_dim` is not power-of-two, the builder must prove a supported padding strategy was explicitly enabled

### 9.2 Codebook

- `len(centroids) == 2^mse_bits`
- `len(midpoints) == len(centroids) - 1`
- centroids strictly increasing
- centroids symmetric about zero within tolerance
- `midpoints[i]` equals adjacent-centroid midpoint within tolerance

### 9.3 QJL Matrix

- shape is `(D, D)`
- dtype is CPU `float32` before registration
- deterministic for fixed `(head_dim, seed)`
- sample statistics are reasonable for a Gaussian draw

### 9.4 Cache Integrity

- cache key changes when `algorithm_version` changes
- stale or corrupt disk entries do not poison runtime state

---

## 10. Testing Contracts

### Unit Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_qr_rotation_is_orthonormal` | build QR rotation for `D=128` | `Π @ Π^T ≈ I` |
| `test_qr_rotation_is_deterministic` | same seed twice | identical tensors |
| `test_qr_supports_non_power_of_two` | `D=96` | succeeds |
| `test_hadamard_requires_supported_shape_or_padding` | `D=96`, padding disabled | explicit failure |
| `test_codebook_count_matches_bits` | `mse_bits in {1,2,3}` | `2^bits` centroids |
| `test_codebook_is_symmetric` | `D=128` | `c[i] ≈ -c[-1-i]` |
| `test_midpoints_match_centroids` | any supported config | midpoint identity holds |
| `test_s_matrix_is_deterministic` | same seed twice | identical tensors |
| `test_s_matrix_changes_with_seed` | different seeds | different tensors |
| `test_disk_cache_fallback_is_nonfatal` | unwritable temp cache | in-memory recompute succeeds |

### Integration Tests

| Test ID | Description | Expected |
|---|---|---|
| `test_offline_artifacts_registered_before_profile_run` | run post-load registration on mock layer | all C2 buffers exist before profiling |
| `test_key_and_value_codebooks_registered_separately` | matching bit-widths | logical key/value buffers both exist |
| `test_no_runtime_gpu_materialization` | forward after load | no first-use allocation path |
| `test_older_vllm_adapter_can_boot_without_upstream_turboquant` | run plugin on a branch without built-in TQ helpers | startup succeeds |

### Reference Validation

When the branch enables `randomized_hadamard`, add an explicit comparison suite against the QR baseline:

- round-trip reconstruction error stays within agreed tolerance
- model-quality regression is within the project threshold

Until that suite passes, `qr` remains the safe default.

---

## 11. Implementation Notes for C3

C3 should consume this module as follows:

- pick `RotationSpec` from the configured rotation mode and seed
- request key and value codebooks separately using `key_mse_bits` and `value_mse_bits`
- request `S` from `(head_dim, seed)`
- move all required artifacts to the final device before `profile_run()`
- optionally cast rotation / `S` to a validated runtime dtype during registration

This separation keeps C2 stable across `vLLM` versions even when the post-load hooks differ.
