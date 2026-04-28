# Agent reference: quantization registration & checkpoint → runtime buffers

**Audience:** LLM drafting HLD/LLD for TQ KV plugins. **Scope:** `@register_quantization_config`, `BaseKVCacheMethod`, `get_cache_scale`, two-pass load → `_k_scale` buffers; QKV/RoPE semantics at attention boundary.

---

## Problem / pattern

- Checkpoints load into **`nn.Parameter`** (loader-friendly, sentinel e.g. `-1` for “unset”).
- Inference wants **`register_buffer`** (no autograd, stable for compile, moves with module).
- Handoff in **`quant_method.process_weights_after_loading`** → copy/delete Parameters.

---

## Phase 0: Registration

- `@register_quantization_config("name")` (`quantization/__init__.py`): append to `QUANTIZATION_METHODS`, platform allowlist, `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG[name] = cls`.
- **Import before `LLM(...)`** or use **`vllm.general_plugins`** entry point so every process registers (see Part 4).

**Config instantiation:** `get_quant_config` (`weight_utils.py`): HF `quantization_config`, overrides, `from_config`, or default ctor.

---

## Phase 1: `Attention` init (`_init_kv_cache_quant`, `attention.py`)

1. `set_default_quant_scales(register_buffer=True)` → `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale` float32 buffers + `_float` mirrors on module.
2. If `should_load_quant_weights(quant_method)` and `BaseKVCacheMethod`: `quant_method.create_weights(layer)` → temp `k_scale`… `Parameter(-1)`.

---

## Phase 2: Loading (`LlamaModel.load_weights`, `llama.py`)

- For each checkpoint tensor: `scale_name = quant_config.get_cache_scale(name)`; if hit, map to `...attn.k_scale` Parameter and `weight_loader`.
- Scalars: squeeze `[1]` → rank-0.
- Legacy names: `maybe_remap_kv_scale_name`.
- **Non-scalar metadata** (codebooks): usually **not** via `get_cache_scale`; load in general weight loop into Parameters/buffers you created in `create_weights`.

**FP8-style mapping example:** `.k_proj.output_scale` → `.attn.k_scale`, `.v_proj.output_scale` → `.attn.v_scale`.

---

## Phase 3: Post-load (`process_weights_after_loading`, `model_loader/utils.py`)

**Pass 1:** every module with `quant_method`: `quant_method.process_weights_after_loading(module)`.

**Pass 2:** every `Attention`: `Attention.process_weights_after_loading(dtype)` → may call `impl.process_weights_after_loading`.

**`BaseKVCacheMethod.process_weights_after_loading` (`kv_cache.py`):** branches for per-token-head scales (buffers ← 1.0, delete params); static KV quant: validate `k_scale>0` etc., copy to `_k_scale`, set `_k_scale_float`, FNUZ FP8 adjust on ROCm, `del k_scale`…; early exit if params already deleted (reload).

**Custom ~3-bit / non-scalar:** often **full override** — avoid naive `super()` (FP8-ish assumptions, `.tolist()` scalars). Still `del` temp Parameters; keep buffers.

---

## `QuantizationConfig` checklist

| Method | Purpose |
|--------|---------|
| `get_name` | Identifier |
| `get_supported_act_dtypes` | e.g. fp16/bf16 |
| `get_quant_method(layer, prefix)` | Return `YourKVCacheMethod` for `Attention`, else `None` |
| `get_cache_scale(checkpoint_name)` | str→str map to `...attn.k_scale` / `v_scale` / … |
| `get_kv_cache_spec(layer, vllm_config)` | Return `TQFullAttentionSpec` / custom spec with `tq_slot_size`, `dtype=int8` |

---

## `BaseKVCacheMethod` subclass checklist

- `create_weights`: usually `super()` for scalar params; add extra `Parameter`s for custom tensors; **pre-register buffers** needed on GPU before profile if profiler must see them.
- `process_weights_after_loading`: validate sentinels; copy to `_` buffers; copy codebooks; delete **all** temp Parameters; don’t delete buffers.
- `apply`: base raises — KV path not used for linear quant via this hook.

**Dtype coupling:** custom `kv_cache_dtype` must be recognized by your branches or fully custom post-load; align with `Attention.__init__` hooks for TQ-style buffer init.

---

## QKV + RoPE before `Attention.forward` (`llama.py` pattern)

1. `qkv_proj` → split `q, k, v` (2D `(T, ·)`).
2. `rotary_emb(positions, q, k)` — **V excluded**.
3. `self.attn(q, k, v)` then `Attention.forward` reshapes to 3D.

| Tensor | RoPE | Role for store |
|--------|------|----------------|
| `q` | Yes | Scores |
| `k` | Yes | **Store quantized K** (post-positional) |
| `v` | No | **Store quantized V** (content) |

**GQA:** `q_size ≠ kv_size`; store uses `num_kv_heads`.

**After reshape:** `(T, Hkv, D)` — no implicit copy; avoid unnecessary `.contiguous()` in hot path.

---

## Relation to execution (Part 2)

- After post-load: kernels read `_k_scale` / `_v_scale` / custom buffers.
- `do_kv_cache_update` runs before `forward`; scales/buffers on `layer` available in both.

---

## Key files

| Topic | Path |
|-------|------|
| Register quant | `vllm/model_executor/layers/quantization/__init__.py` |
| Base KV method | `vllm/model_executor/layers/quantization/kv_cache.py` |
| Attention quant init | `vllm/model_executor/layers/attention/attention.py` |
| Load + `get_cache_scale` | `vllm/model_executor/models/llama.py` |
| `get_quant_config` | `vllm/model_executor/model_loader/weight_utils.py` |
| Post-load orchestration | `vllm/model_executor/model_loader/utils.py` |
| Example `get_cache_scale` | `vllm/model_executor/layers/quantization/fp8.py` |
