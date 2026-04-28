# vLLM custom quantization, KV metadata, and pre-attention tensor state

Technical deep-dive for out-of-tree quantization plugins (e.g. ~3-bit KV cache): how configs are registered and selected, how checkpoint metadata flows into `Attention` before inference, and the exact state of `q` / `k` / `v` at the `Attention.forward` boundary (Llama-style models).

**Scope:** Findings are tied to the vLLM tree at `vllm/model_executor/` (paths below are relative to the `vllm` package root).

---

## 1. Quantization registration (`@register_quantization_config`) and user-facing triggers

### 1.1 What the decorator does

`register_quantization_config(quantization: str)` is a class decorator that:

1. **Registers the string** in `QUANTIZATION_METHODS` (unless it collides with an existing built-in name—in which case it logs a warning and **overwrites** the mapping).
2. **Appends** the same string to `current_platform.supported_quantization` when the platform exposes a non-empty supported list (e.g. ROCm), so `verify_quantization` can accept the custom method.
3. **Stores** `quantization → QuantizationConfig` subclass in `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG`.
4. **`get_quantization_config(name)`** merges `_CUSTOMIZED_METHOD_TO_QUANT_CONFIG` into the built-in `method_to_config` map and returns the class.

Relevant implementation:

```57:104:vllm/model_executor/layers/quantization/__init__.py
def register_quantization_config(quantization: str):
    """Register a customized vllm quantization config.
    ...
    """
    def _wrapper(quant_config_cls):
        if quantization in QUANTIZATION_METHODS:
            logger.warning(
                "The quantization method '%s' already exists and will be "
                "overwritten by the quantization config %s.",
                quantization,
                quant_config_cls,
            )
        else:
            QUANTIZATION_METHODS.append(quantization)
            # Automatically assume the custom quantization config is supported
            if sq := current_platform.supported_quantization:
                sq.append(quantization)
        ...
        _CUSTOMIZED_METHOD_TO_QUANT_CONFIG[quantization] = quant_config_cls
        return quant_config_cls

    return _wrapper
```

```181:183:vllm/model_executor/layers/quantization/__init__.py
    # Update the `method_to_config` with customized quantization methods.
    method_to_config.update(_CUSTOMIZED_METHOD_TO_QUANT_CONFIG)

    return method_to_config[quantization]
```

### 1.2 Out-of-tree usage

1. **Import your plugin module before constructing the engine/LLM**, so the decorator runs at import time and the registry is populated.
2. Pass your method name via **`ModelConfig.quantization`** (same string as the decorator argument).

Supported entry points include:

- **`LLM(..., quantization="your_method")`** — see `ModelConfig.quantization` semantics.
- **CLI / `EngineArgs`**: `--quantization your_method` (same underlying field).

`ModelConfig.quantization` is documented as: if `None`, vLLM may infer from `quantization_config` in the HF config; otherwise the explicit string wins.

```198:202:vllm/config/model.py
    quantization: QuantizationMethods | str | None = None
    """Method used to quantize the weights. If `None`, we first check the
    `quantization_config` attribute in the model config file. If that is
    `None`, we assume the model weights are not quantized and use `dtype` to
    determine the data type of the weights."""
```

Validation accepts any string present in `QUANTIZATION_METHODS` (which includes dynamically appended custom names).

### 1.3 Instantiating `QuantizationConfig` for load

`get_quant_config(model_config, load_config)` resolves the **class** via `get_quantization_config(model_config.quantization)`, then constructs an instance using (in order of precedence):

- HF `quantization_config` / `compression_config` → `quant_cls.from_config(...)` when present.
- `hf_overrides` keys such as `quantization_config_file`, `quantization_config_dict_json`, when implemented.
- Online-quant path via `quantization_config` argument on `ModelConfig`.
- Fallback: JSON files listed in `get_config_filenames()`, or **`quant_cls()`** with no args if that list is empty.

```259:267:vllm/model_executor/model_loader/weight_utils.py
def get_quant_config(
    model_config: ModelConfig, load_config: LoadConfig
) -> QuantizationConfig:
    if model_config.quantization is None:
        raise ValueError("Model quantization method is not specified in the config.")
    quant_cls = get_quantization_config(model_config.quantization)

    # GGUF doesn't have config file
    if model_config.quantization == "gguf":
        return quant_cls()
```

**Implication for plugins:** implement `from_config`, `get_config_filenames`, and/or document how users supply HF-side JSON so `get_quant_config` can build your config. Many tests use `get_config_filenames() -> []` and `from_config` for minimal setups.

`VllmConfig` then holds `quant_config` and models read `vllm_config.quant_config` when building layers.

---

## 2. Metadata injection: `BaseKVCacheMethod`, checkpoints, and `_k_scale` / `_v_scale`

### 2.1 Two different “scale” storages

KV scaling in vLLM uses a deliberate two-phase pattern:

| Phase | Role |
|--------|------|
| **Load-time parameters** | `k_scale`, `v_scale`, `q_scale`, `prob_scale` as **`nn.Parameter`** (scalar), created by `BaseKVCacheMethod.create_weights`, intended to receive checkpoint values. |
| **Runtime buffers** | `_k_scale`, `_v_scale`, `_q_scale`, `_prob_scale` as **`register_buffer`** float tensors, plus host mirrors `_k_scale_float`, etc., used by attention backends / kernels. |

`Attention` initializes the buffers first, then may replace the loadable `Parameter`s:

```93:173:vllm/model_executor/layers/attention/attention.py
def set_default_quant_scales(layer: nn.Module, register_buffer: bool = False) -> None:
    """Sets default quantization scales for the layer."""
    if register_buffer:
        layer.register_buffer("_k_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_v_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_q_scale", torch.tensor(1.0, dtype=torch.float32))
        layer.register_buffer("_prob_scale", torch.tensor(1.0, dtype=torch.float32))
    ...
    layer._q_scale_float = 1.0
    layer._k_scale_float = 1.0
    layer._v_scale_float = 1.0
    ...
def _init_kv_cache_quant(
    layer: nn.Module,
    quant_config: QuantizationConfig | None,
    prefix: str,
) -> None:
    ...
    set_default_quant_scales(layer, register_buffer=True)
    ...
    quant_method = (
        quant_config.get_quant_method(layer, prefix=prefix) if quant_config else None
    )
    ...
    if should_load_quant_weights(quant_method):
        assert isinstance(quant_method, BaseKVCacheMethod)
        ...
        layer.quant_method = quant_method
        layer.quant_method.create_weights(layer)
```

**Dispatch:** Your `QuantizationConfig.get_quant_method` must return a subclass of **`BaseKVCacheMethod`** for `Attention` if you want this path (same pattern as `Fp8Config` → `Fp8KVCacheMethod`).

```172:204:vllm/model_executor/layers/quantization/fp8.py
    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> "QuantizeMethodBase | None":
        ...
        elif isinstance(layer, Attention):
            return Fp8KVCacheMethod(self)
        return None
```

`Fp8KVCacheMethod` is a thin subclass of `BaseKVCacheMethod` with no extra logic—your plugin can mirror this pattern.

### 2.2 `BaseKVCacheMethod.create_weights` and `process_weights_after_loading`

```32:47:vllm/model_executor/layers/quantization/kv_cache.py
    def create_weights(self, layer: torch.nn.Module):
        """
        Create "weight" (aka q_scale, k_scale and v_scale)
        for an attention layer.
        """
        # Initialize the Q and KV cache scales to -1.0, an invalid value.
        # If the q and k/v_scales appear in the checkpoint, it will be
        # overwritten when loading weights.
        layer.q_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.k_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        # Initialize P = softmax(QK^T) scales
        layer.prob_scale = torch.nn.Parameter(torch.tensor(-1.0), requires_grad=False)
```

After weights are loaded, **`process_weights_after_loading` on the quant method** copies validated values into `_k_scale` / `_v_scale` / `_q_scale` / `_prob_scale`, sets the `_float` mirrors, and **deletes** the temporary `Parameter` attributes. The built-in implementation encodes FP8 KV policy (quantized cache vs dynamic per-token scales, `calculate_kv_scales`, FNUZ handling, etc.):

```49:172:vllm/model_executor/layers/quantization/kv_cache.py
    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # skip if there are no weights to process (for example, weight reloading)
        if not hasattr(layer, "q_scale"):
            ...
            return
        ...
        if kv_cache_uses_per_token_head_scales(layer.kv_cache_dtype):
            layer._k_scale.copy_(1.0)
            layer._v_scale.copy_(1.0)
            ...
            del layer.k_scale
            ...
            return
        ...
        if (
            is_quantized_kv_cache(layer.kv_cache_dtype)
            and not layer.calculate_kv_scales
        ):
            ...
            layer._k_scale.copy_(k_scale)
            layer._v_scale.copy_(v_scale)
            layer._k_scale_float = k_scale
            layer._v_scale_float = v_scale
        ...
        layer._q_scale.copy_(q_scale)
        ...
        del layer.k_scale
        del layer.v_scale
        del layer.q_scale
        del layer.prob_scale
```

**For a custom ~3-bit KV plugin:** subclass `BaseKVCacheMethod`, override `create_weights` if you need **additional** `Parameter`s or buffers (e.g. non-scalar metadata), and override `process_weights_after_loading` to copy into whatever buffers your Tile Lang / CUDA kernels read—while staying compatible with vLLM’s expectations where possible (`_k_scale` tensor + `_k_scale_float` is a well-trodden path).

### 2.3 Loading scales from the checkpoint (Llama example)

`LlamaModel.load_weights` intercepts tensor names that the **`QuantizationConfig` maps** to attention scale parameters via `get_cache_scale`:

```459:469:vllm/model_executor/models/llama.py
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
```

FP8 illustrates the name mapping contract (your plugin should implement analogous `get_cache_scale` rules for your checkpoint layout):

```206:224:vllm/model_executor/layers/quantization/fp8.py
    def get_cache_scale(self, name: str) -> str | None:
        ...
        if name.endswith(".output_scale") and ".k_proj" in name:
            return name.replace(".k_proj.output_scale", ".attn.k_scale")
        if name.endswith(".output_scale") and ".v_proj" in name:
            return name.replace(".v_proj.output_scale", ".attn.v_scale")
        if name.endswith(".output_scale") and ".q_proj" in name:
            return name.replace(".q_proj.output_scale", ".attn.q_scale")
        if name.endswith("self_attn.prob_output_scale"):
            return name.replace(".prob_output_scale", ".attn.prob_scale")
        # If no matches, return None
        return None
```

Additional remapping for alternate naming (e.g. deprecated `kv_scale`) is handled in `maybe_remap_kv_scale_name` during the same load pass (see `weight_utils.maybe_remap_kv_scale_name`).

### 2.4 Global ordering: when metadata is guaranteed ready

`BaseModelLoader.load_model` constructs the model, calls `load_weights`, then **`process_weights_after_loading`** on the full module tree:

```76:81:vllm/model_executor/model_loader/base_loader.py
            # Process weights into kernel format. Note that when using online
            # quantization, weights are (typically) quantized as they are loaded.
            if _has_online_quant(model):
                finalize_layerwise_processing(model, model_config)

            process_weights_after_loading(model, model_config, target_device)
```

`process_weights_after_loading` (model loader helper) runs in **two phases**:

1. **Every submodule** with `quant_method: QuantizeMethodBase` → `quant_method.process_weights_after_loading(module)` (this includes **`Fp8KVCacheMethod` / `BaseKVCacheMethod` on each `Attention`**).
2. **Every `Attention` / `MLAAttention`** → `module.process_weights_after_loading(model_config.dtype)` (backend-specific impl post-processing and, if no quant scales were loaded, default scale initialization).

```95:118:vllm/model_executor/model_loader/utils.py
def process_weights_after_loading(
    model: nn.Module, model_config: ModelConfig, target_device: torch.device
) -> None:
    for _, module in model.named_modules():
        quant_method = getattr(module, "quant_method", None)
        if isinstance(quant_method, QuantizeMethodBase):
            ...
            with device_loading_context(module, target_device):
                quant_method.process_weights_after_loading(module)

    # Initialize post-load attention weights for both Attention and MLA.
    # NOTE: Happens after other modules so we can easily decompress weights.
    for _, module in model.named_modules():
        if isinstance(module, (Attention, MLAAttention)) and hasattr(
            module, "process_weights_after_loading"
        ):
            ...
            with device_loading_context(module, target_device):
                module.process_weights_after_loading(model_config.dtype)
```

**Conclusion:** Checkpoint-derived KV metadata is applied **after** all weights are loaded and **before** normal inference forward passes. Your kernels should read **`_k_scale` / `_v_scale` (and floats)** after this stage, not the temporary `k_scale` / `v_scale` parameters (those are removed when processing completes).

`Attention.process_weights_after_loading` delegates to the attention **impl** first, then optionally resets scales if the layer did not use a loadable KV quant method:

```577:589:vllm/model_executor/layers/attention/attention.py
    def process_weights_after_loading(self, act_dtype: torch.dtype):
        self.impl.process_weights_after_loading(act_dtype)

        # If we should not load quant weights, we initialize the scales to 1.0
        ...
        quant_method = (
            self.quant_config.get_quant_method(self, prefix=self.layer_name)
            if self.quant_config
            else None
        )
        if not should_load_quant_weights(quant_method):
            set_default_quant_scales(self, register_buffer=False)
```

---

## 3. RoPE and QKV split: tensor state at the `Attention.forward` entry (Llama)

### 3.1 Order of operations in `LlamaAttention.forward`

```223:232:vllm/model_executor/models/llama.py
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output
```

**RoPE:** Applied to **`q` and `k` only** before `self.attn(...)`. **`v` is not rotated.**

The rotary module reshapes `q`/`k` internally to `[num_tokens, n_heads, head_size]` (or grouped heads for `k`), applies rotation to the first `rotary_dim` dimensions, then restores the original tensor shape. So at the call into `Attention`, **`k` already carries positional encoding** (for standard Llama RoPE).

### 3.2 Rank / layout at `Attention.forward` entry

Immediately inside `Attention.forward`, tensors are still **flat along the last dimension** for the head concatenation (for the common Llama path): shape **`[num_tokens, q_size]`**, **`[num_tokens, kv_size]`**, **`[num_tokens, kv_size]`** where `q_size = num_heads * head_dim` and `kv_size = num_kv_heads * head_dim` **on the current tensor-parallel rank**.

The attention layer then **views** into head-major layout:

```465:516:vllm/model_executor/layers/attention/attention.py
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        ...
    ) -> torch.Tensor:
        ...
        # Reshape the query, key, and value tensors.
        ...
        query = query.view(-1, self.num_heads, self.head_size)
        output = output.view(-1, self.num_heads, self.head_size_v)
        if key is not None:
            key = key.view(-1, self.num_kv_heads, self.head_size)
        if value is not None:
            value = value.view(-1, self.num_kv_heads, self.head_size_v)
```

So:

- **Before** `view`: last dim is the **concatenated** per-rank Q/K/V projection output (full width for that rank), not a reduced-rank projection.
- **After** `view`: `[batch_tokens, heads, head_dim]` as expected by the unified attention op.

The implementation also documents support for **3D** query shapes in some code paths (`[num_tokens, heads, head_dim]`); Llama uses the **2D** packed layout.

### 3.3 Summary table (Llama → `Attention`)

| Tensor | RoPE applied? | Typical shape entering `Attention` (Llama) | Notes |
|--------|----------------|--------------------------------------------|--------|
| `q` | Yes | `[T, q_size]` | `q_size = local_heads * head_dim` |
| `k` | Yes | `[T, kv_size]` | `kv_size = local_kv_heads * head_dim` |
| `v` | No | `[T, kv_size]` | Unrotated value projection |

---

## 4. Practical checklist for a ~3-bit KV plugin

1. **`@register_quantization_config("your_kv3")`** on your `QuantizationConfig`; ensure the module is imported before engine creation.
2. **User sets** `quantization="your_kv3"` (or relies on HF config resolution if you wire `from_config` + `quantization_config` in the model card).
3. **`get_quant_method(..., Attention)`** returns a **`BaseKVCacheMethod`** subclass; implement **`create_weights` / `process_weights_after_loading`** for your metadata and copy into **`_k_scale` / `_v_scale`** (and any extra `register_buffer` tensors your kernel needs).
4. **`get_cache_scale`** (and/or `maybe_remap_kv_scale_name` patterns) so `LlamaModel.load_weights` can map checkpoint keys → `...attn.k_scale` etc.
5. Align **`cache_config.cache_dtype`** / `calculate_kv_scales` with your `BaseKVCacheMethod` logic—the stock `BaseKVCacheMethod.process_weights_after_loading` branches heavily on **`is_quantized_kv_cache`**, **`calculate_kv_scales`**, and **per-token-head** KV modes; a 3-bit scheme may need a **full override** of `process_weights_after_loading` rather than calling `super()` unmodified.

---

## References (primary files)

| Topic | File |
|--------|------|
| Registration API | `vllm/model_executor/layers/quantization/__init__.py` |
| KV method base | `vllm/model_executor/layers/quantization/kv_cache.py` |
| FP8 example: `get_quant_method`, `get_cache_scale`, `Fp8KVCacheMethod` | `vllm/model_executor/layers/quantization/fp8.py` |
| Attention buffers + forward reshape | `vllm/model_executor/layers/attention/attention.py` |
| Llama QKV split + RoPE + `attn` call | `vllm/model_executor/models/llama.py` |
| Post-load pipeline | `vllm/model_executor/model_loader/utils.py`, `vllm/model_executor/model_loader/base_loader.py` |
| `get_quant_config` | `vllm/model_executor/model_loader/weight_utils.py` |
