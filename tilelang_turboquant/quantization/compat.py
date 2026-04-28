"""C3 compatibility helpers for plugin cache dtypes and KV spec dispatch."""

from __future__ import annotations

from typing import Any, Callable

import torch

from tilelang_turboquant.config.variant_registry import (
    PLUGIN_KV_CACHE_DTYPE_3BIT,
    PLUGIN_KV_CACHE_DTYPE_4BIT,
)

CACHE_DTYPE_ALIASES = {
    PLUGIN_KV_CACHE_DTYPE_3BIT: PLUGIN_KV_CACHE_DTYPE_3BIT,
    PLUGIN_KV_CACHE_DTYPE_4BIT: PLUGIN_KV_CACHE_DTYPE_4BIT,
    "tq_3bit": PLUGIN_KV_CACHE_DTYPE_3BIT,
    "tq_4bit": PLUGIN_KV_CACHE_DTYPE_4BIT,
}

VARIANT_BY_CACHE_DTYPE = {
    PLUGIN_KV_CACHE_DTYPE_3BIT: "tq_3bit",
    PLUGIN_KV_CACHE_DTYPE_4BIT: "tq_4bit",
}

_CACHE_CONFIG_INIT_PATCHED = False
_ATTN_BACKEND_PATCHED = False
_KV_SPEC_PATCHED = False
_CUSTOM_BACKEND_NAME_PATCHED = False
_ORIG_CACHE_CONFIG_INIT: Callable[..., Any] | None = None
_ORIG_SELECTOR_GET_ATTN_BACKEND: Callable[..., Any] | None = None
_ORIG_ATTN_GET_ATTN_BACKEND: Callable[..., Any] | None = None
_ORIG_ATTENTION_GET_KV_CACHE_SPEC: Callable[..., Any] | None = None
_ORIG_ATTENTION_INIT: Callable[..., Any] | None = None


def normalize_cache_dtype(cache_dtype: str) -> str:
    """Normalize optional user aliases to canonical plugin dtype literals."""

    try:
        return CACHE_DTYPE_ALIASES[cache_dtype]
    except KeyError as exc:
        raise ValueError(f"Unsupported TileLang TQ cache dtype: {cache_dtype!r}") from exc


def _canonical_plugin_dtype(cache_dtype: str | None) -> str | None:
    # This is a permissive lookup used by our shims. Returning None means
    # "leave vLLM's normal path alone" for non-plugin dtypes.
    if cache_dtype is None:
        return None
    return CACHE_DTYPE_ALIASES.get(cache_dtype)


def install_cache_dtype_admission_shims() -> None:
    """Install process-local patches so plugin cache dtypes survive vLLM parsing.

    vLLM validates `cache_dtype` before our plugin ever gets a chance to build
    layer-local state. These shims make the parser, dtype mapper, and backend
    selector tolerate plugin-owned literals without requiring upstream changes.
    """

    global _CACHE_CONFIG_INIT_PATCHED
    global _ATTN_BACKEND_PATCHED
    global _ORIG_CACHE_CONFIG_INIT
    global _ORIG_SELECTOR_GET_ATTN_BACKEND
    global _ORIG_ATTN_GET_ATTN_BACKEND

    from vllm.config.cache import CacheConfig
    from vllm.utils import torch_utils
    from vllm.v1.attention import selector as attn_selector
    from vllm.model_executor.layers.attention import attention as attention_mod

    # Raw packed slots are byte-addressable, so we extend vLLM's global
    # string-to-dtype map with 1-byte storage dtypes for the plugin literals.
    torch_utils.STR_DTYPE_TO_TORCH_DTYPE[PLUGIN_KV_CACHE_DTYPE_3BIT] = torch.int8
    torch_utils.STR_DTYPE_TO_TORCH_DTYPE[PLUGIN_KV_CACHE_DTYPE_4BIT] = torch.int8

    if not _CACHE_CONFIG_INIT_PATCHED:
        _ORIG_CACHE_CONFIG_INIT = CacheConfig.__init__

        def _patched_cache_config_init(self, *args, **kwargs):
            # CacheConfig's constructor enforces a Literal type that does not
            # know about plugin dtypes. We temporarily substitute a built-in
            # quantized dtype that passes validation, then restore the canonical
            # plugin dtype onto the constructed config object.
            canonical: str | None = None
            args_list = list(args)

            if len(args_list) >= 3 and isinstance(args_list[2], str):
                canonical = _canonical_plugin_dtype(args_list[2])
                if canonical is not None:
                    args_list[2] = "fp8"

            if "cache_dtype" in kwargs and isinstance(kwargs["cache_dtype"], str):
                canonical = _canonical_plugin_dtype(kwargs["cache_dtype"])
                if canonical is not None:
                    kwargs["cache_dtype"] = "fp8"

            assert _ORIG_CACHE_CONFIG_INIT is not None
            _ORIG_CACHE_CONFIG_INIT(self, *tuple(args_list), **kwargs)
            if canonical is not None:
                # Restore the plugin-owned literal so downstream layer setup and
                # spec selection see the real packed-cache format.
                object.__setattr__(self, "cache_dtype", canonical)

        CacheConfig.__init__ = _patched_cache_config_init
        _CACHE_CONFIG_INIT_PATCHED = True

    if not _ATTN_BACKEND_PATCHED:
        _ORIG_SELECTOR_GET_ATTN_BACKEND = attn_selector.get_attn_backend
        _ORIG_ATTN_GET_ATTN_BACKEND = attention_mod.get_attn_backend

        def _patched_get_attn_backend(*args, **kwargs):
            # Backend auto-selection currently asserts that kv_cache_dtype is in
            # vLLM's built-in CacheDType Literal. Until the custom backend is
            # fully registered in later components, we downshift plugin dtypes
            # to "auto" for selection only. The actual plugin dtype is still
            # preserved on the layer/config objects by the CacheConfig shim.
            if "kv_cache_dtype" in kwargs:
                canonical = _canonical_plugin_dtype(kwargs["kv_cache_dtype"])
                if canonical is not None:
                    kwargs["kv_cache_dtype"] = "auto"
            elif len(args) >= 3:
                args_list = list(args)
                canonical = _canonical_plugin_dtype(args_list[2])
                if canonical is not None:
                    args_list[2] = "auto"
                args = tuple(args_list)

            assert _ORIG_SELECTOR_GET_ATTN_BACKEND is not None
            return _ORIG_SELECTOR_GET_ATTN_BACKEND(*args, **kwargs)

        attn_selector.get_attn_backend = _patched_get_attn_backend
        attention_mod.get_attn_backend = _patched_get_attn_backend
        _ATTN_BACKEND_PATCHED = True


def install_kv_spec_dispatch_shim() -> None:
    """Patch Attention.get_kv_cache_spec to consult plugin quant configs first.

    This is the bridge that lets our plugin-local `TileLangTQAttentionSpec`
    override vLLM's default `FullAttentionSpec` logic without editing vLLM
    source. Once installed, any Attention layer whose quant config implements
    `get_kv_cache_spec(...)` gets a first chance to provide a custom spec.
    """

    global _KV_SPEC_PATCHED
    global _ORIG_ATTENTION_GET_KV_CACHE_SPEC

    from vllm.model_executor.layers.attention.attention import Attention

    if _KV_SPEC_PATCHED:
        return

    _ORIG_ATTENTION_GET_KV_CACHE_SPEC = Attention.get_kv_cache_spec

    def _patched_get_kv_cache_spec(self, vllm_config):
        # If the quant config knows how to build a plugin-specific KV spec,
        # use it; otherwise fall back to vLLM's native behavior unchanged.
        if self.quant_config is not None and hasattr(self.quant_config, "get_kv_cache_spec"):
            custom = self.quant_config.get_kv_cache_spec(self, vllm_config)
            if custom is not None:
                return custom
        assert _ORIG_ATTENTION_GET_KV_CACHE_SPEC is not None
        return _ORIG_ATTENTION_GET_KV_CACHE_SPEC(self, vllm_config)

    Attention.get_kv_cache_spec = _patched_get_kv_cache_spec
    _KV_SPEC_PATCHED = True


def install_custom_backend_name_shim() -> None:
    """Let the plugin backend keep its canonical name on branches that expect an enum key.

    The current vLLM Attention layer records `AttentionBackendEnum[get_name()]`
    during initialization. Our backend is registered under `CUSTOM`, but its
    plugin-owned diagnostic name remains `TILELANG_TQ`. This shim temporarily
    aliases that name to `CUSTOM` during `Attention.__init__` only.
    """

    global _CUSTOM_BACKEND_NAME_PATCHED
    global _ORIG_ATTENTION_INIT

    from vllm.config import get_current_vllm_config
    from vllm.model_executor.layers.attention.attention import Attention
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if _CUSTOM_BACKEND_NAME_PATCHED:
        return

    _ORIG_ATTENTION_INIT = Attention.__init__

    def _patched_attention_init(self, *args, **kwargs):
        from tilelang_turboquant.backend.backend import TileLangTQAttentionBackend

        should_alias = False
        target_backend = kwargs.get("attn_backend")
        if target_backend is TileLangTQAttentionBackend:
            should_alias = True
        elif target_backend is None:
            try:
                should_alias = (
                    get_current_vllm_config().attention_config.backend
                    == AttentionBackendEnum.CUSTOM
                )
            except Exception:
                should_alias = False

        if not should_alias:
            assert _ORIG_ATTENTION_INIT is not None
            return _ORIG_ATTENTION_INIT(self, *args, **kwargs)

        original_get_name = TileLangTQAttentionBackend.get_name
        TileLangTQAttentionBackend.get_name = staticmethod(lambda: "CUSTOM")
        try:
            assert _ORIG_ATTENTION_INIT is not None
            return _ORIG_ATTENTION_INIT(self, *args, **kwargs)
        finally:
            TileLangTQAttentionBackend.get_name = original_get_name

    Attention.__init__ = _patched_attention_init
    _CUSTOM_BACKEND_NAME_PATCHED = True
