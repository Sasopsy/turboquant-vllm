"""C3 KV-cache quantization method for TileLang TurboQuant plugin layers."""

from __future__ import annotations

from types import SimpleNamespace

import torch
from torch import nn

from tilelang_turboquant.config import TileLangTQConfig
from tilelang_turboquant.offline import (
    CodebookSpec,
    QJLSpec,
    RotationSpec,
    get_codebook,
    get_rotation,
    get_s_matrix,
)
from tilelang_turboquant.quantization.compat import VARIANT_BY_CACHE_DTYPE, normalize_cache_dtype

from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod

DEFAULT_ROTATION_MODE = "qr"
DEFAULT_ROTATION_SEED = 0
DEFAULT_QJL_SEED = 0


def _extract_optional_scale(param: torch.nn.Parameter, default: float) -> float:
    # vLLM uses negative sentinels for "not loaded from checkpoint". The TQ
    # path treats those as "use the default scale", but explicitly rejects 0
    # because a zero KV scale would make the runtime state invalid.
    value = float(param.item())
    if value < 0.0:
        return default
    if value == 0.0:
        raise ValueError("Loaded zero kv scale for TileLang TQ layer")
    return value


def _select_runtime_matrix_dtype(layer: torch.nn.Module) -> torch.dtype:
    # Keep runtime matrices in float32 for now. This stays conservative until
    # the later backend/kernel components validate narrower dtypes.
    return torch.float32


def _rotation_mode_from_runtime_config() -> str:
    # Placeholder for future runtime-config plumbing. C3 uses a fixed default
    # today so that artifact registration is deterministic and testable.
    return DEFAULT_ROTATION_MODE


def _rotation_seed_from_runtime_config() -> int:
    # Same story as rotation mode: this is where a later config surface can
    # feed seeds into C2 artifact generation without changing the lifecycle.
    return DEFAULT_ROTATION_SEED


def _qjl_seed() -> int:
    # Dedicated hook for the seeded QJL projection matrix.
    return DEFAULT_QJL_SEED


def _validate_variant_match(cache_dtype: str, expected_variant: str) -> str:
    # The quantization config decides "which algorithm family" while the cache
    # dtype decides "which packed slot layout". They must resolve to the same
    # variant or the layer would allocate/interpret KV slots incorrectly.
    normalized = normalize_cache_dtype(cache_dtype)
    actual_variant = VARIANT_BY_CACHE_DTYPE[normalized]
    if actual_variant != expected_variant:
        raise ValueError(
            f"Quantization variant {expected_variant!r} is incompatible with "
            f"kv_cache_dtype={cache_dtype!r} (normalized={normalized!r})."
        )
    return normalized


def _set_or_register_buffer(
    layer: torch.nn.Module,
    name: str,
    tensor: torch.Tensor,
    *,
    persistent: bool = True,
) -> None:
    # This helper lets us update an already-registered buffer in-place on
    # repeat calls while preserving the same state-dict semantics that vLLM
    # expects from `register_buffer`.
    if name in layer._buffers:
        layer._buffers[name] = tensor
    else:
        layer.register_buffer(name, tensor, persistent=persistent)

    if persistent:
        layer._non_persistent_buffers_set.discard(name)
    else:
        layer._non_persistent_buffers_set.add(name)


def _register_decode_scratch_buffers(
    layer: torch.nn.Module,
    cfg: TileLangTQConfig,
    device: torch.device,
) -> None:
    # These buffers stand in for later decode workspaces. C3 owns the timing:
    # they must exist before `profile_run()` so vLLM's memory profiler sees
    # them, but they should stay non-persistent because they are runtime
    # scratch, not checkpoint state.
    workspace_dtype = _select_runtime_matrix_dtype(layer)
    num_heads = getattr(layer, "num_heads", getattr(layer, "num_kv_heads", 1))
    _set_or_register_buffer(
        layer,
        "_tq_mid_o_buf",
        torch.zeros((1, 1, num_heads, cfg.head_dim), dtype=workspace_dtype, device=device),
        persistent=False,
    )
    _set_or_register_buffer(
        layer,
        "_tq_lse_buf",
        torch.zeros((1, 1, num_heads), dtype=torch.float32, device=device),
        persistent=False,
    )
    _set_or_register_buffer(
        layer,
        "_tq_output_buf",
        torch.zeros((1, num_heads, cfg.head_dim), dtype=workspace_dtype, device=device),
        persistent=False,
    )


class TileLangTQKVCacheMethod(BaseKVCacheMethod):
    """Plugin-scoped KV-cache method that registers C2 artifacts before profiling.

    In vLLM terms, this object participates in the weight-loading lifecycle.
    It creates temporary checkpoint-scale placeholders up front, then replaces
    them with permanent runtime buffers plus all C2 artifacts once loading is
    finished and before memory profiling happens.
    """

    def create_weights(self, layer: torch.nn.Module):
        # Match vLLM's expected temporary scalar Parameters so the normal
        # checkpoint-loading path can populate them if scales are present.
        layer.q_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.prob_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # If the temp parameters are already gone, this layer has already been
        # finalized (or reloaded), so there is nothing left to do.
        if not hasattr(layer, "k_scale"):
            return

        # Re-derive the exact C1 config for this layer so every downstream
        # buffer shape, codebook request, and slot-size-dependent decision
        # stays anchored to the plugin's canonical variant math.
        cfg = TileLangTQConfig.from_variant_name(
            self.quant_config.variant_name,
            layer.head_size,
        )
        _validate_variant_match(layer.kv_cache_dtype, self.quant_config.variant_name)

        # Promote checkpoint-loaded scales into the permanent buffers that
        # vLLM Attention layers already carry in their state dict.
        k_scale = _extract_optional_scale(layer.k_scale, default=1.0)
        v_scale = _extract_optional_scale(layer.v_scale, default=1.0)

        layer._k_scale.fill_(k_scale)
        layer._v_scale.fill_(v_scale)
        layer._q_scale.fill_(1.0)
        layer._prob_scale.fill_(1.0)
        layer._k_scale_float = k_scale
        layer._v_scale_float = v_scale
        layer._q_scale_float = 1.0
        layer._prob_scale_float = 1.0

        device = layer._k_scale.device
        runtime_dtype = _select_runtime_matrix_dtype(layer)

        # Pull the canonical CPU artifacts from C2, then move them onto the
        # layer's final device as registered runtime buffers. This is the key
        # "CPU canonical -> GPU runtime" handoff point for the plugin.
        rotation, rotation_t, is_symmetric = get_rotation(
            RotationSpec(
                head_dim=cfg.head_dim,
                mode=_rotation_mode_from_runtime_config(),
                seed=_rotation_seed_from_runtime_config(),
                allow_hadamard_padding=False,
            )
        )
        key_centroids, key_midpoints = get_codebook(
            CodebookSpec(head_dim=cfg.head_dim, mse_bits=cfg.key_mse_bits)
        )
        value_centroids, value_midpoints = get_codebook(
            CodebookSpec(head_dim=cfg.head_dim, mse_bits=cfg.value_mse_bits)
        )
        s_matrix = get_s_matrix(QJLSpec(head_dim=cfg.head_dim, seed=_qjl_seed()))

        # Key/value codebooks are registered separately even when they happen
        # to have the same bit-width. That keeps the plugin's runtime contract
        # explicit and avoids accidental coupling in later components.
        _set_or_register_buffer(
            layer,
            "_tq_key_centroids",
            key_centroids.to(device=device, dtype=torch.float32),
        )
        _set_or_register_buffer(
            layer,
            "_tq_key_midpoints",
            key_midpoints.to(device=device, dtype=torch.float32),
        )
        _set_or_register_buffer(
            layer,
            "_tq_value_centroids",
            value_centroids.to(device=device, dtype=torch.float32),
        )
        _set_or_register_buffer(
            layer,
            "_tq_value_midpoints",
            value_midpoints.to(device=device, dtype=torch.float32),
        )
        _set_or_register_buffer(
            layer,
            "_tq_rotation",
            rotation.to(device=device, dtype=runtime_dtype),
        )
        if not is_symmetric:
            # QR/Hadamard currently return non-symmetric transforms, so decode
            # gets an explicit transpose buffer rather than computing it lazily.
            _set_or_register_buffer(
                layer,
                "_tq_rotation_t",
                rotation_t.to(device=device, dtype=runtime_dtype),
            )
        _set_or_register_buffer(
            layer,
            "_tq_S_matrix",
            s_matrix.to(device=device, dtype=runtime_dtype),
        )

        # Scratch buffers must also exist before profiling so they are part of
        # the visible runtime memory footprint.
        _register_decode_scratch_buffers(layer, cfg, device)

        # Once scales are copied and TQ buffers are registered, the temporary
        # checkpoint Parameters are no longer part of the runtime contract.
        del layer.k_scale
        del layer.v_scale
        del layer.q_scale
        del layer.prob_scale


def build_test_vllm_config(block_size: int = 16):
    """Small helper for tests that need the `cache_config.block_size` surface.

    The real vLLM config object is much larger; tests only need the field that
    `get_kv_cache_spec()` reads to build a plugin-local attention spec.
    """

    return SimpleNamespace(cache_config=SimpleNamespace(block_size=block_size))
