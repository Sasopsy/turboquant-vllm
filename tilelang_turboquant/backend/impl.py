"""C5 runnable reference attention implementation for packed TileLang TQ cache."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import torch

from tilelang_turboquant.backend.metadata import TileLangTQMetadata
from tilelang_turboquant.config import SlotLayout, TileLangTQConfig, get_variant_by_dtype_str
from tilelang_turboquant.quantization.compat import normalize_cache_dtype

from vllm.v1.attention.backend import AttentionImpl, AttentionType


def _float16_to_bytes(value: float) -> torch.Tensor:
    return torch.tensor([value], dtype=torch.float16).view(torch.uint8).clone()


def _bytes_to_float16(data: torch.Tensor) -> float:
    return float(data.to(torch.uint8).clone().view(torch.float16)[0].item())


def _pack_unsigned(values: torch.Tensor, bits: int, byte_len: int) -> torch.Tensor:
    out = torch.zeros(byte_len, dtype=torch.uint8)
    bit_cursor = 0
    for value in values.to(torch.int64).tolist():
        remaining = bits
        current = int(value)
        while remaining > 0:
            byte_index = bit_cursor // 8
            bit_index = bit_cursor % 8
            write_bits = min(remaining, 8 - bit_index)
            mask = (1 << write_bits) - 1
            out[byte_index] |= ((current & mask) << bit_index)
            current >>= write_bits
            remaining -= write_bits
            bit_cursor += write_bits
    return out


def _unpack_unsigned(data: torch.Tensor, count: int, bits: int) -> torch.Tensor:
    values: list[int] = []
    bit_cursor = 0
    raw = data.to(torch.int64).tolist()
    for _ in range(count):
        remaining = bits
        value = 0
        shift = 0
        while remaining > 0:
            byte_index = bit_cursor // 8
            bit_index = bit_cursor % 8
            read_bits = min(remaining, 8 - bit_index)
            chunk = (raw[byte_index] >> bit_index) & ((1 << read_bits) - 1)
            value |= chunk << shift
            shift += read_bits
            remaining -= read_bits
            bit_cursor += read_bits
        values.append(value)
    return torch.tensor(values, dtype=torch.long)


def _pack_sign_bits(signs: torch.Tensor, byte_len: int) -> torch.Tensor:
    encoded = (signs.to(torch.uint8) > 0).to(torch.long)
    return _pack_unsigned(encoded, 1, byte_len)


def _unpack_sign_bits(data: torch.Tensor, count: int) -> torch.Tensor:
    return _unpack_unsigned(data, count, 1).to(torch.float32).mul_(2.0).sub_(1.0)


def _reshape_output_like(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 2:
        return output.view(output.shape[0], -1, output.shape[-1] // output.shape[1])
    return output


class TileLangTQAttentionImpl(AttentionImpl[TileLangTQMetadata]):
    """Reference packed-cache implementation used until TileLang kernels land."""

    supports_quant_query_input = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        del alibi_slopes, sliding_window, logits_soft_cap
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads or num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = normalize_cache_dtype(kv_cache_dtype)
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.attn_type = attn_type

        variant = get_variant_by_dtype_str(self.kv_cache_dtype).name
        self.tq_config = TileLangTQConfig.from_variant_name(variant, head_size)
        self.slot_layout = SlotLayout.from_config(self.tq_config)
        self.max_num_kv_splits = 1
        self._continuation_decode_threshold = 128

    def process_weights_after_loading(self, act_dtype: torch.dtype) -> None:
        del act_dtype

    def _validate_runtime_buffers(self, layer) -> None:
        required = [
            "_tq_key_centroids",
            "_tq_key_midpoints",
            "_tq_value_centroids",
            "_tq_value_midpoints",
            "_tq_rotation",
            "_tq_S_matrix",
        ]
        missing = [name for name in required if not hasattr(layer, name)]
        if missing:
            raise ValueError(f"Layer is missing required TQ runtime buffers: {missing}")

    def _encode_component(
        self,
        vec: torch.Tensor,
        centroids: torch.Tensor,
        S_matrix: torch.Tensor,
        mse_bits: int,
        mse_size: int,
        norm_size: int,
        qjl_bits_size: int,
        qjl_gamma_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        norm = float(vec.abs().max().item())
        if norm <= 0.0:
            norm = 1e-6
        normalized = (vec / norm).to(torch.float32)
        dists = torch.abs(normalized[:, None] - centroids[None, :])
        indices = torch.argmin(dists, dim=1)
        quantized = centroids[indices]
        residual = normalized - quantized
        projection = residual @ S_matrix
        gamma = float(projection.abs().mean().item())
        signs = (projection >= 0).to(torch.uint8)
        return (
            _pack_unsigned(indices, mse_bits, mse_size),
            _float16_to_bytes(norm)[:norm_size],
            _pack_sign_bits(signs, qjl_bits_size),
            _float16_to_bytes(gamma)[:qjl_gamma_size],
        )

    def _decode_component(
        self,
        slot: torch.Tensor,
        *,
        mse_offset: int,
        mse_size: int,
        norm_offset: int,
        norm_size: int,
        qjl_bits_offset: int,
        qjl_bits_size: int,
        qjl_gamma_offset: int,
        qjl_gamma_size: int,
        centroids: torch.Tensor,
        mse_bits: int,
        rotation_t: torch.Tensor,
        S_matrix: torch.Tensor,
    ) -> torch.Tensor:
        indices = _unpack_unsigned(
            slot[mse_offset : mse_offset + mse_size],
            self.head_size,
            mse_bits,
        )
        quantized = centroids[indices].to(torch.float32)
        norm = _bytes_to_float16(slot[norm_offset : norm_offset + norm_size])
        if qjl_bits_size > 0 and qjl_gamma_size > 0:
            signs = _unpack_sign_bits(
                slot[qjl_bits_offset : qjl_bits_offset + qjl_bits_size],
                self.head_size,
            )
            gamma = _bytes_to_float16(
                slot[qjl_gamma_offset : qjl_gamma_offset + qjl_gamma_size]
            )
            correction = gamma * (signs @ S_matrix.transpose(0, 1)) / float(self.head_size)
            normalized = quantized + correction
        else:
            normalized = quantized
        rotated = norm * normalized
        return (rotated @ rotation_t).to(torch.float32)

    def _encode_slot(
        self,
        layer,
        key_vec: torch.Tensor,
        value_vec: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_runtime_buffers(layer)
        rotation = layer._tq_rotation.to(torch.float32)
        S_matrix = layer._tq_S_matrix.to(torch.float32)
        key_rot = key_vec.to(torch.float32) @ rotation
        value_rot = value_vec.to(torch.float32) @ rotation
        slot = torch.zeros(self.tq_config.slot_size_aligned, dtype=torch.uint8, device=key_vec.device)

        key_mse, key_norm, key_qjl_bits, key_qjl_gamma = self._encode_component(
            key_rot,
            layer._tq_key_centroids.to(torch.float32),
            S_matrix,
            self.tq_config.key_mse_bits,
            self.slot_layout.key_mse_size,
            self.slot_layout.key_norm_size,
            self.slot_layout.key_qjl_bits_size,
            self.slot_layout.key_qjl_gamma_size,
        )
        value_mse, value_norm, value_qjl_bits, value_qjl_gamma = self._encode_component(
            value_rot,
            layer._tq_value_centroids.to(torch.float32),
            S_matrix,
            self.tq_config.value_mse_bits,
            self.slot_layout.value_mse_size,
            self.slot_layout.value_norm_size,
            self.slot_layout.value_qjl_bits_size,
            self.slot_layout.value_qjl_gamma_size,
        )

        slot[self.slot_layout.key_mse_offset : self.slot_layout.key_mse_offset + self.slot_layout.key_mse_size] = key_mse
        slot[self.slot_layout.key_norm_offset : self.slot_layout.key_norm_offset + self.slot_layout.key_norm_size] = key_norm
        slot[self.slot_layout.key_qjl_bits_offset : self.slot_layout.key_qjl_bits_offset + self.slot_layout.key_qjl_bits_size] = key_qjl_bits
        slot[self.slot_layout.key_qjl_gamma_offset : self.slot_layout.key_qjl_gamma_offset + self.slot_layout.key_qjl_gamma_size] = key_qjl_gamma
        slot[self.slot_layout.value_mse_offset : self.slot_layout.value_mse_offset + self.slot_layout.value_mse_size] = value_mse
        slot[self.slot_layout.value_norm_offset : self.slot_layout.value_norm_offset + self.slot_layout.value_norm_size] = value_norm
        slot[self.slot_layout.value_qjl_bits_offset : self.slot_layout.value_qjl_bits_offset + self.slot_layout.value_qjl_bits_size] = value_qjl_bits
        slot[self.slot_layout.value_qjl_gamma_offset : self.slot_layout.value_qjl_gamma_offset + self.slot_layout.value_qjl_gamma_size] = value_qjl_gamma
        return slot

    def _decode_slot(self, layer, slot: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rotation_t = getattr(layer, "_tq_rotation_t", layer._tq_rotation).to(torch.float32)
        S_matrix = layer._tq_S_matrix.to(torch.float32)
        key = self._decode_component(
            slot,
            mse_offset=self.slot_layout.key_mse_offset,
            mse_size=self.slot_layout.key_mse_size,
            norm_offset=self.slot_layout.key_norm_offset,
            norm_size=self.slot_layout.key_norm_size,
            qjl_bits_offset=self.slot_layout.key_qjl_bits_offset,
            qjl_bits_size=self.slot_layout.key_qjl_bits_size,
            qjl_gamma_offset=self.slot_layout.key_qjl_gamma_offset,
            qjl_gamma_size=self.slot_layout.key_qjl_gamma_size,
            centroids=layer._tq_key_centroids.to(torch.float32),
            mse_bits=self.tq_config.key_mse_bits,
            rotation_t=rotation_t,
            S_matrix=S_matrix,
        )
        value = self._decode_component(
            slot,
            mse_offset=self.slot_layout.value_mse_offset,
            mse_size=self.slot_layout.value_mse_size,
            norm_offset=self.slot_layout.value_norm_offset,
            norm_size=self.slot_layout.value_norm_size,
            qjl_bits_offset=self.slot_layout.value_qjl_bits_offset,
            qjl_bits_size=self.slot_layout.value_qjl_bits_size,
            qjl_gamma_offset=self.slot_layout.value_qjl_gamma_offset,
            qjl_gamma_size=self.slot_layout.value_qjl_gamma_size,
            centroids=layer._tq_value_centroids.to(torch.float32),
            mse_bits=self.tq_config.value_mse_bits,
            rotation_t=rotation_t,
            S_matrix=S_matrix,
        )
        return key, value

    def _slot_index_to_coords(self, slot_index: int, block_size: int) -> tuple[int, int]:
        return divmod(slot_index, block_size)

    def _load_slot_bytes(self, kv_cache: torch.Tensor, block: int, offset: int, kv_head: int) -> torch.Tensor:
        slot = kv_cache[block, offset, kv_head]
        if slot.dtype != torch.uint8:
            slot = slot.view(torch.uint8)
        return slot

    def _load_cached_prefix(
        self,
        layer,
        kv_cache: torch.Tensor,
        block_table_row: torch.Tensor,
        seq_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = kv_cache.device
        keys = torch.zeros(seq_len, self.num_kv_heads, self.head_size, dtype=torch.float32, device=device)
        values = torch.zeros_like(keys)
        block_size = kv_cache.shape[1]
        for token_idx in range(seq_len):
            physical_block = int(block_table_row[token_idx // block_size].item())
            offset = token_idx % block_size
            for kv_head in range(self.num_kv_heads):
                slot = self._load_slot_bytes(kv_cache, physical_block, offset, kv_head)
                key_vec, value_vec = self._decode_slot(layer, slot)
                keys[token_idx, kv_head] = key_vec.to(device=device)
                values[token_idx, kv_head] = value_vec.to(device=device)
        return keys, values

    def _expand_gqa(self, tensor: torch.Tensor) -> torch.Tensor:
        if self.num_heads == self.num_kv_heads:
            return tensor
        return tensor.repeat_interleave(self.num_kv_groups, dim=1)

    def _causal_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        start_position: int,
    ) -> torch.Tensor:
        key = self._expand_gqa(key).to(query.dtype)
        value = self._expand_gqa(value).to(query.dtype)
        outputs = torch.zeros_like(query)
        for q_idx in range(query.shape[0]):
            visible = start_position + q_idx + 1
            scores = torch.einsum("hd,shd->hs", query[q_idx], key[:visible]) * self.scale
            probs = torch.softmax(scores, dim=-1).to(value.dtype)
            outputs[q_idx] = torch.einsum("hs,shd->hd", probs, value[:visible])
        return outputs

    def _query_lens(self, metadata: TileLangTQMetadata) -> list[int]:
        qsl = metadata.query_start_loc_host
        return [qsl[i + 1] - qsl[i] for i in range(len(qsl) - 1)]

    def _decode_attention(
        self,
        layer,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileLangTQMetadata,
    ) -> torch.Tensor:
        outputs = torch.zeros_like(query)
        q_lens = self._query_lens(attn_metadata)
        cursor = 0
        for req_idx, q_len in enumerate(q_lens):
            seq_len = int(attn_metadata.seq_lens[req_idx].item())
            keys, values = self._load_cached_prefix(
                layer,
                kv_cache,
                attn_metadata.block_table[req_idx],
                seq_len,
            )
            q_slice = query[cursor : cursor + q_len]
            outputs[cursor : cursor + q_len] = self._causal_attention(
                q_slice,
                keys,
                values,
                seq_len - q_len,
            )
            cursor += q_len
        return outputs

    def _prefill_attention(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileLangTQMetadata,
    ) -> torch.Tensor:
        outputs = torch.zeros_like(query)
        q_lens = self._query_lens(attn_metadata)
        cursor = 0
        for req_idx, q_len in enumerate(q_lens):
            seq_len = int(attn_metadata.seq_lens[req_idx].item())
            q_slice = query[cursor : cursor + q_len]
            k_slice = key[cursor : cursor + q_len]
            v_slice = value[cursor : cursor + q_len]
            if seq_len == q_len:
                outputs[cursor : cursor + q_len] = self._causal_attention(
                    q_slice,
                    k_slice,
                    v_slice,
                    0,
                )
            elif q_len <= self._continuation_decode_threshold:
                keys, values = self._load_cached_prefix(
                    layer,
                    kv_cache,
                    attn_metadata.block_table[req_idx],
                    seq_len,
                )
                outputs[cursor : cursor + q_len] = self._causal_attention(
                    q_slice,
                    keys,
                    values,
                    seq_len - q_len,
                )
            else:
                prefix_len = seq_len - q_len
                prefix_keys, prefix_values = self._load_cached_prefix(
                    layer,
                    kv_cache,
                    attn_metadata.block_table[req_idx],
                    prefix_len,
                )
                full_keys = torch.cat([prefix_keys, k_slice.to(torch.float32)], dim=0)
                full_values = torch.cat([prefix_values, v_slice.to(torch.float32)], dim=0)
                outputs[cursor : cursor + q_len] = self._causal_attention(
                    q_slice,
                    full_keys,
                    full_values,
                    prefix_len,
                )
            cursor += q_len
        return outputs

    def _slice_query_start_loc(self, values: Sequence[int]) -> tuple[int, ...]:
        offset = values[0]
        return tuple(value - offset for value in values)

    def _slice_decode_metadata(self, metadata: TileLangTQMetadata) -> TileLangTQMetadata:
        if metadata.num_decodes <= 0:
            raise ValueError("No decode requests available to slice")
        qsl_host = tuple(metadata.query_start_loc_host[: metadata.num_decodes + 1])
        query_start_loc = metadata.query_start_loc[: metadata.num_decodes + 1]
        seq_lens = metadata.seq_lens[: metadata.num_decodes]
        block_table = metadata.block_table[: metadata.num_decodes]
        return TileLangTQMetadata(
            seq_lens=seq_lens,
            slot_mapping=metadata.slot_mapping[: metadata.num_decode_tokens],
            block_table=block_table,
            query_start_loc=query_start_loc,
            query_start_loc_host=qsl_host,
            num_actual_tokens=metadata.num_decode_tokens,
            max_query_len=max(self._query_lens(replace(metadata, query_start_loc_host=qsl_host))) if metadata.num_decodes else 0,
            max_seq_len=int(seq_lens.max().item()) if metadata.num_decodes else 0,
            is_prefill=False,
            num_decodes=metadata.num_decodes,
            num_decode_tokens=metadata.num_decode_tokens,
            causal=metadata.causal,
        )

    def _slice_prefill_metadata(
        self,
        metadata: TileLangTQMetadata,
        num_decode_tokens: int,
        num_decodes: int,
    ) -> TileLangTQMetadata:
        qsl_host = self._slice_query_start_loc(metadata.query_start_loc_host[num_decodes:])
        query_start_loc = metadata.query_start_loc[num_decodes:] - num_decode_tokens
        seq_lens = metadata.seq_lens[num_decodes:]
        block_table = metadata.block_table[num_decodes:]
        num_prefill_tokens = metadata.num_actual_tokens - num_decode_tokens
        query_lens = [qsl_host[i + 1] - qsl_host[i] for i in range(len(qsl_host) - 1)]
        return TileLangTQMetadata(
            seq_lens=seq_lens,
            slot_mapping=metadata.slot_mapping[num_decode_tokens : num_decode_tokens + num_prefill_tokens],
            block_table=block_table,
            query_start_loc=query_start_loc,
            query_start_loc_host=qsl_host,
            num_actual_tokens=num_prefill_tokens,
            max_query_len=max(query_lens) if query_lens else 0,
            max_seq_len=int(seq_lens.max().item()) if seq_lens.numel() else 0,
            is_prefill=True,
            num_decodes=0,
            num_decode_tokens=0,
            causal=metadata.causal,
        )

    def _mixed_attention(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileLangTQMetadata,
    ) -> torch.Tensor:
        output = torch.zeros_like(query)
        decode_metadata = self._slice_decode_metadata(attn_metadata)
        prefill_metadata = self._slice_prefill_metadata(
            attn_metadata,
            attn_metadata.num_decode_tokens,
            attn_metadata.num_decodes,
        )

        if decode_metadata.num_actual_tokens > 0:
            output[: decode_metadata.num_actual_tokens] = self._decode_attention(
                layer,
                query[: decode_metadata.num_actual_tokens],
                kv_cache,
                decode_metadata,
            )
        if prefill_metadata.num_actual_tokens > 0:
            start = attn_metadata.num_decode_tokens
            end = start + prefill_metadata.num_actual_tokens
            output[start:end] = self._prefill_attention(
                layer,
                query[start:end],
                key[start:end],
                value[start:end],
                kv_cache,
                prefill_metadata,
            )
        return output

    def do_kv_cache_update(
        self,
        layer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        N = int(slot_mapping.shape[0])
        if N == 0:
            return
        if key.shape[0] < N or value.shape[0] < N:
            raise ValueError("Key/value rows must cover slot_mapping.shape[0]")
        block_size = kv_cache.shape[1]
        for row_idx in range(N):
            slot_index = int(slot_mapping[row_idx].item())
            if slot_index < 0:
                continue
            block, offset = self._slot_index_to_coords(slot_index, block_size)
            for kv_head in range(self.num_kv_heads):
                slot_bytes = self._encode_slot(layer, key[row_idx, kv_head], value[row_idx, kv_head])
                kv_cache[block, offset, kv_head].view(torch.uint8).copy_(slot_bytes)

    def forward(
        self,
        layer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TileLangTQMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del output_scale, output_block_scale
        if output is None:
            output = torch.zeros_like(query)
        else:
            output.zero_()

        if attn_metadata is None:
            return output

        N = int(attn_metadata.num_actual_tokens)
        if N <= 0:
            return output

        q = query[:N]
        if not attn_metadata.is_prefill:
            attn_out = self._decode_attention(layer, q, kv_cache, attn_metadata)
        elif attn_metadata.num_decodes == 0:
            attn_out = self._prefill_attention(layer, q, key[:N], value[:N], kv_cache, attn_metadata)
        else:
            attn_out = self._mixed_attention(layer, q, key[:N], value[:N], kv_cache, attn_metadata)

        output[:N].copy_(attn_out.to(dtype=output.dtype))
        return output
