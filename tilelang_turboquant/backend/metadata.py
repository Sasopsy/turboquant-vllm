"""C4 metadata contract and builder for the TileLang TQ backend."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Sequence

import torch

from vllm.v1.attention.backend import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata,
    split_decodes_and_prefills,
)


@dataclass
class TileLangTQMetadata:
    """Per-step metadata consumed by the TileLang TQ attention impl."""

    seq_lens: torch.Tensor
    slot_mapping: torch.Tensor
    block_table: torch.Tensor
    query_start_loc: torch.Tensor
    query_start_loc_host: Sequence[int]
    num_actual_tokens: int
    max_query_len: int
    max_seq_len: int
    is_prefill: bool
    num_decodes: int
    num_decode_tokens: int
    causal: bool = True


class TileLangTQMetadataBuilder(AttentionMetadataBuilder[TileLangTQMetadata]):
    """Build backend-local metadata from vLLM's common scheduler metadata."""

    _cudagraph_support = getattr(
        AttentionCGSupport,
        "UNIFORM_BATCH",
        AttentionCGSupport.NEVER,
    )
    supports_update_block_table = True

    def __init__(
        self,
        kv_cache_spec,
        layer_names: list[str],
        vllm_config,
        device: torch.device,
    ) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        if hasattr(self, "_init_reorder_batch_threshold"):
            self._init_reorder_batch_threshold(
                reorder_batch_threshold=1,
                supports_spec_as_decode=False,
            )
        else:
            self.reorder_batch_threshold = 1

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TileLangTQMetadata:
        del common_prefix_len, fast_build
        cam = common_attn_metadata
        threshold = self.reorder_batch_threshold or 1
        num_decodes, _, num_decode_tokens, _ = split_decodes_and_prefills(
            cam,
            decode_threshold=threshold,
        )

        return TileLangTQMetadata(
            seq_lens=cam.seq_lens,
            slot_mapping=cam.slot_mapping,
            block_table=cam.block_table_tensor,
            query_start_loc=cam.query_start_loc,
            query_start_loc_host=tuple(int(v) for v in cam.query_start_loc_cpu.tolist()),
            num_actual_tokens=cam.num_actual_tokens,
            max_query_len=cam.max_query_len,
            max_seq_len=cam.max_seq_len,
            is_prefill=cam.max_query_len > 1,
            num_decodes=num_decodes,
            num_decode_tokens=num_decode_tokens,
            causal=cam.causal,
        )

    def update_block_table(
        self,
        metadata: TileLangTQMetadata,
        blk_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> TileLangTQMetadata:
        return replace(metadata, block_table=blk_table, slot_mapping=slot_mapping)

    def build_for_cudagraph_capture(
        self,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> TileLangTQMetadata:
        metadata = self.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
        )
        return replace(
            metadata,
            seq_lens=torch.ones_like(metadata.seq_lens),
            max_seq_len=1,
        )
