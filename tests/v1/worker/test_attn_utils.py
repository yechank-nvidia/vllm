# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVQuantMode,
    SlidingWindowSpec,
)
from vllm.v1.worker.gpu.attn_utils import _reshape_kv_cache, build_attn_metadata
from vllm.v1.worker.utils import AttentionGroup


class FakeFlashAttentionBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


class FakeHNDFlashAttentionBackend(FakeFlashAttentionBackend):
    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 3, 2, 4)


def test_reshape_padded_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 256

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )


def test_reshape_padded_hnd_flash_attention_kv_cache_strides_by_page():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=3,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=1024,
    )
    assert spec.real_page_size_bytes == 768

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeHNDFlashAttentionBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 3, 2)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == spec.real_page_size_bytes // 2 // 4
    assert kv_cache.stride(2) == 2
    assert kv_cache.stride(3) == spec.block_size * spec.head_size
    assert kv_cache[1, 0].storage_offset() == spec.page_size_bytes // 4
    assert (
        kv_cache[1, 1].storage_offset()
        == (spec.page_size_bytes + spec.real_page_size_bytes // 2) // 4
    )
    assert (
        kv_cache[1, 1, 3, 2].storage_offset()
        == (
            spec.page_size_bytes
            + spec.real_page_size_bytes // 2
            + 3 * spec.head_size * 4
            + 2 * spec.block_size * spec.head_size * 4
        )
        // 4
    )


class FakeDiffKVBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, num_kv_heads, head_size * 2)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3)


def test_reshape_padded_diff_kv_cache_does_not_infer_kv_dim():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=2,
        dtype=torch.float32,
        page_size_padded=384,
    )

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakeDiffKVBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "auto",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 16, 1, 4)
    assert kv_cache.stride(0) == spec.page_size_bytes // 4
    assert kv_cache.stride(1) == 4


class FakePerTokenScaleBackend:
    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, block_size, num_kv_heads, head_size + 4)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        assert not include_num_layers_dimension
        return (0, 1, 2, 3, 4)


def test_reshape_padded_quantized_kv_cache_preserves_scale_stride():
    num_blocks = 3
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=4,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        page_size_padded=384,
    )
    assert spec.real_page_size_bytes == 128
    assert spec.page_size_bytes == 384

    raw_tensors = {
        "layer": torch.zeros(spec.page_size_bytes * num_blocks, dtype=torch.int8)
    }
    attn_groups = [
        AttentionGroup(
            backend=FakePerTokenScaleBackend,
            layer_names=["layer"],
            kv_cache_spec=spec,
            kv_cache_group_id=0,
        )
    ]

    kv_cache = _reshape_kv_cache(
        attn_groups,
        raw_tensors,
        "int8_per_token_head",
        [spec.block_size],
        {},
    )["layer"]

    assert kv_cache.shape == (num_blocks, 2, 16, 1, 8)
    assert kv_cache.stride(0) == spec.page_size_bytes
    assert kv_cache.stride(1) == 16 * 1 * 8
    assert kv_cache[1, 1].storage_offset() == spec.page_size_bytes + 16 * 1 * 8


class FakeReuseBuilder:
    """Records build() and update_block_table() calls."""

    def __init__(self, kv_cache_spec, calls, can_reuse=True):
        self.kv_cache_spec = kv_cache_spec
        self.calls = calls
        self.supports_update_block_table = can_reuse

    def build(self, common_prefix_len, common_attn_metadata):
        self.calls.append(("build", self))
        return SimpleNamespace(
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            built_by=self,
        )

    def build_for_cudagraph_capture(self, common_attn_metadata):
        return self.build(0, common_attn_metadata)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        self.calls.append(("reuse", self))
        return SimpleNamespace(
            block_table=blk_table, slot_mapping=slot_mapping, built_by=metadata.built_by
        )


def _reuse_inputs(can_reuse=True):
    full = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=8, dtype=torch.bfloat16
    )
    sliding = SlidingWindowSpec(
        block_size=16,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.bfloat16,
        sliding_window=64,
    )
    # Hybrid grouping: full, sliding, sliding, full, sliding.
    specs = [full, sliding, sliding, full, sliding]
    calls: list = []
    attn_groups = []
    for i, spec in enumerate(specs):
        group = AttentionGroup(
            backend=FakeFlashAttentionBackend,  # type: ignore[arg-type]
            layer_names=[f"layer.{i}"],
            kv_cache_spec=spec,
            kv_cache_group_id=i,
        )
        group.metadata_builders = [FakeReuseBuilder(spec, calls, can_reuse)]  # type: ignore[list-item]
        attn_groups.append([group])
    kv_cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer.{i}"], spec) for i, spec in enumerate(specs)
        ],
    )
    num_reqs, num_tokens = 2, 3
    block_tables = [torch.full((num_reqs, 4), i, dtype=torch.int32) for i in range(5)]
    slot_mappings = torch.arange(5 * num_tokens).view(5, num_tokens)
    kwargs = dict(
        attn_groups=attn_groups,
        num_reqs=num_reqs,
        num_tokens=num_tokens,
        query_start_loc_gpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        query_start_loc_cpu=torch.tensor([0, 1, 3], dtype=torch.int32),
        max_query_len=2,
        seq_lens=torch.tensor([5, 9], dtype=torch.int32),
        max_seq_len=9,
        block_tables=block_tables,
        slot_mappings=slot_mappings,
        kv_cache_config=kv_cache_config,
    )
    builders = [g[0].metadata_builders[0] for g in attn_groups]
    return kwargs, calls, builders


@pytest.mark.parametrize("for_cudagraph_capture", [False, True])
def test_build_attn_metadata_reuses_metadata_of_the_same_spec(for_cudagraph_capture):
    kwargs, calls, b = _reuse_inputs()
    metadata = build_attn_metadata(
        **kwargs, for_cudagraph_capture=for_cudagraph_capture
    )
    # One build per spec; later groups of a spec derive from its first metadata.
    assert calls == [
        ("build", b[0]),
        ("build", b[1]),
        ("reuse", b[2]),
        ("reuse", b[3]),
        ("reuse", b[4]),
    ]
    for i in range(5):
        m = metadata[f"layer.{i}"]
        # Every group keeps its own block table and slot mapping.
        assert m.block_table is kwargs["block_tables"][i]
        assert torch.equal(m.slot_mapping, kwargs["slot_mappings"][i])
        assert m.built_by is b[0 if i in (0, 3) else 1]


def test_build_attn_metadata_builds_when_the_builder_does_not_reuse():
    kwargs, calls, b = _reuse_inputs(can_reuse=False)
    build_attn_metadata(**kwargs)
    assert calls == [("build", x) for x in b]


def test_build_attn_metadata_builds_every_group_when_not_causal():
    kwargs, calls, b = _reuse_inputs()
    build_attn_metadata(**kwargs, causal=False)
    assert calls == [("build", x) for x in b]


def test_build_attn_metadata_builds_every_group_with_model_specific_metadata():
    kwargs, calls, b = _reuse_inputs()
    model_specific = SimpleNamespace(
        get_extra_common_attn_kwargs=lambda i, num_reqs: {},
        get_extra_attn_kwargs=lambda builder, num_reqs: {},
    )
    build_attn_metadata(**kwargs, model_specific_attn_metadata=model_specific)
    assert calls == [("build", x) for x in b]
