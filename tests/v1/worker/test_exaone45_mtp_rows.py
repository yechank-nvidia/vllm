# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.models import exaone4_5_mtp
from vllm.model_executor.models.exaone4 import Exaone4Attention, Exaone4DecoderLayer
from vllm.model_executor.models.exaone4_5_mtp import (
    Exaone4_5_MTP,
    Exaone4_5MultiTokenPredictor,
    _SelectedRowsPredictor,
)
from vllm.v1.worker.gpu.input_batch import InputBatch, InputBuffers
from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator as spec_module
from vllm.v1.worker.gpu.spec_decode.mtp.speculator import MTPSpeculator


def _empty_module(cls):
    module = cls.__new__(cls)
    nn.Module.__init__(module)
    return module


@pytest.mark.parametrize("fused", [False, True])
def test_prefill_compacts_exaone_output_after_full_attention(monkeypatch, fused):
    monkeypatch.setattr(
        exaone4_5_mtp,
        "get_pp_group",
        lambda: SimpleNamespace(is_first_rank=True, is_last_rank=True),
    )
    forward_context = Mock(side_effect=lambda *args, **kwargs: nullcontext())
    monkeypatch.setattr(spec_module, "set_forward_context", forward_context)

    attn = _empty_module(Exaone4Attention)
    attn.q_size = attn.kv_size = attn.head_dim = 4
    attn.num_heads = attn.num_kv_heads = 1
    attn.qkv_proj = Mock(side_effect=lambda x: (x.repeat(1, 3), None))
    attn.sliding_window = 8
    attn.use_rope = True
    attn.rotary_emb = Mock(
        side_effect=lambda pos, q, k: (q + pos[:, None], k + pos[:, None]),
    )
    # Make each result depend on earlier KV rows as a causal attention stand-in.
    attn.attn = Mock(side_effect=lambda q, k, v, **kwargs: q + (k + v).cumsum(dim=0))
    attn.fuse_qk_norm_rope_kv = fused
    if fused:
        # The fused QK-norm, RoPE and KV-cache write op, same stand-in math.
        attn.q_norm = attn.k_norm = SimpleNamespace(weight=None, variance_epsilon=0.0)

        def qk_norm_rope_kv(qkv, pos, *args):
            q, k, v = qkv.split([4, 4, 4], dim=-1)
            return q + pos[:, None], k + pos[:, None], v, None

        attn.attn.qk_norm_rope_kv_cache_update = Mock(side_effect=qk_norm_rope_kv)
    else:
        attn.q_norm = attn.k_norm = nn.Identity()
    attn.o_proj = Mock(side_effect=lambda x: (x * 3, None))

    layer = _empty_module(Exaone4DecoderLayer)
    layer.self_attn = attn
    layer.post_attention_layernorm = layer.post_feedforward_layernorm = nn.Identity()
    layer.mlp = Mock(side_effect=torch.square)

    predictor = _empty_module(Exaone4_5MultiTokenPredictor)
    predictor.do_not_compile = True
    predictor.num_mtp_layers = 1
    predictor.layers = nn.ModuleList([layer])
    predictor.embed_tokens = Mock(side_effect=lambda ids: ids[:, None].expand(-1, 4))
    predictor.pre_fc_norm_embedding = predictor.pre_fc_norm_hidden = nn.Identity()
    predictor.fc = Mock(side_effect=lambda x: x[:, :4] + x[:, 4:])
    predictor.norm = nn.Identity()

    model = _empty_module(Exaone4_5_MTP)
    model.do_not_compile = True
    model.model = predictor
    compact = _empty_module(_SelectedRowsPredictor)
    compact.do_not_compile = True
    compact.model = predictor
    object.__setattr__(model, "forward_with_selected_rows", compact)

    runner = object.__new__(MTPSpeculator)
    runner.model = model
    runner.supports_mm_inputs = True
    runner.inputs_embeds = torch.zeros(9, 4)
    runner.vllm_config = None
    runner.last_token_indices = torch.tensor([1, 7, 0])
    runner.idx_mapping = torch.arange(3)
    runner.temperature = torch.zeros(3)
    runner.seeds = torch.zeros(3, dtype=torch.int64)
    runner.current_draft_step = torch.tensor(0)
    runner.draft_logits = None
    runner.draft_tokens = torch.zeros(3, 1, dtype=torch.int64)
    runner.sample_draft = Mock(side_effect=lambda h, *args: h.argmax(-1))
    hidden = torch.arange(36, dtype=torch.float32).reshape(9, 4)
    positions = torch.tensor([20, 21, 22, 23, 4, 5, 6, 7, 0])

    for select_output_rows in (False, True):
        runner.hidden_states = hidden.clone()
        runner.input_buffers = SimpleNamespace(
            input_ids=torch.arange(9), positions=positions.clone()
        )
        runner._prefill(2, 9, {}, {}, None, select_output_rows=select_output_rows)
        assert forward_context.call_args.kwargs["num_tokens"] == 9

    for full_input, compact_input in zip(
        attn.attn.call_args_list[0].args, attn.attn.call_args_list[1].args
    ):
        assert full_input.shape == (9, 4)
        torch.testing.assert_close(compact_input, full_input)

    for op in (attn.o_proj, layer.mlp):
        full_input, compact_input = [call.args[0] for call in op.call_args_list]
        assert full_input.shape == (9, 4)
        torch.testing.assert_close(compact_input, full_input[[1, 7]])

    for full_arg, compact_arg in zip(
        runner.sample_draft.call_args_list[0].args[:2],
        runner.sample_draft.call_args_list[1].args[:2],
    ):
        torch.testing.assert_close(compact_arg, full_arg)
    torch.testing.assert_close(
        runner.hidden_states[:2], runner.sample_draft.call_args.args[0]
    )


@pytest.mark.parametrize(
    ("num_tokens", "mode", "supported", "dummy", "profile", "select_rows"),
    [
        (9, CUDAGraphMode.NONE, True, False, False, True),
        (8, CUDAGraphMode.NONE, True, False, False, False),
        (9, CUDAGraphMode.PIECEWISE, True, False, False, False),
        (9, CUDAGraphMode.FULL, True, False, False, False),
        (9, CUDAGraphMode.NONE, False, False, False, False),
        (9, CUDAGraphMode.NONE, True, True, False, False),
        (9, CUDAGraphMode.NONE, True, False, True, False),
    ],
)
def test_propose_limits_compaction_to_real_eager_prefill(
    monkeypatch, num_tokens, mode, supported, dummy, profile, select_rows
):
    runner = object.__new__(MTPSpeculator)
    runner.input_buffers = InputBuffers(2, num_tokens, torch.device("cpu"))
    batch = InputBatch.make_dummy(2, num_tokens, runner.input_buffers)
    runner.last_token_indices = torch.zeros(2, dtype=torch.int64)
    runner.current_draft_step = torch.tensor(0)
    runner.hidden_states = torch.zeros(num_tokens, 4)
    runner.max_num_reqs = 2
    runner.max_model_len = 16
    runner.num_speculative_steps = 3
    runner.dp_size, runner.dp_rank = 1, 0
    runner.eplb_state = None
    runner.model = SimpleNamespace(supports_mtp_row_selection=supported)
    runner.prefill_cudagraph_manager = Mock()
    runner.decode_cudagraph_manager = None
    runner.draft_tokens = torch.zeros(2, 3, dtype=torch.int64)
    runner._copy_request_inputs = Mock()
    runner._multi_step_decode = Mock()
    runner._prefill = Mock()
    monkeypatch.setattr(spec_module, "prepare_prefill_inputs", Mock())
    monkeypatch.setattr(spec_module, "prepare_decode_inputs", Mock())
    monkeypatch.setattr(
        spec_module,
        "dispatch_cg_and_sync_dp",
        Mock(return_value=(SimpleNamespace(cg_mode=mode, num_tokens=num_tokens), None)),
    )

    tokens = torch.zeros(2, dtype=torch.int64)
    runner.propose(
        batch,
        {},
        {},
        runner.hidden_states,
        None,
        tokens,
        tokens,
        tokens,
        tokens,
        tokens,
        tokens,
        dummy_run=dummy,
        is_profile=profile,
    )

    if mode == CUDAGraphMode.FULL:
        runner.prefill_cudagraph_manager.run_fullgraph.assert_called_once()
        runner._prefill.assert_not_called()
    else:
        assert (
            bool(runner._prefill.call_args.kwargs["select_output_rows"]) is select_rows
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_prepare_exaone_prefill_rows_after_rejection_and_chunking():
    device = torch.device("cuda")
    target = InputBuffers(max_num_reqs=4, max_num_tokens=9, device=device)
    draft = InputBuffers(max_num_reqs=4, max_num_tokens=9, device=device)
    batch = InputBatch.make_dummy(2, 8, target)
    batch.input_ids.copy_(torch.arange(8, device=device))
    batch.positions.copy_(torch.tensor([20, 21, 22, 23, 4, 5, 6, 7], device=device))
    last_token_indices = torch.full((4,), -1, dtype=torch.int64, device=device)

    spec_module.prepare_prefill_inputs(
        last_token_indices,
        torch.tensor(-1, dtype=torch.int64, device=device),
        draft,
        batch,
        num_sampled=torch.tensor([2, 0], dtype=torch.int32, device=device),
        num_rejected=torch.tensor([2, 0], dtype=torch.int32, device=device),
        last_sampled=torch.tensor([80, 81], device=device),
        next_prefill_tokens=torch.tensor([90, 91], dtype=torch.int32, device=device),
        max_num_reqs=4,
    )

    assert last_token_indices.tolist() == [1, 7, 0, 0]
    assert draft.input_ids[last_token_indices[:2]].tolist() == [80, 91]
    assert draft.positions[last_token_indices[:2]].tolist() == [21, 7]
