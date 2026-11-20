# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""fused_qk_norm_rope_kv_insert vs the unfused ops it replaces:
per-head RMSNorm (vllm_ir rms_norm math), RoPE (RotaryEmbedding.forward_static
math) and reshape_and_cache_flash."""

import pytest
import torch

from vllm import _custom_ops as ops
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda() or not ops.has_fused_qk_norm_rope_kv_insert(),
    reason="needs CUDA and the fused_qk_norm_rope_kv_insert op",
)


def _ref_q_k(qkv, positions, q_w, k_w, cos_sin_cache, eps, nq, nkv, hd, dtype64=False):
    """Unfused math in fp32 (or fp64) with no rounding between norm and RoPE,
    which is what torch.compile generates for the unfused ops."""
    ct = torch.float64 if dtype64 else torch.float32
    n = qkv.shape[0]
    q, k, _ = qkv.split([nq * hd, nkv * hd, nkv * hd], dim=-1)

    def norm(x, w, heads):
        x = x.reshape(n, heads, hd).to(ct)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + eps) * w.to(ct)

    qn, kn = norm(q, q_w, nq), norm(k, k_w, nkv)
    if cos_sin_cache is not None:
        cos, sin = cos_sin_cache.index_select(0, positions).to(ct).chunk(2, -1)
        cos, sin = cos.unsqueeze(-2), sin.unsqueeze(-2)

        def rope(x):  # NeoX over the full head
            x1, x2 = x.chunk(2, dim=-1)
            return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), -1)

        qn, kn = rope(qn), rope(kn)
    return qn.reshape(n, nq * hd), kn.reshape(n, nkv * hd)


def _make_cache(num_blocks, block_size, nkv, hd, dtype, layout):
    if layout == "NHD":
        return torch.randn(
            num_blocks, 2, block_size, nkv, hd, dtype=dtype, device="cuda"
        )
    # HND: physical [nb, 2, nkv, bs, hd], logical [nb, 2, bs, nkv, hd]
    c = torch.randn(num_blocks, 2, nkv, block_size, hd, dtype=dtype, device="cuda")
    return c.permute(0, 1, 3, 2, 4)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("nq,nkv,hd", [(40, 8, 128), (8, 1, 64), (16, 4, 256)])
@pytest.mark.parametrize("rope", ["none", "neox"])
@pytest.mark.parametrize("layout", ["NHD", "HND"])
@pytest.mark.parametrize("num_tokens,num_slots", [(1, 1), (37, 37), (130, 128)])
@torch.inference_mode()
def test_fused_qk_norm_rope_kv_insert(
    default_vllm_config, dtype, nq, nkv, hd, rope, layout, num_tokens, num_slots
):
    torch.manual_seed(0)
    eps = 1e-5
    block_size, num_blocks, max_pos = 16, 64, 4096
    width = (nq + 2 * nkv) * hd
    # Rows wider than the q|k|v width: the op must honour the row stride.
    base = torch.randn(num_tokens, width + 64, dtype=dtype, device="cuda") * 3
    qkv = base[:, :width]
    q_w = (torch.rand(hd, device="cuda") * 2).to(dtype)
    k_w = (torch.rand(hd, device="cuda") * 2).to(dtype)
    positions = torch.randint(0, max_pos, (num_tokens,), device="cuda")

    cos_sin_cache = None
    if rope == "neox":
        rotary_emb = get_rope(
            hd,
            max_position=max_pos,
            is_neox_style=True,
            rope_parameters={"rope_theta": 1e6},
            dtype=dtype,
        ).cuda()
        cos_sin_cache = rotary_emb.cos_sin_cache.to(dtype)
        assert cos_sin_cache.shape[1] == hd

    kv_cache = _make_cache(num_blocks, block_size, nkv, hd, dtype, layout)
    perm = torch.randperm(num_blocks * block_size, device="cuda")
    slot_mapping = perm[:num_slots].to(torch.int64)
    if num_slots > 2:
        slot_mapping[1] = -1  # padded / unscheduled token
    kv_ref = kv_cache.clone()

    q_out = torch.empty(num_tokens, nq * hd, dtype=dtype, device="cuda")
    k_out = torch.empty(num_tokens, nkv * hd, dtype=dtype, device="cuda")
    qkv_before = qkv.clone()
    ops.fused_qk_norm_rope_kv_insert(
        qkv,
        q_out,
        k_out,
        q_w,
        k_w,
        cos_sin_cache,
        positions,
        nq,
        nkv,
        eps,
        slot_mapping,
        kv_cache,
    )
    torch.accelerator.synchronize()
    assert torch.equal(qkv, qkv_before), "qkv must not be modified"

    q_ref, k_ref = _ref_q_k(qkv, positions, q_w, k_w, cos_sin_cache, eps, nq, nkv, hd)
    tol = (
        dict(atol=2e-2, rtol=1e-2)
        if dtype == torch.bfloat16
        else dict(atol=2e-3, rtol=2e-3)
    )
    torch.testing.assert_close(q_out.float(), q_ref, **tol)
    torch.testing.assert_close(k_out.float(), k_ref, **tol)

    # The cache must hold exactly k_out (rounded once) and v, as
    # reshape_and_cache_flash would write them, and nothing else.
    v = qkv[:, (nq + nkv) * hd :]
    ops.reshape_and_cache_flash(
        k_out.view(num_tokens, nkv, hd),
        v.reshape(num_tokens, nkv, hd),
        kv_ref[:, 0],
        kv_ref[:, 1],
        slot_mapping,
        "auto",
        torch.ones(1, device="cuda"),
        torch.ones(1, device="cuda"),
    )
    assert torch.equal(kv_cache, kv_ref)


@torch.inference_mode()
def test_no_cache_and_empty():
    nq, nkv, hd, dtype = 40, 8, 128, torch.bfloat16
    qkv = torch.randn(5, (nq + 2 * nkv) * hd, dtype=dtype, device="cuda")
    w = torch.ones(hd, dtype=dtype, device="cuda")
    pos = torch.arange(5, device="cuda")
    q_out = torch.empty(5, nq * hd, dtype=dtype, device="cuda")
    k_out = torch.empty(5, nkv * hd, dtype=dtype, device="cuda")
    ops.fused_qk_norm_rope_kv_insert(qkv, q_out, k_out, w, w, None, pos, nq, nkv, 1e-5)
    q_ref, k_ref = _ref_q_k(qkv, pos, w, w, None, 1e-5, nq, nkv, hd)
    torch.testing.assert_close(q_out.float(), q_ref, atol=2e-2, rtol=1e-2)
    torch.testing.assert_close(k_out.float(), k_ref, atol=2e-2, rtol=1e-2)
    # zero tokens: no launch, no error
    e = qkv[:0]
    ops.fused_qk_norm_rope_kv_insert(
        e, q_out[:0], k_out[:0], w, w, None, pos[:0], nq, nkv, 1e-5
    )
