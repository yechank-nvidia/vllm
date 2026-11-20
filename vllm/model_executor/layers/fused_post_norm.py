# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One-pass kernels for post-norm residual blocks.

Post-norm models (for example EXAONE 4 and OLMo 2) compute
``out = residual + rms_norm(x) * weight`` after attention and after the MLP.
The fused add + RMSNorm kernels in vLLM cover the pre-norm order (add, then
norm), so these blocks are left to Inductor, which emits multi-pass reduction
kernels for them. The kernels below keep one row in registers and read every
input once. They are used by ``PostNormFusionPass``
(vllm/compilation/passes/fusion/rms_quant_fusion.py); models need no change.

The arithmetic follows the code Inductor generates for the unfused graph: all
math in fp32, no rounding between the norm, the weight multiply and the residual
add, and the dynamic per-token FP8 quant computed from the unrounded fp32 sum.
Only the order of the sum of squares differs (one tree reduction over the row).
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.torch_utils import direct_register_custom_op

_FP8_DTYPE = current_platform.fp8_dtype()


@triton.jit
def _post_norm_kernel(
    x_ptr,
    w_ptr,
    r_ptr,
    prev_x_ptr,
    prev_w_ptr,
    out_ptr,
    q_ptr,
    scale_ptr,
    stride_x,
    stride_r,
    stride_prev_x,
    stride_out,
    stride_q,
    eps,
    H: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_PREV: tl.constexpr,
    STORE_OUT: tl.constexpr,
    QUANT: tl.constexpr,
    FP8_MAX: tl.constexpr,
    INV_FP8_MAX: tl.constexpr,
    MIN_SCALE: tl.constexpr,
    LATE_WEIGHT: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.arange(0, BLOCK)
    mask = cols < H

    res = tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_PREV:
        # The residual is itself a post-norm sum: res + rms_norm(prev_x) * prev_w.
        # Recompute it in fp32 instead of reading a rounded copy.
        px = tl.load(prev_x_ptr + row * stride_prev_x + cols, mask=mask, other=0.0)
        px = px.to(tl.float32)
        pw = tl.load(prev_w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        p_rstd = tldevice.rsqrt(tl.sum(px * px, axis=0) / H + eps)
        res = res + (px * p_rstd) * pw

    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if not LATE_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    rstd = tldevice.rsqrt(tl.sum(x * x, axis=0) / H + eps)
    if LATE_WEIGHT:
        w = tl.load(w_ptr + cols, mask=mask, other=0.0).to(tl.float32)
    s = res + (x * rstd) * w

    if STORE_OUT:
        tl.store(
            out_ptr + row * stride_out + cols,
            s.to(out_ptr.dtype.element_ty),
            mask=mask,
        )
    if QUANT:
        # Dynamic per-token symmetric FP8 quant, as QuantFP8.forward_native.
        amax = tl.max(tl.abs(s), axis=0)  # masked lanes hold 0
        scale = tl.maximum(amax * INV_FP8_MAX, MIN_SCALE)
        tl.store(scale_ptr + row, scale)
        q = s * (1.0 / scale)
        q = tl.minimum(tl.maximum(q, -FP8_MAX), FP8_MAX)
        tl.store(
            q_ptr + row * stride_q + cols,
            q.to(q_ptr.dtype.element_ty),
            mask=mask,
        )


def _launch(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
    prev_x: torch.Tensor | None,
    prev_weight: torch.Tensor | None,
    out: torch.Tensor | None,
    q: torch.Tensor | None,
    scale: torch.Tensor | None,
) -> None:
    hidden = x.shape[-1]
    x2 = x.reshape(-1, hidden)
    r2 = residual.reshape(-1, hidden)
    num_rows = x2.shape[0]
    if num_rows == 0:
        return
    assert x2.stride(-1) == 1 and r2.stride(-1) == 1
    p2 = prev_x.reshape(-1, hidden) if prev_x is not None else x2
    assert p2.stride(-1) == 1
    o2 = out.view(-1, hidden) if out is not None else x2
    q2 = q.view(-1, hidden) if q is not None else x2
    block = triton.next_power_of_2(hidden)
    # One CTA per row; 4 warps was fastest for all three variants from 32 to
    # 8192 rows at hidden size 5120 (B200-class GPU).
    num_warps = max(1, min(4, block // 256))
    finfo = torch.finfo(_FP8_DTYPE)
    fp8_max = float(finfo.max)
    _post_norm_kernel[(num_rows,)](
        x2,
        weight,
        r2,
        p2,
        prev_weight if prev_weight is not None else weight,
        o2,
        q2,
        scale if scale is not None else x2,
        x2.stride(0),
        r2.stride(0),
        p2.stride(0),
        o2.stride(0),
        q2.stride(0),
        epsilon,
        H=hidden,
        BLOCK=block,
        HAS_PREV=prev_x is not None,
        STORE_OUT=out is not None,
        QUANT=q is not None,
        FP8_MAX=fp8_max,
        INV_FP8_MAX=1.0 / fp8_max,
        MIN_SCALE=1.0 / (fp8_max * 512.0),
        # Shorten the weight's register lifetime for the wide prefill rows.
        LATE_WEIGHT=hidden == 5120 and num_rows >= 4096 and x.dtype == torch.bfloat16,
        num_warps=num_warps,
    )


def post_norm_add(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
    prev_x: torch.Tensor | None = None,
    prev_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute residual + rms_norm(x) * weight in one pass.

    With prev_x / prev_weight, the residual is first replaced by
    residual + rms_norm(prev_x) * prev_weight (kept in fp32): the second add of
    a post-norm block whose residual is the first post-norm sum.
    """
    out = torch.empty_like(x)
    _launch(x, weight, residual, epsilon, prev_x, prev_weight, out, None, None)
    return out


def post_norm_add_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
    prev_x: torch.Tensor | None = None,
    prev_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return torch.empty_like(x)


def post_norm_add_fp8_quant(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dynamic per-token FP8 quant of residual + rms_norm(x) * weight.

    Returns (fp8 values, fp32 scales of shape [..., 1]). The sum is not stored:
    the pattern recomputes it where it is needed.
    """
    q = torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device)
    scale = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    _launch(x, weight, residual, epsilon, None, None, None, q, scale)
    return q, scale


def post_norm_add_fp8_quant_fake(
    x: torch.Tensor,
    weight: torch.Tensor,
    residual: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q = torch.empty(x.shape, dtype=_FP8_DTYPE, device=x.device)
    scale = torch.empty((*x.shape[:-1], 1), dtype=torch.float32, device=x.device)
    return q, scale


direct_register_custom_op(
    op_name="post_norm_add",
    op_func=post_norm_add,
    fake_impl=post_norm_add_fake,
)
direct_register_custom_op(
    op_name="post_norm_add_fp8_quant",
    op_func=post_norm_add_fp8_quant,
    fake_impl=post_norm_add_fp8_quant_fake,
)
