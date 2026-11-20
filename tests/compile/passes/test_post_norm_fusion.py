# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Post-norm fusion: residual + rms_norm(x) * w (+ dynamic per-token fp8 quant)."""

import pytest
import torch

import vllm.config
import vllm.ir.ops
from tests.compile.backend import TestBackend
from vllm.compilation.passes.fusion.rms_quant_fusion import PostNormFusionPass
from vllm.compilation.passes.ir.lowering_pass import VllmIRLoweringPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    ModelConfig,
    PassConfig,
    VllmConfig,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape
from vllm.platforms import current_platform

FP8_DTYPE = current_platform.fp8_dtype()
POST_NORM_ADD_OP = torch.ops.vllm.post_norm_add.default
POST_NORM_ADD_FP8_QUANT_OP = torch.ops.vllm.post_norm_add_fp8_quant.default
RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="post-norm fusion runs on CUDA only"
)


def _native_rms_norm(x, w, eps):
    return vllm.ir.ops.rms_norm.impls["native"].impl_fn(x, w, eps)


_QUANT = None


def _ref_add_quant(x, w, r, eps):
    s = r + _native_rms_norm(x, w, eps)
    return _QUANT.forward_native(s)


def _ref_chained_add(x, w, px, pw, r, eps):
    s = r + _native_rms_norm(px, pw, eps)
    return s + _native_rms_norm(x, w, eps)


def _ref_add(x, w, r, eps):
    return r + _native_rms_norm(x, w, eps)


def _inputs(num_tokens, hidden, exact_sum, dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)

    def rnd(*shape, scale=1.0):
        return torch.randn(*shape, generator=g, device="cuda") * scale

    if exact_sum:
        # Values k/8 with |k| <= 16: every partial sum of squares is exact in
        # fp32, so the result does not depend on the reduction order.
        def norm_in():
            k = torch.randint(-16, 17, (num_tokens, hidden), generator=g, device="cuda")
            return (k.float() / 8).to(dtype)
    else:

        def norm_in():
            return rnd(num_tokens, hidden, scale=3.0).to(dtype)

    x, px = norm_in(), norm_in()
    w = (torch.rand(hidden, generator=g, device="cuda") + 0.5).to(dtype)
    pw = (torch.rand(hidden, generator=g, device="cuda") + 0.5).to(dtype)
    r = rnd(num_tokens, hidden, scale=4.0).to(dtype)
    return x, w, px, pw, r


@pytest.mark.parametrize("num_tokens", [1, 7, 128, 400, 4096, 8192])
@pytest.mark.parametrize("hidden", [5120, 4096, 1000])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_post_norm_kernels_match_inductor(num_tokens, hidden, eps):
    """Bitwise equal to Inductor's code for the unfused graph when the sum of
    squares does not depend on the reduction order; close otherwise."""
    global _QUANT
    with vllm.config.set_current_vllm_config(VllmConfig()):
        _QUANT = QuantFP8(False, GroupShape.PER_TOKEN, compile_native=False)
    torch._dynamo.reset()
    ref_q = torch.compile(_ref_add_quant, dynamic=False, fullgraph=True)
    ref_c = torch.compile(_ref_chained_add, dynamic=False, fullgraph=True)
    ref_a = torch.compile(_ref_add, dynamic=False, fullgraph=True)

    # Exact sums of squares: everything must be bitwise equal.
    x, w, px, pw, r = _inputs(num_tokens, hidden, exact_sum=True)
    q, scale = POST_NORM_ADD_FP8_QUANT_OP(x, w, r, eps)
    q_ref, scale_ref = ref_q(x, w, r, eps)
    assert torch.equal(scale, scale_ref)
    assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
    out = POST_NORM_ADD_OP(x, w, r, eps, px, pw)
    assert torch.equal(out, ref_c(x, w, px, pw, r, eps))
    out = POST_NORM_ADD_OP(x, w, r, eps)
    assert torch.equal(out, ref_a(x, w, r, eps))

    # General inputs: only the summation order differs, so the sum of squares
    # may differ in its last bit.
    x, w, px, pw, r = _inputs(num_tokens, hidden, exact_sum=False, seed=1)
    q, scale = POST_NORM_ADD_FP8_QUANT_OP(x, w, r, eps)
    q_ref, scale_ref = ref_q(x, w, r, eps)
    torch.testing.assert_close(scale, scale_ref, rtol=1e-6, atol=0)
    # at most one fp8 step apart, and rarely
    torch.testing.assert_close(q.float(), q_ref.float(), rtol=0.125, atol=2**-9)
    assert (q.view(torch.uint8) != q_ref.view(torch.uint8)).float().mean() < 1e-3
    out = POST_NORM_ADD_OP(x, w, r, eps, px, pw)
    ref = ref_c(x, w, px, pw, r, eps)
    # one bf16 step apart (more only where the sum cancels to ~0), and rarely
    torch.testing.assert_close(out, ref, rtol=2**-6, atol=1e-5)
    assert (out != ref).float().mean() < 1e-3


class PostNormBlock(torch.nn.Module):
    """Two post-norm layers: h = r + norm(o(h)); h = h + norm(mlp(quant(h)))."""

    def __init__(self, hidden, eps, quant: bool, extra_user: bool = False):
        super().__init__()
        self.layers = torch.nn.ModuleList()
        self.quant = quant
        self.extra_user = extra_user
        for _ in range(2):
            layer = torch.nn.Module()
            layer.o = torch.nn.Linear(hidden, hidden, bias=False)
            layer.up = torch.nn.Linear(hidden, hidden, bias=False)
            layer.norm1 = RMSNorm(hidden, eps)
            layer.norm2 = RMSNorm(hidden, eps)
            layer.norm1.weight.data.uniform_(0.5, 1.5)
            layer.norm2.weight.data.uniform_(0.5, 1.5)
            self.layers.append(layer)
        self.quant_fp8 = QuantFP8(False, GroupShape.PER_TOKEN)

    def forward(self, h):
        extra = []
        for layer in self.layers:
            r = h
            y = layer.norm1(layer.o(h))
            if self.extra_user:
                extra.append(y.sum())  # norm output has a second user
            h = r + y
            r = h
            if self.quant:
                q, scale = self.quant_fp8(h)
                m = layer.up(q.to(h.dtype) * scale.to(h.dtype))
            else:
                m = layer.up(h)
            h = r + layer.norm2(m)
        return (h, *extra)


@pytest.fixture
def vllm_config():
    """VLLM_COMPILE config with bf16 CUDA defaults and a fixed seed."""
    config = VllmConfig(
        model_config=ModelConfig(dtype=torch.bfloat16),
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            pass_config=PassConfig(eliminate_noops=True),
        ),
    )
    with (
        vllm.config.set_current_vllm_config(config),
        config.kernel_config.ir_op_priority.set_priority(),
    ):
        torch.set_default_device("cuda")
        torch.set_default_dtype(torch.bfloat16)
        torch.manual_seed(0)
        try:
            yield config
        finally:
            torch.set_default_dtype(torch.float32)


def _compile(model, passes, x):
    backend = TestBackend(*passes)
    out = torch.compile(model, backend=backend)(x)
    return backend, out


@pytest.mark.parametrize("quant", [True, False])
@pytest.mark.parametrize("eps", [1e-5, 1e-6])
def test_post_norm_fusion_pass(quant, eps, vllm_config):
    hidden, num_tokens = 256, 257
    model = PostNormBlock(hidden, eps, quant)
    x = torch.randn(num_tokens, hidden)
    fusion = PostNormFusionPass(vllm_config)
    cleanup = PostCleanupPass(vllm_config)
    # Lower the IR ops like the pass manager does, so the reference is
    # Inductor's code for the unfused graph.
    lowering = VllmIRLoweringPass(vllm_config)
    torch._dynamo.reset()
    backend, out = _compile(model, [fusion, cleanup, lowering, cleanup], x)
    torch._dynamo.reset()
    _, ref = _compile(model, [cleanup, lowering, cleanup], x)
    # Only the order of the sum of squares differs.
    torch.testing.assert_close(out[0], ref[0], rtol=2e-2, atol=2e-2)
    assert (out[0] == ref[0]).float().mean() > 0.9

    # Every norm is replaced: 2 layers x (post-attn + post-mlp).
    assert backend.op_count(RMS_NORM_OP) == 0
    if quant:
        # post-attn: add+quant, post-mlp: chained add (recomputes the sum)
        assert fusion.matched_count == 4
        assert backend.op_count(POST_NORM_ADD_FP8_QUANT_OP) == 2
        assert backend.op_count(POST_NORM_ADD_OP) == 2
    else:
        # The first sum also feeds the next linear, so it is stored and
        # both adds are plain post-norm adds.
        assert fusion.matched_count == 4
        assert backend.op_count(POST_NORM_ADD_OP) == 4


def test_post_norm_fusion_skips_other_users_and_pre_norm(vllm_config):
    hidden, num_tokens, eps = 256, 33, 1e-5
    x = torch.randn(num_tokens, hidden)
    cleanup = PostCleanupPass(vllm_config)

    # Post-attn norm output also used elsewhere: only the post-mlp norms fuse.
    model = PostNormBlock(hidden, eps, quant=True, extra_user=True)
    fusion = PostNormFusionPass(vllm_config)
    torch._dynamo.reset()
    backend, _ = _compile(model, [fusion, cleanup], x)
    assert backend.op_count(POST_NORM_ADD_FP8_QUANT_OP) == 0
    assert backend.op_count(RMS_NORM_OP) == 2

    # Pre-norm block (fused add + norm, then linear) never matches.
    class PreNorm(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = RMSNorm(hidden, eps)
            self.lin = torch.nn.Linear(hidden, hidden, bias=False)
            self.norm2 = RMSNorm(hidden, eps)

        def forward(self, h):
            y = self.norm(h)
            h2, r = self.norm2(self.lin(y), h)
            return self.lin(h2) + r

    fusion = PostNormFusionPass(vllm_config)
    torch._dynamo.reset()
    backend, _ = _compile(PreNorm(), [fusion, cleanup], x)
    assert fusion.matched_count == 0


def _compile_xr(model, passes, x, r):
    backend = TestBackend(*passes)
    out = torch.compile(model, backend=backend)(x, r)
    return backend, out


def _assert_kernel_args_present(backend):
    # A post-norm op never gets a missing weight (weight=None).
    for node in backend.graph_post_pass.nodes:
        if node.target in (POST_NORM_ADD_OP, POST_NORM_ADD_FP8_QUANT_OP):
            assert all(arg is not None for arg in node.args)


class UnsupportedPostNorm(torch.nn.Module):
    """Post-norm adds the kernels cannot compute; they must stay unfused."""

    def __init__(self, hidden, eps, case):
        super().__init__()
        self.case = case
        self.lin = torch.nn.Linear(hidden, hidden, bias=False)
        self.norm = RMSNorm(hidden, eps, has_weight=case != "weightless")

    def forward(self, x, r):
        y = self.norm(self.lin(x))
        if self.case == "residual_broadcast":
            # Gemma3n AltUp correction: preds[1:] += norm(x), x is [T, H]
            preds = r.clone()
            preds[1:] += y
            return preds
        # norm_broadcast: x is [A, T, H], r is [T, H]
        # promote: r is fp32, so the sum is fp32
        # weightless: RMSNorm without weight (HRM-Text), weight=None
        return r + y


@pytest.mark.parametrize(
    "case", ["residual_broadcast", "norm_broadcast", "promote", "weightless"]
)
def test_post_norm_fusion_skips_unsupported(case, vllm_config):
    hidden, num_tokens, eps = 256, 33, 1e-6
    x = torch.randn(num_tokens, hidden)
    r = torch.randn(num_tokens, hidden)
    if case == "residual_broadcast":
        r = torch.randn(4, num_tokens, hidden)
    elif case == "norm_broadcast":
        x = torch.randn(3, num_tokens, hidden)
    elif case == "promote":
        r = r.float()
    model = UnsupportedPostNorm(hidden, eps, case)
    if case != "weightless":
        model.norm.weight.data.uniform_(0.5, 1.5)
    fusion = PostNormFusionPass(vllm_config)
    cleanup = PostCleanupPass(vllm_config)
    lowering = VllmIRLoweringPass(vllm_config)
    torch._dynamo.reset()
    backend, out = _compile_xr(model, [fusion, cleanup, lowering, cleanup], x, r)
    torch._dynamo.reset()
    _, ref = _compile_xr(model, [cleanup, lowering, cleanup], x, r)

    assert fusion.matched_count == 0
    assert backend.op_count(POST_NORM_ADD_OP) == 0
    assert backend.op_count(POST_NORM_ADD_FP8_QUANT_OP) == 0
    assert out.dtype == ref.dtype and out.shape == ref.shape
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("weightless", ["post_attn", "post_mlp"])
def test_post_norm_fusion_weightless_norm_in_block(weightless, vllm_config):
    """Weightless norms stay unfused; the weighted ones in the block fuse."""
    hidden, num_tokens, eps = 256, 33, 1e-5
    x = torch.randn(num_tokens, hidden)
    model = PostNormBlock(hidden, eps, quant=True)
    # One layer: the unfused sum is stored in bf16 when the chained add
    # cannot match, and a second layer's fp8 quant would amplify that.
    del model.layers[1]
    name = "norm1" if weightless == "post_attn" else "norm2"
    setattr(model.layers[0], name, RMSNorm(hidden, eps, has_weight=False))
    fusion = PostNormFusionPass(vllm_config)
    cleanup = PostCleanupPass(vllm_config)
    lowering = VllmIRLoweringPass(vllm_config)
    torch._dynamo.reset()
    backend, out = _compile(model, [fusion, cleanup, lowering, cleanup], x)
    torch._dynamo.reset()
    _, ref = _compile(model, [cleanup, lowering, cleanup], x)
    torch.testing.assert_close(out[0], ref[0], rtol=2e-2, atol=2e-2)

    _assert_kernel_args_present(backend)
    if weightless == "post_attn":
        # The post-mlp add reads the stored first sum: plain post-norm add.
        assert fusion.matched_count == 1
        assert backend.op_count(POST_NORM_ADD_FP8_QUANT_OP) == 0
        assert backend.op_count(POST_NORM_ADD_OP) == 1
    else:
        # post-attn add+quant fuses; the chained post-mlp add does not, so
        # the rebuilt first sum is a plain post-norm add.
        assert fusion.matched_count == 2
        assert backend.op_count(POST_NORM_ADD_FP8_QUANT_OP) == 1
        assert backend.op_count(POST_NORM_ADD_OP) == 1
