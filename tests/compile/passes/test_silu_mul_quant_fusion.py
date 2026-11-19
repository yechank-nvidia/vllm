# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import itertools
from functools import partial

import pytest
import torch

import vllm.envs as envs
from tests.compile.backend import TestBackend
from tests.kernels.quantization.nvfp4_utils import quant_nvfp4_tensor
from tests.utils import TestFP8Layer
from vllm._aiter_ops import IS_AITER_FOUND, rocm_aiter_ops
from vllm._custom_ops import cutlass_scaled_fp4_mm, scaled_fp4_quant
from vllm.compilation.passes.fusion.act_quant_fusion import (
    FUSED_OPS,
    SILU_MUL_OP,
    ActivationQuantFusionPass,
)
from vllm.compilation.passes.fusion.rms_quant_fusion import QUANT_OPS
from vllm.compilation.passes.utility.noop_elimination import NoOpEliminationPass
from vllm.compilation.passes.utility.post_cleanup import PostCleanupPass
from vllm.config import (
    CompilationConfig,
    CompilationMode,
    PassConfig,
    VllmConfig,
    set_current_vllm_config,
)
from vllm.model_executor.kernels.linear import (
    CutlassFP8ScaledMMLinearKernel,
    FlashInferFP8ScaledMMLinearKernel,
    FP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
    ROCmFP8ScaledMMLinearKernel,
)
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.quantization.input_quant_fp8 import QuantFP8
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    create_fp8_quant_key,
    kFp8Dynamic128Sym,
    kFp8DynamicTokenSym,
    kFp8StaticChannelSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
)
from vllm.platforms import current_platform
from vllm.utils.deep_gemm import is_deep_gemm_supported

FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


def is_nvfp4_supported():
    return current_platform.has_device_capability(100)


class TestSiluMulFp8QuantModel(torch.nn.Module):
    quant_key = kFp8StaticTensorSym

    def __init__(
        self,
        hidden_size: int,
        force_kernel: FP8ScaledMMLinearKernel,
        dtype: torch.dtype,
        **kwargs,
    ):
        super().__init__()
        self.silu_and_mul = SiluAndMul()

        self.fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=self.quant_key,
            weight_quant_key=self.quant_key,
            force_kernel=force_kernel,
            input_dtype=dtype,
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.fp8_linear.is_quant_fp8_enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.fp8_linear(y)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            (
                QUANT_OPS[kFp8StaticTensorSym]
                if self.enable_quant_fp8_custom_op
                else torch.ops.aten.reciprocal
            ),
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kFp8StaticTensorSym]]


class TestSiluMulPerTokenQuantModel(torch.nn.Module):
    """SiLU-and-mul feeding an FP8 linear with dynamic per-token activations
    and per-channel weights (e.g. compressed-tensors W8A8 FP8 channel/token)."""

    def __init__(self, hidden_size: int, dtype: torch.dtype, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            activation_quant_key=kFp8DynamicTokenSym,
            weight_quant_key=kFp8StaticChannelSym,
            force_kernel=CutlassFP8ScaledMMLinearKernel,
            input_dtype=dtype,
        )
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.fp8_linear.is_quant_fp8_enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        return self.fp8_linear(y)

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            (
                QUANT_OPS[kFp8DynamicTokenSym]
                if self.enable_quant_fp8_custom_op
                else torch.ops.aten.reciprocal
            ),
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kFp8DynamicTokenSym]]


class TestSiluMulNvfp4QuantModel(torch.nn.Module):
    def __init__(self, hidden_size: int, x: torch.Tensor, **kwargs):
        super().__init__()
        from vllm.compilation.passes.fusion.act_quant_fusion import (
            silu_and_mul_nvfp4_quant_supported,
        )

        assert silu_and_mul_nvfp4_quant_supported

        self.silu_and_mul = SiluAndMul()
        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

        # create nvfp4 weight
        w = torch.rand((hidden_size, hidden_size))
        self.w, self.w_block_scale, self.w_global_scale = quant_nvfp4_tensor(w)

        # get global scale offline
        _, _, self.y_global_scale = quant_nvfp4_tensor(self.silu_and_mul(x))

        self.alpha = 1.0 / (self.w_global_scale * self.y_global_scale)

    def forward(self, x):
        y = self.silu_and_mul(x)
        y_quant, y_block_scale = scaled_fp4_quant(y, self.y_global_scale)
        out = cutlass_scaled_fp4_mm(
            a=y_quant,
            b=self.w,
            block_scale_a=y_block_scale,
            block_scale_b=self.w_block_scale,
            alpha=self.alpha,
            out_dtype=y.dtype,
        )
        return out

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            QUANT_OPS[kNvfp4Dynamic],
        ]

    def ops_in_model_after(self):
        return [FUSED_OPS[kNvfp4Dynamic]]


class TestSiluMulGroupFp8QuantModel(torch.nn.Module):
    act_quant_key = kFp8Dynamic128Sym

    def __init__(self, hidden_size: int, dtype: torch.dtype, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.weight_quant_key = create_fp8_quant_key(
            static=True, group_shape=GroupShape(hidden_size, hidden_size)
        )

        self.w8a8_block_fp8_linear = TestFP8Layer(
            weight_shape=(hidden_size, hidden_size),
            weight_quant_key=self.weight_quant_key,
            activation_quant_key=self.act_quant_key,
            input_dtype=dtype,
        )

        if not current_platform.is_fp8_fnuz():
            kernel = self.w8a8_block_fp8_linear.kernel
            orig_quant = kernel.quant_fp8
            kernel.quant_fp8 = lambda *a, use_triton=False, **kw: orig_quant(
                *a, use_triton=True, **kw
            )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        x2 = self.w8a8_block_fp8_linear(y)
        return x2

    def ops_in_model_before(self):
        return [
            SILU_MUL_OP if self.enable_silu_mul_custom_op else torch.ops.aten.mul,
            rocm_aiter_ops.get_group_quant_op()
            if current_platform.is_fp8_fnuz()
            else torch.ops.vllm.triton_per_token_group_quant_fp8.default,
        ]

    def ops_in_model_after(self):
        return [torch.ops.vllm.rocm_aiter_act_mul_and_fp8_group_quant]


class TestSiluMulBlockQuantModel(torch.nn.Module):
    quant_key = kFp8Dynamic128Sym

    def __init__(self, hidden_size: int, is_scale_transposed: bool = False, **kwargs):
        super().__init__()
        self.silu_and_mul = SiluAndMul()
        self.is_scale_transposed = is_scale_transposed
        self.quant_fp8 = QuantFP8(
            static=False,
            group_shape=GroupShape(1, 128),
            column_major_scales=is_scale_transposed,
            compile_native=False,
        )

        self.enable_silu_mul_custom_op = self.silu_and_mul.enabled()
        self.enable_quant_fp8_custom_op = self.quant_fp8.enabled()

    def forward(self, x):
        y = self.silu_and_mul(x)
        out, scale = self.quant_fp8(y)
        group_size = self.quant_key.scale.group_shape[1]
        scale_expanded = scale.repeat_interleave(group_size, dim=1)
        dequant = out.to(dtype=torch.float32) * scale_expanded
        return (dequant,)

    def ops_in_model_before(self):
        ops = []
        if self.enable_silu_mul_custom_op:
            ops.append(SILU_MUL_OP)
        # When silu custom op is disabled, aten.mul.Tensor also appears
        # in dequant code, so we skip checking it to avoid false positives.
        ops.append(
            QUANT_OPS[self.quant_key]
            if self.enable_quant_fp8_custom_op
            else torch.ops.aten.reciprocal.default
        )
        return ops

    def ops_in_model_after(self):
        return [FUSED_OPS[self.quant_key]]


ROCM_KERNELS = [ROCmFP8ScaledMMLinearKernel, PerTensorTorchFP8ScaledMMLinearKernel]
CUDA_KERNELS = [
    FlashInferFP8ScaledMMLinearKernel,
    CutlassFP8ScaledMMLinearKernel,
    PerTensorTorchFP8ScaledMMLinearKernel,
]
TEST_KERNELS = ROCM_KERNELS if current_platform.is_rocm() else CUDA_KERNELS


@pytest.mark.parametrize("num_tokens", [32, 64])
@pytest.mark.parametrize("hidden_size", [128, 256])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
@pytest.mark.parametrize(
    "model_class, enable_quant_fp8_custom_op, force_kernel",
    list(itertools.product([TestSiluMulFp8QuantModel], [True, False], TEST_KERNELS))
    + [
        pytest.param(
            TestSiluMulNvfp4QuantModel,
            False,
            None,
            marks=pytest.mark.skipif(
                not current_platform.is_cuda(), reason="CUDA only"
            ),
        ),
        # GroupFP8Quant fusion only works with AITER on ROCm.
        # and the enable_quant_fp8_custom_op must be True.
        pytest.param(
            TestSiluMulGroupFp8QuantModel,
            True,
            None,
            marks=pytest.mark.skipif(
                not current_platform.is_rocm(), reason="ROCm only"
            ),
        ),
        # Block quant fusion for per-group FP8 (CUDA only).
        *[
            pytest.param(
                partial(TestSiluMulBlockQuantModel, is_scale_transposed=transposed),
                True,
                None,
                marks=pytest.mark.skipif(
                    not current_platform.is_cuda(), reason="CUDA only"
                ),
                id=f"TestSiluMulBlockQuant-transposed={transposed}",
            )
            for transposed in [False, True]
        ],
    ],
)
@pytest.mark.skipif(
    envs.VLLM_TARGET_DEVICE not in ["cuda", "rocm"], reason="Only test on CUDA and ROCm"
)
def test_fusion_silu_and_mul_quant(
    num_tokens: int,
    hidden_size: int,
    dtype: torch.dtype,
    model_class: type[
        TestSiluMulFp8QuantModel
        | TestSiluMulNvfp4QuantModel
        | TestSiluMulGroupFp8QuantModel
        | TestSiluMulBlockQuantModel
    ],
    enable_silu_mul_custom_op: bool,
    enable_quant_fp8_custom_op: bool,
    force_kernel: FP8ScaledMMLinearKernel | None,
    monkeypatch: pytest.MonkeyPatch,
):
    if model_class is TestSiluMulNvfp4QuantModel and not is_nvfp4_supported():
        pytest.skip("NVFP4 is not supported on this GPU.")
    if model_class is TestSiluMulGroupFp8QuantModel and not IS_AITER_FOUND:
        pytest.skip("AITER is not supported on this GPU.")
    if (
        isinstance(model_class, partial)
        and model_class.func is TestSiluMulBlockQuantModel
        and is_deep_gemm_supported()
    ):
        pytest.skip("SiluMul+BlockQuant fusion not applicable with DeepGemm")

    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)

    x = torch.rand(num_tokens, hidden_size * 2)

    # Reshape pass is needed for the fusion pass to work
    custom_ops = ["none"]
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",  # avoid compilation for SiluAndMul and QuantFP8
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )

    with set_current_vllm_config(config), monkeypatch.context() as m:
        fusion_passes = [ActivationQuantFusionPass(config)]
        if IS_AITER_FOUND and model_class is TestSiluMulGroupFp8QuantModel:
            from vllm.compilation.passes.fusion.rocm_aiter_fusion import (
                RocmAiterSiluMulFp8GroupQuantFusionPass,
            )

            m.setenv("VLLM_ROCM_USE_AITER", "1")
            rocm_aiter_ops.refresh_env_variables()
            fusion_passes += [RocmAiterSiluMulFp8GroupQuantFusionPass(config)]

        passes = [NoOpEliminationPass(config), *fusion_passes, PostCleanupPass(config)]
        backend = TestBackend(*passes)
        model = model_class(
            hidden_size=hidden_size, force_kernel=force_kernel, x=x, dtype=dtype
        )

        # First dimension dynamic
        torch._dynamo.mark_dynamic(x, 0)

        result = model(x)

        model2 = torch.compile(model, backend=backend)
        result2 = model2(x)

        # Check that it gives the same answer
        if isinstance(model, TestSiluMulFp8QuantModel):
            atol, rtol = 1e-3, 1e-3
        elif isinstance(model, TestSiluMulNvfp4QuantModel):
            atol, rtol = 1e-1, 1e-1
        elif isinstance(model, TestSiluMulGroupFp8QuantModel):
            atol, rtol = 5e-2, 5e-2
        elif isinstance(model, TestSiluMulBlockQuantModel):
            if current_platform.is_rocm():
                atol, rtol = 1e-3, 1e-3
            else:
                # CUDA fused kernel computes silu*mul in fp32 while the reference
                # goes through bf16/fp16 storage, so group maxima (and thus scales)
                # can shift by one FP8-e4m3 code (~1/8 relative step).
                atol, rtol = 5e-2, 5e-2

        torch.testing.assert_close(
            result[0].to(dtype=dtype), result2[0].to(dtype=dtype), atol=atol, rtol=rtol
        )

        assert sum([p.matched_count for p in fusion_passes]) == 1

        # In pre-nodes, quant op should be present and fused kernels should not
        backend.check_before_ops(model.ops_in_model_before())

        # In post-nodes, fused kernels should be present and quant op should not
        backend.check_after_ops(model.ops_in_model_after())


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("enable_quant_fp8_custom_op", [True, False])
@pytest.mark.parametrize("enable_silu_mul_custom_op", [True, False])
def test_fusion_silu_mul_per_token_quant(
    enable_silu_mul_custom_op: bool,
    enable_quant_fp8_custom_op: bool,
    dtype: torch.dtype,
):
    """Match only native ops and check the downstream FP8 linear output."""
    torch.set_default_device("cuda")
    torch.set_default_dtype(dtype)
    custom_ops = ["none"]
    if enable_silu_mul_custom_op:
        custom_ops.append("+silu_and_mul")
    if enable_quant_fp8_custom_op:
        custom_ops.append("+quant_fp8")
    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=custom_ops,
            backend="eager",
            pass_config=PassConfig(fuse_act_quant=True, eliminate_noops=True),
        ),
    )
    native = not (enable_silu_mul_custom_op or enable_quant_fp8_custom_op)

    with set_current_vllm_config(config):
        model = TestSiluMulPerTokenQuantModel(hidden_size=256, dtype=dtype)
        x = torch.randn(64, 512) * 4
        torch._dynamo.mark_dynamic(x, 0)
        outs = []
        for fuse in (False, True):
            passes = [NoOpEliminationPass(config)]
            if fuse:
                fusion_pass = ActivationQuantFusionPass(config)
                passes.append(fusion_pass)
            backend = TestBackend(*passes, PostCleanupPass(config))
            torch._dynamo.reset()
            outs.append(torch.compile(model, backend=backend)(x))

        torch.testing.assert_close(outs[1], outs[0], atol=5e-2, rtol=5e-2)
        assert fusion_pass.matched_count == int(native)
        if native:
            backend.check_before_ops(model.ops_in_model_before())
            backend.check_after_ops(model.ops_in_model_after())
        else:
            assert backend.op_count(FUSED_OPS[kFp8DynamicTokenSym]) == 0


def _native_silu_mul_per_token_quant(x: torch.Tensor):
    quant = QuantFP8(
        static=False, group_shape=GroupShape.PER_TOKEN, compile_native=False
    )
    return quant.forward_native(SiluAndMul.forward_native(x))


def _special_rows(x: torch.Tensor) -> torch.Tensor:
    """Rows of -0.0, NaN, +-inf and fp16-range values next to normal rows."""
    x = x.clone()
    d = x.shape[-1] // 2
    gate, up = x[:, :d], x[:, d:]
    kind = torch.arange(x.shape[0], device=x.device) % 6
    gate[kind == 0] = -1000.0  # SiLU is -0.0: a row of +-0.0
    gate[kind == 1, 0] = float("nan")
    up[kind == 2, d // 2] = float("inf")
    gate[kind == 3] = -24.0  # SiLU underflows in fp16, not in fp32
    up[kind == 3] = 65504.0
    gate[kind == 4, -1] = float("-inf")  # SiLU(-inf) is NaN
    return x


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("dynamic", [True, False])
@pytest.mark.parametrize("d", [27392, 14336, 64, 40000])
@pytest.mark.parametrize("num_tokens", [1, 3, 33, 128, 513, 4096, 16384])
def test_silu_and_mul_per_token_quant(
    num_tokens: int, d: int, dynamic: bool, dtype: torch.dtype, default_vllm_config
):
    """Compare with Inductor on finite inputs and rows with NaN or infinity."""
    torch.manual_seed(0)
    torch._dynamo.reset()
    ref = torch.compile(_native_silu_mul_per_token_quant, dynamic=dynamic)
    x = torch.randn(num_tokens, 2 * d, device="cuda", dtype=dtype) * 2
    zeros = x.clone()
    zeros[: max(1, num_tokens // 3)] = 0
    for inp in (x, zeros, x * 300, x * 1e-6, _special_rows(x)):
        q_ref, s_ref = ref(inp)
        q, s = torch.ops.vllm.silu_and_mul_per_token_quant(inp)
        assert q.dtype == q_ref.dtype
        torch.testing.assert_close(s, s_ref, atol=0, rtol=2e-6, equal_nan=True)
        # An FP32 rounding difference at an E4M3 boundary can change one code.
        torch.testing.assert_close(
            q.float(), q_ref.float(), atol=2**-9, rtol=1 / 8, equal_nan=True
        )


def _ct_config(*groups):
    """compressed-tensors config, one group per (weights, input_activations)."""
    from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
        CompressedTensorsConfig,
    )

    config_groups = {
        f"group_{i}": {
            "targets": [f"re:.*layer_{i}$"],
            "weights": weights,
            "input_activations": input_activations,
        }
        for i, (weights, input_activations) in enumerate(groups)
    }
    return CompressedTensorsConfig.from_config(
        {
            "format": "float-quantized",
            "quant_method": "compressed-tensors",
            "config_groups": config_groups,
            "ignore": ["lm_head"],
        }
    )


_W = {"num_bits": 8, "type": "float", "symmetric": True, "dynamic": False}
_A = {"num_bits": 8, "type": "float", "symmetric": True}
W_CHANNEL = {**_W, "strategy": "channel"}
W_TENSOR = {**_W, "strategy": "tensor"}
W_BLOCK = {**_W, "strategy": "block", "block_structure": [128, 128]}
A_TOKEN = {**_A, "dynamic": True, "strategy": "token"}
A_DYNAMIC_TENSOR = {**_A, "dynamic": True, "strategy": "tensor"}
A_STATIC_TENSOR = {**_A, "dynamic": False, "strategy": "tensor"}
A_GROUP = {**_A, "dynamic": True, "strategy": "group", "group_size": 128}

QUANT_CONFIGS = {
    "token": [(W_CHANNEL, A_TOKEN)],
    "tensor_weight_token": [(W_TENSOR, A_TOKEN)],
    "dynamic_tensor": [(W_CHANNEL, A_DYNAMIC_TENSOR)],
    "static": [(W_TENSOR, A_STATIC_TENSOR)],
    "block": [(W_BLOCK, A_GROUP)],
    "weight_only": [(W_CHANNEL, None)],
    "mixed_static_token": [(W_TENSOR, A_STATIC_TENSOR), (W_CHANNEL, A_TOKEN)],
}


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("name", list(QUANT_CONFIGS))
def test_compressed_tensors_dynamic_per_token_fp8_detection(name, monkeypatch):
    """Detection follows the activation quant key the linear layers run."""
    from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (  # noqa: E501
        compressed_tensors_w8a8_fp8,
        compressed_tensors_w8a16_fp8,
    )

    # The FP8 schemes read the model dtype from the current config.
    stub = type("Stub", (), {"model_config": type("M", (), {"dtype": torch.bfloat16})})
    for module in (compressed_tensors_w8a8_fp8, compressed_tensors_w8a16_fp8):
        monkeypatch.setattr(module, "get_current_vllm_config", stub)

    config = _ct_config(*QUANT_CONFIGS[name])
    runs_per_token = False
    for scheme_args in config.target_scheme_map.values():
        scheme = config._get_scheme_from_parts(
            weight_quant=scheme_args["weights"],
            input_quant=scheme_args["input_activations"],
            format=scheme_args["format"],
        )
        key = getattr(scheme, "activation_quant_key", None)
        runs_per_token |= key == kFp8DynamicTokenSym
    assert runs_per_token == (
        name in ("token", "tensor_weight_token", "dynamic_tensor", "mixed_static_token")
    )
    assert config.has_dynamic_per_token_fp8_activations() == runs_per_token


@pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA only")
@pytest.mark.parametrize("quant", [None, *QUANT_CONFIGS])
@pytest.mark.parametrize("fuse_act_quant", [None, True, False])
@pytest.mark.parametrize("custom_op", [None, "+silu_and_mul", "+quant_fp8"])
def test_act_quant_fusion_patterns_per_config(quant, fuse_act_quant, custom_op):
    """Default: on where it was before, and also where the per-token pattern
    can match. Explicit False: no activation+quant fusion at all. When on,
    every pattern is registered, the per-token one included."""
    from vllm.compilation.passes.fusion.act_quant_fusion import (
        SiluMulBlockQuantPattern,
        SiluMulFp8StaticQuantPattern,
        SiluMulNvfp4QuantPattern,
        SiluMulPerTokenQuantPattern,
        silu_and_mul_nvfp4_quant_supported,
    )
    from vllm.compilation.passes.pass_manager import PostGradPassManager

    config = VllmConfig(
        compilation_config=CompilationConfig(
            mode=CompilationMode.VLLM_COMPILE,
            custom_ops=["none"] + ([custom_op] if custom_op else []),
            pass_config=PassConfig(
                fuse_act_quant=fuse_act_quant, fuse_norm_quant=False
            ),
        ),
        quant_config=_ct_config(*QUANT_CONFIGS[quant]) if quant else None,
    )
    pass_manager = PostGradPassManager()
    pass_manager.configure(config)
    act_passes = [
        p for p in pass_manager.passes if isinstance(p, ActivationQuantFusionPass)
    ]
    patterns = [type(pr) for p in act_passes for pr in p._pattern_replacements]

    cc = config.compilation_config
    custom = cc.is_custom_op_enabled("silu_and_mul") or cc.is_custom_op_enabled(
        "quant_fp8"
    )
    can_match = not custom and quant in (
        "token",
        "tensor_weight_token",
        "dynamic_tensor",
        "mixed_static_token",
    )
    # Default: either custom op on (no FP4 model here; block FP8 weights turn
    # +quant_fp8 on), or layers that quantize activations to FP8 per token.
    if fuse_act_quant is None:
        fuse_act_quant = custom or can_match
    assert cc.pass_config.fuse_act_quant is fuse_act_quant
    expected = [SiluMulFp8StaticQuantPattern, SiluMulPerTokenQuantPattern]
    expected += [SiluMulNvfp4QuantPattern] * silu_and_mul_nvfp4_quant_supported
    expected += [SiluMulBlockQuantPattern] * 16
    assert patterns == (expected if fuse_act_quant else [])
    assert len(act_passes) == int(fuse_act_quant)
