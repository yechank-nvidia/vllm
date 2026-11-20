# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any, NamedTuple

import torch
import torch._inductor.pattern_matcher as pm
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._inductor.pattern_matcher import PatternMatcherPass
from torch._ops import OpOverload
from torch.fx.experimental.symbolic_shapes import statically_known_true

import vllm.ir.ops
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers import fused_post_norm
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    GroupShape,
    QuantKey,
    ScaleDesc,
    get_fp8_min_max,
    kFp8Dynamic64Sym,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8DynamicTokenSym,
    kFp8StaticTensorSym,
    kNvfp4Dynamic,
    kStaticTensorScale,
)
from vllm.platforms import current_platform

from ..inductor_pass import enable_fake_mode
from ..vllm_inductor_pass import VllmInductorPass, VllmPatternMatcherPass
from .matcher_utils import (
    MatcherQuantFP8,
)

logger = init_logger(__name__)
FP8_DTYPE = current_platform.fp8_dtype()
FP4_DTYPE = torch.uint8


_RMS_NORM_OP = torch.ops.vllm_ir.rms_norm.default
_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default
_POST_NORM_ADD_OP = torch.ops.vllm.post_norm_add.default
_POST_NORM_ADD_FP8_QUANT_OP = torch.ops.vllm.post_norm_add_fp8_quant.default


# TODO: extend rmsnorm quant kernels to support mixed input/weight dtypes,
# and remove this check.
def _rms_input_weight_dtype_match(match: pm.Match) -> bool:
    """Prevent fusion when rms_norm input and weight dtypes differ."""
    for node in match.nodes:
        if node.target == _RMS_NORM_OP:
            # rms_norm(x, weight, epsilon, variance_size)
            x, weight = node.args[0], node.args[1]
        elif node.target == _FUSED_ADD_RMS_NORM_OP:
            # fused_add_rms_norm(x, residual, weight, epsilon, variance_size)
            x, weight = node.args[0], node.args[2]
        else:
            continue
        if isinstance(x, fx.Node) and isinstance(weight, fx.Node):
            return x.meta["val"].dtype == weight.meta["val"].dtype
    return True


def empty_bf16(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.bfloat16, device=current_platform.device_type
    )


def empty_fp32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.float32, device=current_platform.device_type
    )


def empty_i32(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.int32, device=current_platform.device_type
    )


def empty_i64(*args: Any, **kwargs: Any) -> torch.Tensor:
    return torch.empty(
        *args, **kwargs, dtype=torch.int64, device=current_platform.device_type
    )


RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default

QUANT_OPS: dict[QuantKey, OpOverload] = {
    kFp8StaticTensorSym: torch.ops._C.static_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTensorSym: torch.ops._C.dynamic_scaled_fp8_quant.default,  # noqa: E501
    kFp8DynamicTokenSym: torch.ops._C.dynamic_per_token_scaled_fp8_quant.default,  # noqa: E501
}
if hasattr(torch.ops._C, "per_token_group_fp8_quant"):
    QUANT_OPS[kFp8Dynamic128Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
    QUANT_OPS[kFp8Dynamic64Sym] = torch.ops._C.per_token_group_fp8_quant.default  # noqa: E501
if current_platform.is_cuda() and hasattr(torch.ops._C, "scaled_fp4_quant"):
    QUANT_OPS[kNvfp4Dynamic] = torch.ops._C.scaled_fp4_quant.out


class FusedRMSQuantKey(NamedTuple):
    """
    Named tuple for identifying the type of RMSNorm + quant fusion.
    quant: type of quantization
    fused_add: does the op also perform the residual add
    """

    quant: QuantKey
    fused_add: bool

    def __str__(self) -> str:
        return (
            f"FusedQuantKey({self.quant}, with"
            f"{'' if self.fused_add else 'out'} residual)"
        )


FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {
    FusedRMSQuantKey(
        kFp8StaticTensorSym, False
    ): torch.ops._C.rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8StaticTensorSym, True
    ): torch.ops._C.fused_add_rms_norm_static_fp8_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, False
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8DynamicTokenSym, True
    ): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic128Sym, False
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic128Sym, True
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic64Sym, False
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
    FusedRMSQuantKey(
        kFp8Dynamic64Sym, True
    ): torch.ops._C.rms_norm_per_block_quant.default,  # noqa: E501
}


class RMSNormQuantPattern:
    def __init__(
        self,
        epsilon: float,
        key: FusedRMSQuantKey,
        has_col_major_scales: bool = False,
        is_e8m0: bool = False,
        is_tma_aligned: bool = False,
    ) -> None:
        self.epsilon = epsilon
        self.quant_dtype = key.quant.dtype
        config = get_current_vllm_config()
        self.model_dtype = config.model_config.dtype if config.model_config else None

        assert key in FUSED_OPS, f"unsupported fused rmsnorm+quant op for {key}"
        self.FUSED_OP = FUSED_OPS[key]

        self.quant_matcher = MatcherQuantFP8(
            key.quant,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )


class RMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None:
        fused_key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, fused_key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        # Cannot use methods, as the self argument affects tracing
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            return self.quant_matcher(result_rms, scale)[0]

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> torch.Tensor:
            result = torch.empty(
                input.shape, device=input.device, dtype=self.quant_dtype
            )
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result
            return at[1]

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            self.quant_matcher.inputs()[1],  # scale
        ]
        pattern(*inputs)

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormStaticQuantPattern(RMSNormQuantPattern):
    def __init__(
        self, epsilon: float, quant_dtype: torch.dtype, symmetric: bool = True
    ) -> None:
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(
                dtype=quant_dtype, scale=kStaticTensorScale, symmetric=symmetric
            ),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result, _ = self.quant_matcher(result_rms, scale)

            return result, residual

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                residual=residual,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
            )

            # result, residual
            return at[1], at[2]

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
            self.quant_matcher.inputs()[1],  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        self.is_e8m0 = is_e8m0
        self.has_col_major_scales = has_col_major_scales
        self.is_tma_aligned = is_tma_aligned
        super().__init__(
            epsilon,
            key,
            has_col_major_scales=has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result = torch.empty(
                result_rms.shape,
                device=result_rms.device,
                dtype=self.quant_matcher.quant_key.dtype,
            )
            assert scale is not None
            fp8_min, fp8_max = get_fp8_min_max()

            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=result_rms,
                output_q=result,
                output_s=scale,
                group_size=self.quant_matcher.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.quant_matcher.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )

            return result, residual, scale

        def replacement(
            input: torch.Tensor,
            weight: torch.Tensor,
            residual: torch.Tensor,
            scale: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)

            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual,
                group_size=self.group_shape[1],
                is_scale_transposed=self.has_col_major_scales,
            )

            # result, residual, scale
            return at[1], at[3], at[2]

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
            self.quant_matcher.empty_f32(1, 1),  # scale
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class RMSNormGroupQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape,
        symmetric: bool = True,
        is_e8m0: bool = False,
        has_col_major_scales: bool = True,
        is_tma_aligned: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        self.group_shape = group_shape
        self.has_col_major_scales = has_col_major_scales
        self.is_tma_aligned = is_tma_aligned
        super().__init__(
            epsilon,
            key,
            has_col_major_scales=self.has_col_major_scales,
            is_e8m0=is_e8m0,
            is_tma_aligned=is_tma_aligned,
        )

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            result = torch.empty(
                result_rms.shape,
                device=result_rms.device,
                dtype=self.quant_matcher.quant_key.dtype,
            )
            assert scale is not None
            fp8_min, fp8_max = get_fp8_min_max()

            _, result, scale = auto_functionalized(
                self.quant_matcher.QUANT_OP,
                input=result_rms,
                output_q=result,
                output_s=scale,
                group_size=self.quant_matcher.quant_key.scale.group_shape[1],
                eps=1e-10,
                fp8_min=fp8_min,
                fp8_max=fp8_max,
                scale_ue8m0=self.quant_matcher.is_e8m0,
                dummy_is_scale_transposed=self.has_col_major_scales,
                dummy_is_tma_aligned=self.is_tma_aligned,
            )

            return result, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, scale: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None,
                group_size=self.group_shape[1],
                is_scale_transposed=self.has_col_major_scales,
            )

            # result, scale
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            [
                empty_bf16(5, 16),  # input
                empty_bf16(16),  # weight
                self.quant_matcher.empty_f32(1, 1),  # scale
            ],
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class RMSNormDynamicQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=False,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            result_rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)
            # result, scale
            return self.quant_matcher(result_rms)  # type: ignore[no-any-return]

        def replacement(
            input: torch.Tensor, weight: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(input)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=None,
            )

            # result, scale
            return at[1], at[2]

        pm.register_replacement(
            pattern,
            replacement,
            [
                empty_bf16(5, 16),  # input
                empty_bf16(16),  # weight
            ],
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


class FusedAddRMSNormDynamicQuantPattern(RMSNormQuantPattern):
    def __init__(
        self,
        epsilon: float,
        quant_dtype: torch.dtype,
        group_shape: GroupShape = GroupShape.PER_TOKEN,
        symmetric: bool = True,
    ) -> None:
        scale = ScaleDesc(torch.float32, False, group_shape)
        key = FusedRMSQuantKey(
            fused_add=True,
            quant=QuantKey(dtype=quant_dtype, scale=scale, symmetric=symmetric),
        )
        super().__init__(epsilon, key)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        def pattern(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result_rms, residual = vllm.ir.ops.fused_add_rms_norm(
                input, residual, weight, self.epsilon
            )
            result, scale = self.quant_matcher(result_rms)

            return result, residual, scale

        def replacement(
            input: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            # In case we're matching native rms-norm, conversions might be
            # optimized out. We convert here just to be safe.
            input = input.to(dtype=self.model_dtype)

            result = torch.empty_like(input, dtype=self.quant_dtype)
            scale = self.quant_matcher.make_scale(input)
            at = auto_functionalized(
                self.FUSED_OP,
                result=result,
                input=input,
                weight=weight,
                scale=scale,
                epsilon=self.epsilon,
                scale_ub=None,
                residual=residual,
            )

            # result, residual, scale
            return at[1], at[3], at[2]

        inputs = [
            empty_bf16(5, 16),  # input
            empty_bf16(16),  # weight
            empty_bf16(5, 16),  # residual
        ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_rms_input_weight_dtype_match,
        )


def _post_norm_kernel_supported(match: pm.Match) -> bool:
    """Match only what the post_norm kernels compute.

    The residual and every norm input must be contiguous rows of one shape and
    dtype (no broadcast, no type promotion), and every norm needs a weight of
    the hidden size (no weightless norm). Anything else stays with Inductor.
    """
    rows: list[Any] = [match.kwargs.get("residual")]
    weights: list[Any] = []
    for node in match.nodes:
        if node.target == _RMS_NORM_OP:
            # rms_norm(x, weight, epsilon, variance_size)
            rows.append(node.args[0])
            weights.append(node.args[1])
    vals = [n.meta.get("val") if isinstance(n, fx.Node) else None for n in rows]
    w_vals = [n.meta.get("val") if isinstance(n, fx.Node) else None for n in weights]
    if not all(isinstance(v, torch.Tensor) for v in vals + w_vals):
        return False  # e.g. weight=None

    def same(a: Any, b: Any) -> bool:
        return len(a) == len(b) and all(
            statically_known_true(i == j) for i, j in zip(a, b)
        )

    def contiguous(t: torch.Tensor) -> bool:
        expected: Any = 1
        for size, stride in zip(reversed(t.shape), reversed(t.stride())):
            if not statically_known_true(stride == expected):
                return False
            expected = expected * size
        return True

    ref = vals[0]
    return all(
        v.dtype == ref.dtype and same(v.shape, ref.shape) and contiguous(v)
        for v in vals
    ) and all(
        w.dtype == ref.dtype and same(w.shape, ref.shape[-1:]) and contiguous(w)
        for w in w_vals
    )


class PostNormAddDynamicQuantPattern:
    """Post-norm block: residual + rms_norm(x) * w -> dynamic per-token fp8 quant.

    The norm output must feed only the add. The quant becomes one
    post_norm_add_fp8_quant kernel. The sum is rebuilt from the native ops for
    its other users, so the next post-norm add can still match
    PostNormAddPattern (which recomputes it in fp32) and Inductor fuses any
    other use, as before.
    """

    def __init__(self, epsilon: float, residual_first: bool) -> None:
        self.epsilon = epsilon
        self.residual_first = residual_first
        self.quant_matcher = MatcherQuantFP8(kFp8DynamicTokenSym)

    def register(self, pm_pass: PatternMatcherPass) -> None:
        eps = self.epsilon
        residual_first = self.residual_first

        def add_norm(x, weight, residual):
            y = vllm.ir.ops.rms_norm(x, weight, eps)
            return residual + y if residual_first else y + residual

        def pattern(
            x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            out = add_norm(x, weight, residual)
            result, scale = self.quant_matcher(out)
            return out, result, scale

        def replacement(
            x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            result, scale = _POST_NORM_ADD_FP8_QUANT_OP(x, weight, residual, eps)
            return add_norm(x, weight, residual), result, scale

        pm.register_replacement(
            pattern,
            replacement,
            [empty_bf16(5, 16), empty_bf16(16), empty_bf16(5, 16)],
            pm.fwd_only,
            pm_pass,
            extra_check=_post_norm_kernel_supported,
        )


class PostNormAddPattern:
    """Post-norm block: residual + rms_norm(x) * w as one post_norm_add kernel.

    chained=True matches the second add of a post-norm block, whose residual
    is the first post-norm sum (prev_residual + rms_norm(prev_x) * prev_w)
    with no other user. The kernel recomputes that sum in fp32, as Inductor
    does for the unfused graph. If the first sum has other users (it is stored
    anyway), both adds match the plain pattern instead, which reads it.
    """

    def __init__(
        self,
        epsilon: float,
        residual_first: bool,
        chained: bool = False,
        prev_residual_first: bool = True,
    ) -> None:
        self.epsilon = epsilon
        self.residual_first = residual_first
        self.chained = chained
        self.prev_residual_first = prev_residual_first

    def register(self, pm_pass: PatternMatcherPass) -> None:
        eps = self.epsilon

        def add_norm(x, weight, residual, residual_first):
            y = vllm.ir.ops.rms_norm(x, weight, eps)
            return residual + y if residual_first else y + residual

        if not self.chained:

            def pattern(
                x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
            ) -> torch.Tensor:
                return add_norm(x, weight, residual, self.residual_first)

            def replacement(
                x: torch.Tensor, weight: torch.Tensor, residual: torch.Tensor
            ) -> torch.Tensor:
                return _POST_NORM_ADD_OP(x, weight, residual, eps)

            inputs = [empty_bf16(5, 16), empty_bf16(16), empty_bf16(5, 16)]
        else:

            def pattern(x, weight, prev_x, prev_weight, residual):  # type: ignore[misc]
                prev_sum = add_norm(
                    prev_x, prev_weight, residual, self.prev_residual_first
                )
                return add_norm(x, weight, prev_sum, self.residual_first)

            def replacement(x, weight, prev_x, prev_weight, residual):  # type: ignore[misc]
                return _POST_NORM_ADD_OP(x, weight, residual, eps, prev_x, prev_weight)

            inputs = [
                empty_bf16(5, 16),  # x
                empty_bf16(16),  # weight
                empty_bf16(5, 16),  # prev_x
                empty_bf16(16),  # prev_weight
                empty_bf16(5, 16),  # residual
            ]

        pm.register_replacement(
            pattern,
            replacement,
            inputs,
            pm.fwd_only,
            pm_pass,
            extra_check=_post_norm_kernel_supported,
        )


class RMSNormQuantFusionPass(VllmPatternMatcherPass):
    """
    This pass fuses rms_norm & quant custom ops into a fused rms_norm_quant op.
    It also supports fused_add_rms_norm.
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)

        self.patterns: PatternMatcherPass = PatternMatcherPass(
            pass_name="rmsnorm_quant_fusion_pass"
        )

        # Make sure fused add patterns are before simple rms norm,
        # as the latter is a subset of the former in torch ops
        for epsilon in [1e-5, 1e-6]:
            # Fuse fused_add_rms_norm + static fp8 quant
            FusedAddRMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

            # Fuse rms_norm + static fp8 quant
            RMSNormStaticQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

            # Fuse fused_add_rms_norm + dynamic per-token fp8 quant
            FusedAddRMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(
                self.patterns
            )

            # Fuse rms_norm + dynamic per-token fp8 quant
            RMSNormDynamicQuantPattern(epsilon, FP8_DTYPE).register(self.patterns)

            # Only register group quant patterns on CUDA/ROCm where the C++ op exists
            for group_shape in [GroupShape(1, 128), GroupShape(1, 64)]:
                for has_col_major_scales in [True, False]:
                    for is_e8m0 in [True, False]:
                        for is_tma_aligned in [False, True]:
                            # Fuse fused_add_rms_norm + fp8 group quant
                            FusedAddRMSNormGroupQuantPattern(
                                epsilon,
                                FP8_DTYPE,
                                group_shape=group_shape,
                                is_e8m0=is_e8m0,
                                has_col_major_scales=has_col_major_scales,
                                is_tma_aligned=is_tma_aligned,
                            ).register(self.patterns)

                            # Fuse rms_norm + fp8 group quant
                            RMSNormGroupQuantPattern(
                                epsilon,
                                FP8_DTYPE,
                                group_shape=group_shape,
                                is_e8m0=is_e8m0,
                                has_col_major_scales=has_col_major_scales,
                                is_tma_aligned=is_tma_aligned,
                            ).register(self.patterns)

        self.dump_patterns(config, self.patterns)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = self.patterns.apply(graph)
        logger.debug("Replaced %s patterns", self.matched_count)

    def uuid(self) -> str:
        return self.hash_source(
            self,
            RMSNormGroupQuantPattern,
            RMSNormQuantPattern,
            RMSNormStaticQuantPattern,
            RMSNormDynamicQuantPattern,
            FusedAddRMSNormStaticQuantPattern,
            FusedAddRMSNormDynamicQuantPattern,
            FusedAddRMSNormGroupQuantPattern,
        )


class PostNormFusionPass(VllmPatternMatcherPass):
    """Fuses post-norm blocks: residual + rms_norm(x) * w (+ per-token fp8 quant).

    vLLM's fused add + RMSNorm kernels and patterns cover the pre-norm order
    (add, then norm), so post-norm models (for example EXAONE 4, OLMo 2) leave
    these blocks to Inductor's multi-pass reduction kernels. Pre-norm graphs
    never match. Stages run in order so each one sees the previous rewrite:
      1. rms_norm -> add -> dynamic per-token fp8 quant
      2. rms_norm -> add, where the residual is the previous post-norm sum and
         has no other user (recomputed in fp32, as Inductor does)
      3. rms_norm -> add
    """

    @enable_fake_mode
    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        self.stages: list[PatternMatcherPass] = [
            PatternMatcherPass(pass_name=f"post_norm_fusion_pass_{i}") for i in range(3)
        ]
        orders = (True, False)
        for epsilon in [1e-5, 1e-6]:
            for first in orders:
                if torch.float8_e4m3fn == FP8_DTYPE:
                    PostNormAddDynamicQuantPattern(epsilon, first).register(
                        self.stages[0]
                    )
                for prev_first in orders:
                    PostNormAddPattern(
                        epsilon, first, chained=True, prev_residual_first=prev_first
                    ).register(self.stages[1])
                PostNormAddPattern(epsilon, first).register(self.stages[2])

        for stage in self.stages:
            self.dump_patterns(config, stage)

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        from torch._inductor.pattern_matcher import stable_topological_sort

        self.matched_count = 0
        for stage in self.stages:
            count = stage.apply(graph)
            if count:
                # Drop the rebuilt sums that lost their users before the next
                # stage looks at them.
                stable_topological_sort(graph)
                graph.eliminate_dead_code()
            self.matched_count += count
        logger.debug("Replaced %s post-norm patterns", self.matched_count)

    def uuid(self) -> str:
        return self.hash_source(
            self,
            PostNormAddDynamicQuantPattern,
            PostNormAddPattern,
            _post_norm_kernel_supported,
            fused_post_norm._launch,
            fused_post_norm._post_norm_kernel.src,
        )
