# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inference-only EXAONE-4_5 MTP model."""

from collections.abc import Iterable

import torch
from torch import nn

from vllm.compilation.backends import set_model_tag
from vllm.compilation.decorators import support_torch_compile
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    ColumnParallelLinear,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import (  # noqa: E501
    CompressedTensorsLinearMethod,
)
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import (
    CompressedTensorsW8A8Fp8,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.models.exaone4 import Exaone4DecoderLayer
from vllm.model_executor.models.exaone_moe_mtp import (
    ExaoneMoeMTP,
    ExaoneMoeMultiTokenPredictor,
)
from vllm.sequence import IntermediateTensors

from .interfaces import (
    MultiModalEmbeddings,
    SupportsMultiModal,
    _require_is_multimodal,
)
from .utils import (
    AutoWeightsLoader,
    _merge_multimodal_embeddings,
    maybe_prefix,
)

logger = init_logger(__name__)

KVCache = tuple[torch.Tensor, torch.Tensor]


@support_torch_compile
class Exaone4_5MultiTokenPredictor(ExaoneMoeMultiTokenPredictor):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        nn.Module.__init__(self)

        model_config = vllm_config.model_config
        quant_config = vllm_config.quant_config
        lora_config = vllm_config.lora_config
        config = model_config.hf_config
        text_config = config.text_config

        self.config = config
        lora_vocab = (
            (lora_config.lora_extra_vocab_size * (lora_config.max_loras or 1))
            if lora_config
            else 0
        )
        self.vocab_size = config.vocab_size + lora_vocab
        self.org_vocab_size = config.vocab_size

        self.mtp_start_layer_idx = text_config.num_hidden_layers
        self.num_mtp_layers = getattr(config, "num_nextn_predict_layers", 1)

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            text_config.hidden_size,
            org_num_embeddings=config.vocab_size,
        )

        self.fc = ColumnParallelLinear(
            text_config.hidden_size * 2,
            text_config.hidden_size,
            gather_output=True,
            bias=False,
            return_bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.fc",
        )
        self.layers = nn.ModuleList(
            Exaone4DecoderLayer(
                text_config,
                quant_config=quant_config,
                prefix=f"{prefix}.layers.{idx}",
            )
            for idx in range(self.num_mtp_layers)
        )

        self.norm = RMSNorm(text_config.hidden_size, eps=text_config.rms_norm_eps)
        self.pre_fc_norm_hidden = RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )
        self.pre_fc_norm_embedding = RMSNorm(
            text_config.hidden_size, eps=text_config.rms_norm_eps
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        spec_step_idx: int = 0,
        *,
        output_token_indices: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if get_pp_group().is_first_rank:
            if inputs_embeds is None:
                inputs_embeds = self.get_input_embeddings(input_ids)
            assert hidden_states.shape[-1] == inputs_embeds.shape[-1]
            inputs_embeds = self.pre_fc_norm_embedding(inputs_embeds)
            hidden_states = self.pre_fc_norm_hidden(hidden_states)
            hidden_states = torch.cat([inputs_embeds, hidden_states], dim=-1)
            hidden_states = self.fc(hidden_states)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        current_step_idx = spec_step_idx % self.num_mtp_layers
        hidden_states, residual = self.layers[current_step_idx](
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
            output_token_indices=output_token_indices,
        )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


@support_torch_compile
class _SelectedRowsPredictor(nn.Module):
    def __init__(self, *, model: Exaone4_5MultiTokenPredictor):
        super().__init__()
        self.model = model

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        output_token_indices: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.forward(
            input_ids,
            positions,
            hidden_states,
            inputs_embeds=inputs_embeds,
            output_token_indices=output_token_indices,
        )


@support_torch_compile
class Exaone4_5_MTP(ExaoneMoeMTP, SupportsMultiModal):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        config = vllm_config.model_config.hf_config
        text_config = config.text_config
        self.vllm_config = vllm_config
        self.quant_config = vllm_config.quant_config

        nn.Module.__init__(self)
        self.config = config
        self.model = Exaone4_5MultiTokenPredictor(
            vllm_config=vllm_config, prefix=maybe_prefix(prefix, "mtp")
        )
        parallel_config = vllm_config.parallel_config
        compilation_config = vllm_config.compilation_config
        scheduler_config = vllm_config.scheduler_config
        self.supports_mtp_row_selection = (
            parallel_config.tensor_parallel_size == 1
            and parallel_config.pipeline_parallel_size == 1
            and parallel_config.data_parallel_size == 1
            and vllm_config.lora_config is None
            # The token and selected-row axes must stay independently dynamic.
            and not compilation_config.compile_sizes
            and not any(
                r.is_single_size() for r in compilation_config.get_compile_ranges()
            )
            and 1
            < scheduler_config.max_num_seqs
            < scheduler_config.max_num_batched_tokens
            # The projection and MLP must quantize activations independently
            # of the other rows in the batch.
            and all(
                isinstance(linear.quant_method, UnquantizedLinearMethod)
                or (
                    isinstance(linear.quant_method, CompressedTensorsLinearMethod)
                    and isinstance(linear.scheme, CompressedTensorsW8A8Fp8)
                )
                for layer in self.model.layers
                for linear in (
                    layer.self_attn.o_proj,
                    layer.mlp.gate_up_proj,
                    layer.mlp.down_proj,
                )
            )
        )
        if self.supports_mtp_row_selection:
            # Keep one weight path for loading. The wrapper compiles a second
            # signature over the same predictor and weights.
            with set_model_tag("eagle_head_selected_rows"):
                object.__setattr__(
                    self,
                    "forward_with_selected_rows",
                    _SelectedRowsPredictor(vllm_config=vllm_config, model=self.model),
                )
        self.unpadded_vocab_size = config.vocab_size
        self.lm_head = ParallelLMHead(
            self.unpadded_vocab_size,
            text_config.hidden_size,
            org_num_embeddings=config.vocab_size,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self.logits_processor = LogitsProcessor(
            self.unpadded_vocab_size, config.vocab_size
        )

    def embed_input_ids(
        self,
        input_ids: torch.Tensor,
        multimodal_embeddings: MultiModalEmbeddings | None = None,
        *,
        is_multimodal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        inputs_embeds = self._embed_text_input_ids(
            input_ids,
            self.model.embed_input_ids,
            is_multimodal=is_multimodal,
        )

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            return inputs_embeds

        is_multimodal = _require_is_multimodal(is_multimodal)

        inputs_embeds = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=multimodal_embeddings,
            is_multimodal=is_multimodal,
        )

        return inputs_embeds

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        shared_weight_names = ["embed_tokens", "lm_head"]

        def remap_weight_names(weights):
            for name, weight in weights:
                if name.startswith("mtp."):
                    name = name.replace("mtp.", "model.")
                elif any(key in name for key in shared_weight_names):
                    if "embed_tokens" in name:
                        name = name.replace("language_model.", "")
                else:
                    continue
                yield name, weight

        loader = AutoWeightsLoader(self)
        return loader.load_weights(remap_weight_names(weights))
