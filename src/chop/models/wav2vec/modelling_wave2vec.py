import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from transformers import Wav2Vec2Model, Wav2Vec2Config
from chop.models.utils import register_mase_model, register_mase_checkpoint

logger = logging.getLogger(__name__)


@register_mase_model(
    name="wav2vec",
    checkpoints=[
        "wav2vec2-base",
        "wav2vec2-base-960h",
        "wav2vec2-large",
        "wav2vec2-large-960h",
        "wav2vec2-large-xlsr",
    ],
    model_source="hf_transformers",
    task_type="speech",
    is_fx_traceable=True,
)
class Wav2VecModelWrapper(nn.Module):
    def __init__(self, config, **kwargs):
        super().__init__()
        self.model = Wav2Vec2Model(config)
        # Expose weights to top level to make them accessible for CHOP
        self._expose_weights()

    def _expose_weights(self):
        """Expose nested model weights to make them visible to CHOP's metadata extraction."""
        # Process layers in feature encoder (convolutional layers)
        if hasattr(self.model, "feature_extractor"):
            for name, module in self.model.feature_extractor.named_modules():
                if hasattr(module, "weight") and isinstance(
                    module.weight, torch.Tensor
                ):
                    # Create flattened name by replacing dots with underscores
                    flat_name = f"weight_feature_extractor_{name.replace('.', '_')}"
                    setattr(self, flat_name, module.weight)

                    if hasattr(module, "bias") and module.bias is not None:
                        flat_bias_name = (
                            f"bias_feature_extractor_{name.replace('.', '_')}"
                        )
                        setattr(self, flat_bias_name, module.bias)

        # Process transformer layers
        if hasattr(self.model, "encoder"):
            # Extract weights from transformer layers
            for layer_idx, layer in enumerate(self.model.encoder.layers):
                # Handle attention weights
                if hasattr(layer, "attention"):
                    attn = layer.attention
                    if hasattr(attn, "q_proj") and hasattr(attn.q_proj, "weight"):
                        setattr(
                            self, f"weight_layer_{layer_idx}_q_proj", attn.q_proj.weight
                        )
                        if (
                            hasattr(attn.q_proj, "bias")
                            and attn.q_proj.bias is not None
                        ):
                            setattr(
                                self, f"bias_layer_{layer_idx}_q_proj", attn.q_proj.bias
                            )

                    if hasattr(attn, "k_proj") and hasattr(attn.k_proj, "weight"):
                        setattr(
                            self, f"weight_layer_{layer_idx}_k_proj", attn.k_proj.weight
                        )
                        if (
                            hasattr(attn.k_proj, "bias")
                            and attn.k_proj.bias is not None
                        ):
                            setattr(
                                self, f"bias_layer_{layer_idx}_k_proj", attn.k_proj.bias
                            )

                    if hasattr(attn, "v_proj") and hasattr(attn.v_proj, "weight"):
                        setattr(
                            self, f"weight_layer_{layer_idx}_v_proj", attn.v_proj.weight
                        )
                        if (
                            hasattr(attn.v_proj, "bias")
                            and attn.v_proj.bias is not None
                        ):
                            setattr(
                                self, f"bias_layer_{layer_idx}_v_proj", attn.v_proj.bias
                            )

                    if hasattr(attn, "out_proj") and hasattr(attn.out_proj, "weight"):
                        setattr(
                            self,
                            f"weight_layer_{layer_idx}_out_proj",
                            attn.out_proj.weight,
                        )
                        if (
                            hasattr(attn.out_proj, "bias")
                            and attn.out_proj.bias is not None
                        ):
                            setattr(
                                self,
                                f"bias_layer_{layer_idx}_out_proj",
                                attn.out_proj.bias,
                            )

                # Handle feed-forward weights
                if hasattr(layer, "feed_forward"):
                    ff = layer.feed_forward
                    if hasattr(ff, "intermediate_dense") and hasattr(
                        ff.intermediate_dense, "weight"
                    ):
                        setattr(
                            self,
                            f"weight_layer_{layer_idx}_ff_intermediate",
                            ff.intermediate_dense.weight,
                        )
                        if (
                            hasattr(ff.intermediate_dense, "bias")
                            and ff.intermediate_dense.bias is not None
                        ):
                            setattr(
                                self,
                                f"bias_layer_{layer_idx}_ff_intermediate",
                                ff.intermediate_dense.bias,
                            )

                    if hasattr(ff, "output_dense") and hasattr(
                        ff.output_dense, "weight"
                    ):
                        setattr(
                            self,
                            f"weight_layer_{layer_idx}_ff_output",
                            ff.output_dense.weight,
                        )
                        if (
                            hasattr(ff.output_dense, "bias")
                            and ff.output_dense.bias is not None
                        ):
                            setattr(
                                self,
                                f"bias_layer_{layer_idx}_ff_output",
                                ff.output_dense.bias,
                            )

    def forward(
        self,
        input_values,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.model(input_values, **kwargs)


def _get_wav2vec_model(model_size: str, pretrained: bool = False, **kwargs):
    """Helper function to get Wav2Vec models"""
    if pretrained:
        model_id = f"facebook/wav2vec2-{model_size}"
        model = Wav2Vec2Model.from_pretrained(model_id)
        logger.info(f"Loaded pretrained Wav2Vec2 {model_size} model")
    else:
        config = Wav2Vec2Config.from_pretrained(f"facebook/wav2vec2-{model_size}")
        model = Wav2VecModelWrapper(config)
        logger.info(f"Initialized Wav2Vec2 {model_size} model from config")

    return model


@register_mase_checkpoint("wav2vec2-base")
def get_wav2vec_base(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("base", pretrained, **kwargs)


@register_mase_checkpoint("wav2vec2-base-960h")
def get_wav2vec_base_960h(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("base", pretrained, **kwargs)


@register_mase_checkpoint("wav2vec2-large")
def get_wav2vec_large(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large", pretrained, **kwargs)


@register_mase_checkpoint("wav2vec2-large-960h")
def get_wav2vec_large_960h(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large-960h", pretrained, **kwargs)


@register_mase_checkpoint("wav2vec2-large-xlsr")
def get_wav2vec_large_xlsr(pretrained: bool = False, **kwargs):
    return _get_wav2vec_model("large-xlsr-53", pretrained, **kwargs)


class CombinedWav2Vec2CTC(nn.Module):
    def __init__(self, encoder, ctc_head, blank_id=0, beam_width=10, decoder=None):
        """
        Args:
            encoder: The traced encoder (e.g., mg.model)
            ctc_head: The CTC head (usually a linear layer)
            blank_id: The token ID for the blank symbol (typically 0)
            beam_width: Width for beam search decoding (if using a decoder)
            decoder: (Optional) A beam search decoder (e.g., from pyctcdecode)
        """
        super().__init__()
        self.encoder = encoder
        self.ctc_head = ctc_head
        self.blank_id = blank_id
        self.beam_width = beam_width
        self.decoder = decoder  # Only used during inference

    def forward(self, input_values, attention_mask=None, labels=None):
        encoder_out = self.encoder(input_values, attention_mask=attention_mask)
        hidden_states = encoder_out["last_hidden_state"]
        logits = self.ctc_head(hidden_states)  # outputs tensor as expected

        output = {"logits": logits, "labels": labels}

        if labels is not None:
            log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
            batch_size, time_steps, _ = logits.shape
            input_lengths = torch.full(
                (batch_size,), time_steps, dtype=torch.long, device=logits.device
            )
            target_lengths = (labels != -100).sum(dim=1)

            loss = F.ctc_loss(
                log_probs,
                labels,
                input_lengths,
                target_lengths,
                blank=self.blank_id,
                reduction="mean",
                zero_infinity=True,
            )
            output["loss"] = loss
        else:
            if self.decoder is not None:
                log_probs = logits.log_softmax(dim=-1)
                log_probs_np = log_probs[0].cpu().detach().numpy()
                transcription = self.decoder.decode(
                    log_probs_np, beam_width=self.beam_width
                ).lower()
                output["transcription"] = transcription
        return output
