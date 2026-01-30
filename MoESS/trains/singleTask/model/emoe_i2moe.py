import torch
import torch.nn as nn

from src.imoe.InteractionMoE import InteractionMoE

from .emoe import EMOE


class EMOEFusionWrapper(nn.Module):
    """Wrap EMOE as a fusion model for I2MOE InteractionMoE."""

    def __init__(self, args):
        super().__init__()
        self.emoe = EMOE(args)

    def forward(self, inputs):
        eeg, eog = inputs
        output = self.emoe(eeg, eog)
        return output["logits_c"]


class EMOEI2MOE(nn.Module):
    """EMOE experts trained with the I2MOE framework."""

    def __init__(self, args, seq_len):
        super().__init__()
        hidden_dim = int(getattr(args, "i2moe_hidden_dim", seq_len))
        hidden_dim_rw = int(getattr(args, "i2moe_hidden_dim_rw", 256))
        num_layer_rw = int(getattr(args, "i2moe_num_layer_rw", 2))
        temperature_rw = float(getattr(args, "i2moe_temperature_rw", 1.0))

        fusion_model = EMOEFusionWrapper(args)
        self.ensemble = InteractionMoE(
            num_modalities=2,
            fusion_model=fusion_model,
            fusion_sparse=False,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layer_rw=num_layer_rw,
            temperature_rw=temperature_rw,
        )

    def forward(self, eeg, eog):
        expert_outputs, routing_weights, logits, interaction_losses = self.ensemble(
            [eeg, eog]
        )
        return {
            "logits": logits,
            "interaction_losses": interaction_losses,
            "routing_weights": routing_weights,
            "expert_outputs": expert_outputs,
        }

    def inference(self, eeg, eog):
        expert_outputs, routing_weights, logits = self.ensemble.inference([eeg, eog])
        return {
            "logits": logits,
            "routing_weights": routing_weights,
            "expert_outputs": expert_outputs,
        }
