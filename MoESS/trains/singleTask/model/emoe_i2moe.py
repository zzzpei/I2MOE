import torch
import torch.nn as nn

from src.imoe.InteractionMoE import InteractionMoE

from .emoe import EMOE


class EMOEExpertWrapper(nn.Module):
    """Wrap EMOE and expose a specific expert output for I2MOE."""

    def __init__(self, args, mode: str):
        super().__init__()
        self.emoe = EMOE(args)
        self.mode = mode

    def forward(self, inputs):
        eeg, eog = inputs
        output = self.emoe(eeg, eog)
        if self.mode == "eeg":
            return output["logits_eeg"]
        if self.mode == "eog":
            return output["logits_eog"]
        if self.mode == "data":
            return output["logits_data"]
        if self.mode == "fusion":
            return output["logits_c"]
        raise ValueError(f"Unknown EMOE expert mode: {self.mode}")


class EMOEI2MOE(nn.Module):
    """EMOE experts trained with the I2MOE framework."""

    def __init__(self, args, seq_len):
        super().__init__()
        hidden_dim = int(getattr(args, "i2moe_hidden_dim", seq_len))
        hidden_dim_rw = int(getattr(args, "i2moe_hidden_dim_rw", 256))
        num_layer_rw = int(getattr(args, "i2moe_num_layer_rw", 2))
        temperature_rw = float(getattr(args, "i2moe_temperature_rw", 1.0))

        expert_modes = ["eeg", "eog", "data", "fusion"]
        fusion_models = [EMOEExpertWrapper(args, mode) for mode in expert_modes]
        self.ensemble = InteractionMoE(
            num_modalities=2,
            fusion_models=fusion_models,
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
