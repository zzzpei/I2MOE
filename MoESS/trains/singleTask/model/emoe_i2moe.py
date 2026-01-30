import sys
from pathlib import Path

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.imoe.InteractionMoE import InteractionMoE

from .emoe import DataExpert


class EMOEExpertWrapper(nn.Module):
    """Wrap EMOE data-expert architecture for I2MOE."""

    def __init__(self, args, mode: str):
        super().__init__()
        self.expert = DataExpert(args)
        self.mode = mode

    def forward(self, inputs):
        eeg, eog = inputs
        if self.mode == "eeg":
            return self.expert(eeg, eeg)
        if self.mode == "eog":
            return self.expert(eog, eog)
        if self.mode in {"synergy", "redundancy"}:
            return self.expert(eeg, eog)
        raise ValueError(f"Unknown EMOE expert mode: {self.mode}")


class EMOEI2MOE(nn.Module):
    """EMOE experts trained with the I2MOE framework."""

    def __init__(self, args, seq_len):
        super().__init__()
        hidden_dim = int(getattr(args, "i2moe_hidden_dim", seq_len))
        hidden_dim_rw = int(getattr(args, "i2moe_hidden_dim_rw", 256))
        num_layer_rw = int(getattr(args, "i2moe_num_layer_rw", 2))
        temperature_rw = float(getattr(args, "i2moe_temperature_rw", 1.0))

        expert_modes = ["eeg", "eog", "synergy", "redundancy"]
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
