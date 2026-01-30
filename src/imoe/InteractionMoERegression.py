import os
import sys

sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))

import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
from src.common.modules.common import MLP


class MLPReWeighting(nn.Module):
    """Use MLP to re-weight all interaction experts."""

    def __init__(
        self,
        num_modalities,
        num_branches,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layers=2,
        temperature=1,
    ):
        """args:
        hidden_dim: hidden dimension of input embeddings.
        hidden_dim_rw: hidden dimension of the re-weighting model.
        """
        super(MLPReWeighting, self).__init__()
        self.temperature = temperature
        self.mlp = MLP(
            hidden_dim * num_modalities,
            hidden_dim_rw,
            num_branches,
            num_layers,
            activation=nn.ReLU(),
            dropout=0.5,
        )

    def temperature_scaled_softmax(self, logits):
        logits = logits / self.temperature
        return torch.softmax(logits, dim=1)

    def forward(self, inputs):
        if inputs[0].dim() == 3:
            x = [item.mean(dim=1) for item in inputs]
            x = torch.cat(x, dim=1)
        else:
            x = torch.cat(inputs, dim=1)
        x = self.mlp(x)
        return self.temperature_scaled_softmax(x)


class InteractionExpert(nn.Module):
    """
    Interaction Expert.
    """

    def __init__(self, fusion_model, fusion_sparse):
        super(InteractionExpert, self).__init__()
        self.fusion_model = fusion_model
        self.fusion_sparse = fusion_sparse

    def forward(self, inputs):
        """
        Forward pass with all modalities present.
        """
        return self._forward_with_replacement(inputs, replace_index=None)

    def forward_with_replacement(self, inputs, replace_index):
        """
        Forward pass with one modality replaced by a random vector.

        Args:
            inputs (list of tensors): List of modality inputs.
            replace_index (int): Index of the modality to replace. If None, no modality is replaced.
        """
        return self._forward_with_replacement(inputs, replace_index=replace_index)

    def _forward_with_replacement(self, inputs, replace_index=None):
        """
        Internal function to handle forward pass with optional modality replacement.
        """
        # Replace specified modality with a random vector
        if replace_index is not None:
            random_vector = torch.randn_like(inputs[replace_index])
            inputs = (
                inputs[:replace_index] + [random_vector] + inputs[replace_index + 1 :]
            )

        x = self.fusion_model(inputs)
        if self.fusion_sparse:
            return x, self.fusion_model.gate_loss()

        return x

    def forward_multiple(self, inputs):
        """
        Perform a single forward pass with all modalities present.

        Args:
            inputs (list of tensors): List of modality inputs.

        Returns:
            List containing the output from the single forward pass.
        """
        if self.fusion_sparse:
            output, gate_loss = self.forward(inputs)
            return [output], [gate_loss]

        return [self.forward(inputs)]


class OutputHead(nn.Module):
    def __init__(self):
        super(OutputHead, self).__init__()
        self.proj = None

    def forward(self, x, output_dim):
        if self.proj is None:
            self.proj = nn.Linear(x.size(-1), output_dim).to(x.device)
        return self.proj(x)


class InteractionMoERegression(nn.Module):
    def __init__(
        self,
        num_modalities=3,
        fusion_model=None,
        fusion_sparse=True,
        hidden_dim=256,
        hidden_dim_rw=256,
        num_layer_rw=2,
        temperature_rw=1,
        shared_forward=True,
    ):
        super(InteractionMoERegression, self).__init__()
        num_branches = num_modalities + 1 + 1  # uni + syn + red
        self.num_modalities = num_modalities
        self.num_branches = num_branches
        self.shared_forward = shared_forward
        self.reweight = MLPReWeighting(
            num_modalities,
            num_branches,
            hidden_dim=hidden_dim,
            hidden_dim_rw=hidden_dim_rw,
            num_layers=num_layer_rw,
            temperature=temperature_rw,
        )
        if self.shared_forward:
            self.fusion_model = fusion_model
            self.output_heads = nn.ModuleList(
                [OutputHead() for _ in range(num_branches)]
            )
        else:
            self.interaction_experts = nn.ModuleList(
                [
                    InteractionExpert(deepcopy(fusion_model), fusion_sparse)
                    for _ in range(num_branches)
                ]
            )
        self.fusion_sparse = fusion_sparse

    def _forward_shared(self, inputs):
        try:
            output, latent = self.fusion_model(inputs, return_latent=True)
        except TypeError:
            output = self.fusion_model(inputs)
            latent = output
        return output, latent

    def uniqueness_loss_single(self, anchor, pos, neg):
        mse_loss = nn.MSELoss()
        return mse_loss(anchor, pos) - mse_loss(anchor, neg)

    def synergy_loss(self, anchor, negatives):
        mse_loss = nn.MSELoss()
        total_syn_loss = 0
        for negative in negatives:
            total_syn_loss += mse_loss(anchor, negative)
        return total_syn_loss / len(negatives)

    def redundancy_loss(self, anchor, positives):
        mse_loss = nn.MSELoss()
        total_redundancy_loss = 0
        for positive in positives:
            total_redundancy_loss += mse_loss(anchor, positive)
        return total_redundancy_loss / len(positives)

    def forward(self, inputs, return_expert_outputs=True):
        if self.shared_forward:
            base_output, latent = self._forward_shared(inputs)
            head_outputs = []
            expert_outputs = [] if return_expert_outputs else None
            for head in self.output_heads:
                head_output = head(latent, base_output.size(-1))
                head_outputs.append(head_output)
                if return_expert_outputs:
                    expert_outputs.append([head_output])

            all_predictions = torch.stack(head_outputs, dim=1)
            interaction_losses = [
                torch.zeros(
                    (), device=base_output.device, dtype=base_output.dtype
                )
                for _ in range(self.num_branches)
            ]
            if self.fusion_sparse:
                gate_loss = self.fusion_model.gate_loss()
                expert_gate_losses = [gate_loss for _ in range(self.num_branches)]
        else:
            expert_outputs = [] if return_expert_outputs else None
            all_predictions = []
            interaction_losses = []

            if self.fusion_sparse:
                expert_gate_losses = []

            for expert_idx, expert in enumerate(self.interaction_experts):
                if self.fusion_sparse:
                    expert_output, expert_gate_loss = expert.forward_multiple(inputs)
                    expert_gate_losses.append(expert_gate_loss)
                else:
                    expert_output = expert.forward_multiple(inputs)

                if return_expert_outputs:
                    expert_outputs.append(expert_output)

                all_predictions.append(expert_output[0])

                if len(expert_output) <= 1:
                    interaction_losses.append(
                        torch.zeros(
                            (),
                            device=expert_output[0].device,
                            dtype=expert_output[0].dtype,
                        )
                    )
                    continue

                if expert_idx < self.num_modalities:
                    uniqueness_loss = 0
                    anchor = expert_output[0]
                    neg = expert_output[expert_idx + 1]
                    positives = expert_output[1 : expert_idx + 1] + expert_output[
                        expert_idx + 2 :
                    ]
                    for pos in positives:
                        uniqueness_loss += self.uniqueness_loss_single(anchor, pos, neg)
                    interaction_losses.append(uniqueness_loss / len(positives))
                elif expert_idx == len(self.interaction_experts) - 2:
                    synergy_anchor = expert_output[0]
                    synergy_negatives = torch.stack(expert_output[1:])
                    interaction_losses.append(
                        self.synergy_loss(synergy_anchor, synergy_negatives)
                    )
                else:
                    redundancy_anchor = expert_output[0]
                    redundancy_positives = torch.stack(expert_output[1:])
                    interaction_losses.append(
                        self.redundancy_loss(redundancy_anchor, redundancy_positives)
                    )

            all_predictions = torch.stack(all_predictions, dim=1)

        ###### MLP reweighting the experts' outputs ######
        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_predictions = (all_predictions * weights_transposed).sum(dim=1)

        if self.fusion_sparse:
            return (
                expert_outputs,
                interaction_weights,
                weighted_predictions,
                interaction_losses,
                expert_gate_losses,
            )

        return (
            expert_outputs,
            interaction_weights,
            weighted_predictions,
            interaction_losses,
        )

    def inference(self, inputs):
        if self.shared_forward:
            base_output, latent = self._forward_shared(inputs)
            expert_outputs = [
                head(latent, base_output.size(-1)) for head in self.output_heads
            ]
            all_predictions = torch.stack(expert_outputs, dim=1)
        else:
            # Get outputs for each interaction type
            expert_outputs = []
            if self.fusion_sparse:
                for expert in self.interaction_experts:
                    expert_output, _ = expert.forward(inputs)
                    expert_outputs.append(expert_output)
            else:
                for expert in self.interaction_experts:
                    expert_outputs.append(expert.forward(inputs))

            all_predictions = torch.stack(expert_outputs, dim=1)

        interaction_weights = self.reweight(inputs)  # Get interaction weights
        weights_transposed = interaction_weights.unsqueeze(2)
        weighted_predictions = (all_predictions * weights_transposed).sum(dim=1)

        return expert_outputs, interaction_weights, weighted_predictions
