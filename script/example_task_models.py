"""
# Example task models for protein language modeling and classification tasks.
This module provides example implementations of task-specific models for protein language modeling
and classification tasks. It includes a modular MLP classifier head for both protein-level and residue-level classification,
as well as a message-passing neural network (MPNN) classifier head for residue-level tasks.
"""

from typing import Optional, List
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MLPClassifierHead(nn.Module):
    """Modular MLP head with configurable pooling, masked handling, and variable hidden layers;
    supports per-protein (mean or CLS) or per-residue classification."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_hidden_layers: int = 1,
        dropout_rate: float = 0.1,
        classification_mode: str = "protein",  # 'protein' or 'residue'
        pooling_strategy: str = "mean",  # 'mean' or 'cls' when protein-level
    ):
        super().__init__()
        assert classification_mode in (
            "protein",
            "residue",
        ), "classification_mode must be 'protein' or 'residue'"
        assert pooling_strategy in (
            "mean",
            "cls",
        ), "pooling_strategy must be 'mean' or 'cls'"
        self.classification_mode = classification_mode
        self.pooling_strategy = pooling_strategy
        self.output_dim = output_dim
        dims = [input_dim] + [hidden_dim] * num_hidden_layers + [output_dim]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Dropout(dropout_rate))
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.Tanh())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.classifier = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the MLP classifier head.
        """
        if self.classification_mode == "protein":
            if self.pooling_strategy == "cls":
                x = hidden_states[:, 0, :]
            else:
                if attention_mask is not None:
                    mask = attention_mask.unsqueeze(-1)
                    masked = hidden_states * mask
                    sum_hidden = masked.sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1)
                    x = sum_hidden / lengths
                else:
                    x = hidden_states.mean(dim=1)
            return self.classifier(x)

        B, L, D = hidden_states.shape
        flat = hidden_states.view(B * L, D)
        logits_flat = self.classifier(flat)
        return logits_flat.view(B, L, self.output_dim)


class MPNNClassifierHead(nn.Module):
    """Graph-based MPNN head for residue-level classification/regression."""

    def __init__(
        self,
        input_dim: int,
        mpnn_hidden_dim: int,
        output_dim: int,
        num_steps: int = 3,
        dropout_rate: float = 0.1,
    ):
        super().__init__()
        self.project = (
            nn.Linear(input_dim, mpnn_hidden_dim)
            if input_dim != mpnn_hidden_dim
            else nn.Identity()
        )
        self.convs = nn.ModuleList(
            [
                GCNConv(mpnn_hidden_dim, mpnn_hidden_dim, aggr="mean")
                for _ in range(num_steps)
            ]
        )
        self.num_steps = num_steps
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(mpnn_hidden_dim, output_dim)

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        edge_index: torch.LongTensor = None,
        edge_attr: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the MPNN classifier head.
        Args:
            hidden_states: Node features of shape (B, L, D)
            edge_index: Graph connectivity in COO format (2, E)
            edge_attr: Edge features of shape (E, D_edge) or None
        Returns:
            logits_flat: Node-level logits of shape (B, L, output_dim)
        """
        B, L, D = hidden_states.shape
        # Flatten to nodes
        x = hidden_states.view(B * L, D)
        x = self.project(x)
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.tanh(x)
            x = self.dropout(x)
        # Classification per node
        logits_flat = self.classifier(x)
        return logits_flat.view(B, L, -1)
