"""
# Example task models for protein language modeling and classification tasks.
This module provides example implementations of task-specific models for protein language modeling
and classification tasks. It includes a modular MLP classifier head for both protein-level 
and residue-level classification, as well as a message-passing neural network (MPNN) classifier
head for residue-level tasks.
"""

from typing import Optional, List
import torch
from torch import nn
from torch_geometric.nn import GCNConv


class MLPHead(nn.Module):
    """Modular MLP head with configurable pooling method.
    Supports per-protein (mean or CLS) or per-residue classification."""

    def __init__(
        self,
        input_dim: int,         # Should match the pLM embedding dimension
        hidden_dim: int,        # Hidden layer dimensions
        output_dim: int,        # Number of classes
        num_hidden_layers: int = 1, # Variable number of hidden layers in MLP
        dropout_rate: float = 0.1,  # Dropout rate
        classification_mode: str = "protein", # 'protein' or 'residue'
        pooling_strategy: str = "mean",       # 'mean' or 'cls' when protein-level
    ):
        """
        Initializes the MLP prediction head.
        """
        super().__init__()
        # Define classification mode
        assert classification_mode in (
            "protein",
            "residue",
        ), "classification_mode must be 'protein' or 'residue'"
        # Define pooling strategy
        if classification_mode=="protein":
            assert pooling_strategy in (
                    "mean",
                    "cls",
                ), "pooling_strategy must be 'mean' or 'cls'"
        self.classification_mode = classification_mode
        self.pooling_strategy = pooling_strategy

        # Define the architecture with the input num_hidden_layers
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
        if self.classification_mode == "protein": # Protein-level prediction
            if self.pooling_strategy == "cls": # BERT-style 'cls' token
                x = hidden_states[:, 0, :]
            else: # Use mean of embeddings 
                if attention_mask is not None: # Exclude padding from the mean
                    mask = attention_mask.unsqueeze(-1)
                    masked = hidden_states * mask
                    sum_hidden = masked.sum(dim=1)
                    lengths = mask.sum(dim=1).clamp(min=1)
                    x = sum_hidden / lengths
                else:
                    x = hidden_states.mean(dim=1)
            return self.classifier(x)

        else: # Residue-level prediction
            B, L, D = hidden_states.shape
            flat = hidden_states.view(B * L, D)
            logits_flat = self.classifier(flat)
            return logits_flat.view(B, L, self.output_dim)


class MPNNClassifierHead(nn.Module):
    """Graph-based MPNN head for residue-level classification/regression."""
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_steps: int = 3,
        dropout_rate: float = 0.1,
    ):
        """
        Initializes the MPNN classifier head.
        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (int): Hidden dimension for MPNN layers.
            output_dim (int): Output dimension (number of classes).
            num_steps (int): Number of MPNN steps.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__()
        self.project = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        ) # Project input to MPNN hidden dimension
        self.convs = nn.ModuleList(
            [
                GCNConv(hidden_dim, hidden_dim, aggr="mean")
                for _ in range(num_steps)
            ]
        ) # Create MPNN layers
        self.classifier = nn.Linear(hidden_dim, output_dim) # Classifier layer
        # Store parameters
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        # Store MPNN configuration
        self.num_steps = num_steps
        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        node_features: torch.FloatTensor,
        edge_index: torch.LongTensor = None,
        edge_attr: Optional[torch.FloatTensor] = None,
    ) -> torch.FloatTensor:
        """
        Forward pass for the MPNN classifier head.
        Args:
            node_features: Node features of shape (B, L, D)
            edge_index: Graph connectivity in COO format (2, E)
            edge_attr: Edge features of shape (E, D_edge) or None
        Returns:
            logits_flat: Node-level logits of shape (B, L, output_dim)
        """
        B, L, D = node_features.shape
        # Flatten the nodes and project to MPNN hidden dimension
        x = node_features.view(B * L, D)
        x = self.project(x)
        # Apply GCN layers
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = torch.tanh(x)
            x = self.dropout(x)
        # Classification per node
        logits_flat = self.classifier(x)
        return logits_flat.view(B, L, -1)
