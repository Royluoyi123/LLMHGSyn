from __future__ import annotations

from typing import List

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool


class MolecularGraphEncoder(nn.Module):
    """Two-layer GINE molecular encoder with explicit chemical-bond features."""

    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()
        self.input_layer = nn.Linear(atom_dim, hidden_dim)
        layer_dims = [(hidden_dim, hidden_dim), (hidden_dim, output_dim)]
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(
                        nn.Linear(input_dim, output_dim * 2),
                        nn.GELU(),
                        nn.Linear(output_dim * 2, output_dim),
                    ),
                    train_eps=True,
                    edge_dim=bond_dim,
                )
                for input_dim, output_dim in layer_dims
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(hidden_dim), nn.LayerNorm(output_dim)]
        )
        self.dropout = dropout

    def forward(
        self,
        atom_x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        molecule_batch: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.gelu(self.input_layer(atom_x))
        for conv, norm in zip(self.convs, self.norms):
            hidden = F.gelu(norm(conv(hidden, edge_index, edge_attr)))
            hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        mean_pool = global_mean_pool(hidden, molecule_batch)
        max_pool = global_max_pool(hidden, molecule_batch)
        return 0.5 * (mean_pool + max_pool)


class CellLineEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )

    def forward(self, expression: torch.Tensor) -> torch.Tensor:
        return self.network(expression)


class HypergraphLayer(nn.Module):
    """Node-to-hyperedge-to-node mean aggregation for 3-uniform synergy edges."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.message_linear = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = dropout

    def forward(self, nodes: torch.Tensor, incidence: torch.Tensor) -> torch.Tensor:
        node_ids, edge_ids = incidence[0], incidence[1]
        num_edges = int(edge_ids.max().item()) + 1
        edge_sum = nodes.new_zeros((num_edges, nodes.size(-1)))
        edge_sum.index_add_(0, edge_ids, nodes[node_ids])
        edge_degree = torch.bincount(edge_ids, minlength=num_edges).to(nodes.dtype).unsqueeze(-1)
        edge_hidden = edge_sum / edge_degree.clamp_min(1.0)

        node_sum = nodes.new_zeros(nodes.shape)
        node_sum.index_add_(0, node_ids, edge_hidden[edge_ids])
        node_degree = torch.bincount(node_ids, minlength=nodes.size(0)).to(nodes.dtype).unsqueeze(-1)
        messages = node_sum / node_degree.clamp_min(1.0)
        output = self.self_linear(nodes) + self.message_linear(messages)
        output = F.gelu(self.norm(output))
        return F.dropout(output, p=self.dropout, training=self.training)


class ConcatenationHyperedgeDecoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, nodes: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        drug_a, drug_b, cell = triples[:, 0], triples[:, 1], triples[:, 2]
        triple_features = torch.cat(
            [nodes[drug_a], nodes[drug_b], nodes[cell]], dim=-1
        )
        return self.network(triple_features).squeeze(-1)


class LLMHGNSynergy(nn.Module):
    def __init__(
        self,
        atom_dim: int,
        bond_dim: int,
        expression_dim: int,
        hidden_dim: int = 256,
        hgnn_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.drug_encoder = MolecularGraphEncoder(
            atom_dim,
            bond_dim,
            hidden_dim,
            hidden_dim,
            dropout,
        )
        self.cell_encoder = CellLineEncoder(expression_dim, hidden_dim, hidden_dim, dropout)
        self.hgnn_layers = nn.ModuleList(
            [HypergraphLayer(hidden_dim, dropout) for _ in range(hgnn_layers)]
        )
        self.fusion_logits = nn.Parameter(torch.zeros(hgnn_layers + 1))
        self.decoder = ConcatenationHyperedgeDecoder(hidden_dim, dropout)

    def encode_nodes(
        self,
        atom_x: torch.Tensor,
        molecule_edge_index: torch.Tensor,
        molecule_edge_attr: torch.Tensor,
        molecule_batch: torch.Tensor,
        expression: torch.Tensor,
        incidence: torch.Tensor,
    ) -> List[torch.Tensor]:
        drug_nodes = self.drug_encoder(
            atom_x,
            molecule_edge_index,
            molecule_edge_attr,
            molecule_batch,
        )
        cell_nodes = self.cell_encoder(expression)
        hidden = torch.cat([drug_nodes, cell_nodes], dim=0)
        states = [hidden]
        for layer in self.hgnn_layers:
            hidden = layer(hidden, incidence)
            states.append(hidden)
        return states

    def fuse_states(self, node_states: List[torch.Tensor]) -> torch.Tensor:
        if len(node_states) != self.fusion_logits.numel():
            raise ValueError(
                "The number of student scales does not match the fusion weights"
            )
        weights = F.softmax(self.fusion_logits, dim=0)
        return torch.stack(
            [weight * state for weight, state in zip(weights, node_states)],
            dim=0,
        ).sum(dim=0)

    def predict(self, node_states: torch.Tensor, triples: torch.Tensor) -> torch.Tensor:
        return self.decoder(node_states, triples)


class MultiScaleProjector(nn.Module):
    def __init__(
        self,
        teacher_dim: int,
        student_dim: int,
        num_scales: int,
        num_drugs: int,
    ):
        super().__init__()
        self.num_drugs = num_drugs
        self.drug_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(teacher_dim, student_dim),
                    nn.GELU(),
                    nn.LayerNorm(student_dim),
                )
                for _ in range(num_scales)
            ]
        )
        self.cell_projectors = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(teacher_dim, student_dim),
                    nn.GELU(),
                    nn.LayerNorm(student_dim),
                )
                for _ in range(num_scales)
            ]
        )
        self.scale_logits = nn.Parameter(torch.zeros(num_scales))

    def forward(self, teacher_states: torch.Tensor) -> List[torch.Tensor]:
        projected_states = []
        for scale, (drug_projector, cell_projector) in enumerate(
            zip(self.drug_projectors, self.cell_projectors)
        ):
            drug_states = drug_projector(
                teacher_states[scale, : self.num_drugs]
            )
            cell_states = cell_projector(
                teacher_states[scale, self.num_drugs :]
            )
            projected_states.append(torch.cat([drug_states, cell_states], dim=0))
        return projected_states
