from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.nn import (
    GATConv,
    SAGEConv,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)


class EnsemblePooling(nn.Module):
    """Lightweight ensemble pooling used by FlowMiner."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.hidden_dim = hidden_dim

    @property
    def out_features(self) -> int:
        return self.hidden_dim * 3

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        mean_pool = global_mean_pool(x, batch)
        max_pool = global_max_pool(x, batch)
        weights = torch.sigmoid(self.attention(x))
        att_pool = global_add_pool(x * weights, batch)
        return torch.cat([mean_pool, max_pool, att_pool], dim=-1)


class FlowMiner(nn.Module):
    """Implementation of the FlowMiner encoder described in the paper.

    Parameters
    ----------
    use_idp:
        Whether to enable the Integrated Decision Pooling (IDP) module
        (node-level MLP + dropout before global pooling). When set to False,
        the model becomes a simpler GNN with plain global mean pooling,
        which is useful for ablation studies.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 128,
        sage_layers: int = 2,
        gat_layers: int = 2,
        gat_heads: int = 4,
        dropout: float = 0.2,
        use_idp: bool = True,
    ):
        super().__init__()
        self.dropout = dropout
        self.use_idp = use_idp
        self.sage_convs = nn.ModuleList()
        self.sage_norms = nn.ModuleList()
        self.sage_acts = nn.ModuleList()
        prev_dim = input_dim
        for _ in range(sage_layers):
            self.sage_convs.append(SAGEConv(prev_dim, hidden_dim))
            self.sage_norms.append(nn.BatchNorm1d(hidden_dim))
            self.sage_acts.append(nn.PReLU(hidden_dim))
            prev_dim = hidden_dim

        self.gat_convs = nn.ModuleList()
        self.gat_norms = nn.ModuleList()
        self.gat_acts = nn.ModuleList()
        for _ in range(gat_layers):
            self.gat_convs.append(GATConv(hidden_dim, hidden_dim, heads=gat_heads, concat=False, dropout=dropout))
            self.gat_norms.append(nn.BatchNorm1d(hidden_dim))
            self.gat_acts.append(nn.PReLU(hidden_dim))

        # IDP: 3 层 [Linear → PReLU → Dropout]
        if use_idp:
            self.idp_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.PReLU(hidden_dim),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            # 在关闭 IDP 的消融设置下，保持接口一致但跳过额外的 MLP。
            self.idp_mlp = nn.Identity()
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def embed(self, data) -> torch.Tensor:
        x, edge_index = data.x, data.edge_index
        for conv, bn, act in zip(self.sage_convs, self.sage_norms, self.sage_acts):
            x = conv(x, edge_index)
            x = bn(x)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        for conv, bn, act in zip(self.gat_convs, self.gat_norms, self.gat_acts):
            x = conv(x, edge_index)
            x = bn(x)
            x = act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return x

    def forward(self, data) -> torch.Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        node_embeddings = self.embed(data)
        node_embeddings = self.idp_mlp(node_embeddings)
        graph_embedding = global_mean_pool(node_embeddings, batch)
        return self.classifier(graph_embedding)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
