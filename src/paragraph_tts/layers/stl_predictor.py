"""Contains the components of the STL predictor model."""

import pydantic
import torch
from torch_geometric.nn import HeteroConv, GATv2Conv
from torch_geometric.data import HeteroData

from paragraph_tts.layers.shared import moe


class _GATTransformerLayer(torch.nn.Module):
    """A single layer of the GAT-based transformer encoder used in the STL predictor model."""

    def __init__(self,
                 hidden_dim: int,
                 n_heads: int,
                 fft_proj_dim: int,
                 n_experts: int,
                 top_k_experts: int,
                 dropout: float) -> None:

        super().__init__()

        supported_edge_types = (
            ('word_emb', 'follows', 'word_emb'),
            ('word_emb', 'precedes', 'word_emb'),
            ('global_emb', 'follows', 'global_emb'),
            ('global_emb', 'precedes', 'global_emb'),
            ('word_emb', 'same_utt', 'global_emb'),
        )

        self._gat_layer = HeteroConv({
            edge_type: GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=n_heads,
                concat=False,
                add_self_loops=False,
                dropout=dropout
            ) for edge_type in supported_edge_types
        }, aggr='sum')

        self._norms_1 = torch.nn.ModuleDict({
            node_type: torch.nn.RMSNorm(hidden_dim) for node_type in ['word_emb', 'global_emb']
        })

        self._fft_proj = torch.nn.ModuleDict({
            node_type: moe.MoEProjection(input_dim=hidden_dim,
                                         proj_dim=fft_proj_dim,
                                         n_experts=n_experts,
                                         top_k=top_k_experts,
                                         dropout=dropout)
            for node_type in ['word_emb', 'global_emb']
        })

        self._norms_2 = torch.nn.ModuleDict({
            node_type: torch.nn.RMSNorm(hidden_dim) for node_type in ['word_emb', 'global_emb']
        })

    def forward(self, x_dict, edge_index_dict):  # type: ignore
        """Performs a forward pass through the GAT-based transformer layer."""

        original_x_dict = {x_type: x for x_type, x in x_dict.items()}
        top_k_indices_dict = {}
        router_logits_dict = {}

        for node_type in x_dict:
            x_dict[node_type] = self._norms_1[node_type](x_dict[node_type])

        x_dict = self._gat_layer(x_dict, edge_index_dict)

        for node_type in x_dict:
            x = x_dict[node_type] + original_x_dict[node_type]

            (proj_output,
             topk_indices,
             router_logits) = self._fft_proj[node_type](self._norms_2[node_type](x))

            x_dict[node_type] = x + proj_output

            top_k_indices_dict[node_type] = topk_indices
            router_logits_dict[node_type] = router_logits

        return x_dict, top_k_indices_dict, router_logits_dict


class Encoder(torch.nn.Module):
    """Main component of the STL predictor model.

    It encodes the input graph and predicts the GST/WSV weights for the individual sentences. 
    """

    class Configuration(pydantic.BaseModel):
        """Configuration for the STL predictor encoder."""

        input_embedding_dim: int
        hidden_dim: int
        num_heads: int
        fft_proj_dim: int
        n_experts: int
        top_k_experts: int
        num_layers: int
        gst_weights_dim: int
        wsv_weights_dim: int
        dropout: float

    def __init__(self, config: Configuration) -> None:

        super().__init__()

        self._global_prenet = torch.nn.Linear(config.input_embedding_dim, config.hidden_dim)
        self._local_prenet = torch.nn.Linear(config.input_embedding_dim, config.hidden_dim)

        self._blocks = torch.nn.ModuleList([
            _GATTransformerLayer(
                hidden_dim=config.hidden_dim,
                n_heads=config.num_heads,
                fft_proj_dim=config.fft_proj_dim,
                n_experts=config.n_experts,
                top_k_experts=config.top_k_experts,
                dropout=config.dropout
            ) for _ in range(config.num_layers)
        ])

        self._wsv_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.wsv_weights_dim)
        )
        self._gst_predictor = torch.nn.Sequential(
            torch.nn.Linear(config.hidden_dim, config.gst_weights_dim)
        )

    def forward(self, graph: HeteroData):  # type: ignore
        """Predicts the GST/WSV weights for the sentences in the input graph."""

        topk_indices_dict = {node_type: [] for node_type in ('word_emb', 'global_emb')}
        router_logits_dict = {node_type: [] for node_type in ('word_emb', 'global_emb')}
        x_dict = graph.x_dict

        x_dict['word_emb'] = self._local_prenet(x_dict['word_emb'])
        x_dict['global_emb'] = self._global_prenet(x_dict['global_emb'])

        for block in self._blocks:
            x_dict, topk_indices, router_logits = block(x_dict, graph.edge_index_dict)

            for node_type in ('word_emb', 'global_emb'):
                topk_indices_dict[node_type].append(topk_indices[node_type])
                router_logits_dict[node_type].append(router_logits[node_type])

        return {
            'wsv_logits': self._wsv_predictor(x_dict['word_emb']),
            'gst_logits': self._gst_predictor(x_dict['global_emb']),
            'topk_indices': topk_indices_dict,
            'router_logits': router_logits_dict
        }
