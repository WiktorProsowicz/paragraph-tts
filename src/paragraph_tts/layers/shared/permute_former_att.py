"""Contains implementation of PermuteFormer Multi-Head Attention module."""

from fast_transformers.attention import linear_attention
from fast_transformers.feature_maps import fourier_features
import matplotlib.pyplot as plt

import torch


class PermuteFormerMHA(linear_attention.LinearAttention):
    """Multi-Head Attention module with PermuteFormer style relative positional encoding."""

    def __init__(self,
                 d_model: int,
                 num_heads: int,
                 feature_map_dim: int):

        assert d_model % num_heads == 0, "d_model % num_heads should be zero."

        def feature_map_factory(query_dims):
            return fourier_features.Favor(query_dims, feature_map_dim)

        super().__init__(d_model // num_heads, feature_map_factory)

        self._head_size = d_model // num_heads
        self._feature_map_dim = feature_map_dim

        self._permutation = self._generate_permutation_sequence(num_heads,
                                                                feature_map_dim,
                                                                max_seq_length=5000,
                                                                seed=2137)

        self._q_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self._k_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self._v_proj = torch.nn.Linear(d_model, d_model, bias=False)
        self._out_proj = torch.nn.Linear(d_model, d_model)

    def forward(self,  # pylint: disable=arguments-differ
                queries: torch.Tensor,
                keys: torch.Tensor,
                values: torch.Tensor,
                key_mask: torch.Tensor,
                query_mask: torch.Tensor):
        """Computes weighted sum of values given queries and keys."""

        batch_size, query_len, _ = queries.size()
        _, key_len, _ = keys.size()
        _, value_len, _ = values.size()

        queries = (
            self._q_proj(queries)
            .view(batch_size, query_len, -1, self._head_size)
        )

        keys = (
            self._k_proj(keys)
            .view(batch_size, key_len, -1, self._head_size)
        )

        values = (
            self._v_proj(values)
            .view(batch_size, value_len, -1, self._head_size)
        )

        self.feature_map.new_feature_map(queries.device)
        queries = self.feature_map.forward_queries(queries)
        keys = self.feature_map.forward_keys(keys)

        keys = keys * key_mask.unsqueeze(-1).unsqueeze(-1)
        queries = queries * query_mask.unsqueeze(-1).unsqueeze(-1)

        permutation = self._permutation[:query_len, :, :].unsqueeze(0).expand_as(queries)
        queries = torch.gather(queries, -1, permutation)

        permutation = self._permutation[:key_len, :, :].unsqueeze(0).expand_as(keys)
        keys = torch.gather(keys, -1, permutation)

        kv_matrix = torch.einsum('blhd,blhm->bhmd', keys, values)

        normalizer = 1.0 / (torch.einsum('blhd,bhd->blh', queries, keys.sum(dim=1)) + self.eps)

        out = torch.einsum('blhd,bhmd,blh->blhm', queries, kv_matrix, normalizer)

        out = out.reshape(batch_size, query_len, -1)

        return self._out_proj(out)

    def _generate_permutation_sequence(self, n_heads, feature_map_dim, max_seq_length, seed):

        rng = torch.Generator().manual_seed(seed)

        perm = [torch.randperm(feature_map_dim, generator=rng) for _ in range(n_heads)]
        perm = torch.stack(perm, dim=0)

        expanded_perm = [torch.arange(feature_map_dim).unsqueeze(0).expand(n_heads, -1)]

        for _ in range(max_seq_length - 1):
            prev = expanded_perm[-1]
            expanded_perm.append(torch.gather(prev, 1, perm))

        return torch.stack(expanded_perm)
