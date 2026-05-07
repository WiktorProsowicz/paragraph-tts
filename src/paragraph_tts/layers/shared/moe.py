"""Contains implementation of Mixture of Experts (MoE) based projection layer."""


import torch


class SwiGLUFFN(torch.nn.Module):
    """A feedforward network (FFN) layer that uses the SwiGLU activation function."""

    def __init__(self, hidden_dim: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()

        inner_dim = int((2 / 3) * ffn_dim)

        self._w_gate = torch.nn.Linear(hidden_dim, inner_dim, bias=False)
        self._w_up = torch.nn.Linear(hidden_dim, inner_dim, bias=False)
        self._w_down = torch.nn.Linear(inner_dim, hidden_dim, bias=False)
        self._dropout = torch.nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Projects the input through the SwiGLU FFN layer and returns the output."""

        gate = torch.nn.functional.silu(self._w_gate(x))

        x = self._dropout(gate * self._w_up(x))

        return self._w_down(x)


class MoEProjection(torch.nn.Module):
    """A Mixture of Experts (MoE) based projection layer."""

    def __init__(self,
                 input_dim: int,
                 proj_dim: int,
                 n_experts: int,
                 top_k: int,
                 dropout: float
                 ) -> None:

        super().__init__()

        self._top_k = top_k
        self._num_experts = n_experts

        self._router = torch.nn.Linear(input_dim, n_experts)

        self._experts = torch.nn.ModuleList([SwiGLUFFN(input_dim, proj_dim, dropout)
                                             for _ in range(n_experts)])

    def forward(self, x):
        """Processes the input sequence through the MoE layer and returns the projected output.

        Returns:
            Tensor of shape (batch_size, seq_len, input_dim) containing the projected output.
        """

        router_logits = self._router(x)

        topk_logits, topk_indices = torch.topk(router_logits, k=self._top_k, dim=-1)

        routing_weights = torch.softmax(topk_logits, dim=-1)

        output = torch.zeros_like(x)

        for k in range(self._top_k):

            expert_indices = topk_indices[:, k]
            expert_weights = routing_weights[:, k]

            for expert_id, expert in enumerate(self._experts):

                mask = expert_indices == expert_id

                if mask.any():
                    output[mask] += (expert(x[mask]) * expert_weights[mask].unsqueeze(-1))

        return output, topk_indices, router_logits
