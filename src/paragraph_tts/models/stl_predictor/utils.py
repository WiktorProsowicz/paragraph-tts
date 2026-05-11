"""Contains utilities used by the STL Predictor model."""

from typing import TypedDict
from typing import Literal

import torch
from torch_geometric.data import HeteroData

from paragraph_tts.utils import neural as neural_utils


def moe_load_balancing_loss(router_logits: torch.Tensor,
                            topk_indices: torch.Tensor) -> torch.Tensor:
    """Calculates load balancing loss for a MoE layer."""

    n_experts = router_logits.size(-1)

    load = moe_expert_usage(router_logits, topk_indices)
    importance = torch.softmax(router_logits, dim=-1).mean(dim=0)

    return (load * importance).sum() * n_experts


def moe_router_z_loss(router_logits: torch.Tensor) -> torch.Tensor:
    """Calculates router Z loss for a MoE layer."""

    log_z = torch.logsumexp(router_logits, dim=-1)
    return torch.mean(log_z ** 2)


def moe_expert_usage(router_logits: torch.Tensor, topk_indices: torch.Tensor) -> torch.Tensor:
    """Calculates the usage of each expert in a MoE layer."""

    expert_mask = torch.zeros_like(router_logits, dtype=torch.float32)
    expert_mask.scatter_(dim=-1, index=topk_indices, value=1.0)
    usage = expert_mask.sum(dim=0) / router_logits.size(0)

    return usage


class STLPredictorOutput(TypedDict):
    """Output of the forward pass through the STL predictor."""
    wsv_logits: torch.Tensor
    gst_logits: torch.Tensor
    topk_indices: dict[str, list[torch.Tensor]]
    router_logits: dict[str, list[torch.Tensor]]
    final_wsv_weights: torch.Tensor
    final_gst_weights: torch.Tensor
    wsv_cl_logits: torch.Tensor | None


class STLPredictorLoss(torch.nn.Module):
    """Calculates losses w.r.t. the STL predictor's output."""

    def __init__(self,
                 loss_weights: dict[str, float],
                 loss_weight_decays: dict[str, float],
                 model_output_mode: Literal['logits', 'weights', 'classification-plus-weights'],
                 wsv_cl_pos_weight: float | None
                 ) -> None:

        super().__init__()

        self._loss_weights = loss_weights
        self._loss_weight_decays = loss_weight_decays
        self._model_output_mode = model_output_mode

        if self._model_output_mode == 'classification-plus-weights':
            assert wsv_cl_pos_weight is not None

        self._wsv_cl_pos_weight = torch.tensor(wsv_cl_pos_weight)

    def forward(self,
                predictions: STLPredictorOutput,
                batch_graph: HeteroData,
                epoch: int) -> dict[str, torch.Tensor]:
        """Computes loss components and total loss for the STL predictor."""

        chosen_gst_logits = predictions['gst_logits'][batch_graph.has_gst_mask]
        chosen_wsv_logits = predictions['wsv_logits'][batch_graph.has_wsv_mask]

        loss_components = {}

        if self._model_output_mode == 'logits':
            loss_components['gst_pred_loss'] = torch.nn.functional.kl_div(
                torch.log_softmax(chosen_gst_logits, dim=-1),
                batch_graph.gst_weights,
                reduction='batchmean')
            loss_components['wsv_pred_loss'] = torch.nn.functional.kl_div(
                torch.log_softmax(chosen_wsv_logits, dim=-1),
                batch_graph.wsv_weights,
                reduction='batchmean')

        elif self._model_output_mode == 'weights':
            loss_components['gst_pred_loss'] = torch.nn.functional.l1_loss(chosen_gst_logits,
                                                                           batch_graph.gst_weights)
            loss_components['wsv_pred_loss'] = torch.nn.functional.l1_loss(chosen_wsv_logits,
                                                                           batch_graph.wsv_weights)

        else:
            loss_components['gst_pred_loss'] = torch.nn.functional.kl_div(
                torch.log_softmax(chosen_gst_logits, dim=-1),
                batch_graph.gst_weights,
                reduction='batchmean'
            )

            pos_wsv_mask = batch_graph.wsv_weights > 1e-5

            loss_components['wsv_cl_loss'] = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions['wsv_cl_logits'][batch_graph.has_wsv_mask],
                pos_wsv_mask.float(),
                pos_weight=self._wsv_cl_pos_weight
            )
            loss_components['wsv_pred_loss'] = torch.nn.functional.l1_loss(
                chosen_wsv_logits[pos_wsv_mask],
                batch_graph.wsv_weights[pos_wsv_mask]
            )

        load_balancing_losses = {}
        z_losses = {}

        for node_type in ('word_emb', 'global_emb'):

            for block_idx, (topk_indices, router_logits) in enumerate(
                zip(predictions['topk_indices'][node_type],
                    predictions['router_logits'][node_type])
            ):

                load_balancing_losses[f'moe_lb_loss/{node_type}_block_{block_idx}'] = (
                    moe_load_balancing_loss(router_logits, topk_indices)
                )

                z_losses[f'moe_z_loss/{node_type}_block_{block_idx}'] = (
                    moe_router_z_loss(router_logits)
                )

        loss_weights = {
            name: neural_utils.calc_decayed_loss_weight(self._loss_weights[name],
                                                        self._loss_weight_decays[name],
                                                        epoch)
            for name in self._loss_weights
        }

        loss_components['moe_lb_loss'] = sum(load_balancing_losses.values())
        loss_components['moe_z_loss'] = sum(z_losses.values())

        total_loss = sum(loss * loss_weights[name] for name, loss in loss_components.items())

        return {
            'total_loss': total_loss,
            **loss_components,
            **load_balancing_losses,
            **z_losses
        }
