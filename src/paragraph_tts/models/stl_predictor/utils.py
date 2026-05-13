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
                 wsv_cl_pos_weight: float | None,
                 wsv_cl_eps: float | None) -> None:

        super().__init__()

        self._loss_weights = loss_weights
        self._loss_weight_decays = loss_weight_decays
        self._model_output_mode = model_output_mode

        if self._model_output_mode == 'classification-plus-weights':
            assert wsv_cl_pos_weight is not None
            assert wsv_cl_eps is not None

            self._wsv_cl_pos_weight = torch.tensor(wsv_cl_pos_weight)
            self._wsv_cl_eps = wsv_cl_eps

        else:
            self._wsv_cl_pos_weight = None
            self._wsv_cl_eps = None

    def forward(self,
                predictions: STLPredictorOutput,
                batch_graph: HeteroData,
                epoch: int) -> dict[str, torch.Tensor]:
        """Computes loss components and total loss for the STL predictor."""

        loss_components = {
            **self._calc_gst_losses(predictions, batch_graph),
            **self._calc_wsv_losses(predictions, batch_graph),
            **self._calc_moe_losses(predictions)
        }

        loss_weights = {
            name: neural_utils.calc_decayed_loss_weight(self._loss_weights[name],
                                                        self._loss_weight_decays[name],
                                                        epoch)
            for name in self._loss_weights
        }

        total_loss = sum(loss * loss_weights[name] for name, loss in loss_components.items())

        return {
            'total_loss': total_loss,
            **loss_components
        }

    def _calc_gst_losses(self,
                         predictions: STLPredictorOutput,
                         batch_graph: HeteroData) -> dict[str, torch.Tensor]:

        chosen_gst_logits = predictions['gst_logits'][batch_graph.has_gst_mask]

        if self._model_output_mode == 'weights':
            gst_pred_loss = torch.nn.functional.l1_loss(chosen_gst_logits, batch_graph.gst_weights)

        else:
            gst_pred_loss = torch.nn.functional.kl_div(torch.log_softmax(chosen_gst_logits, dim=-1),
                                                       batch_graph.gst_weights,
                                                       reduction='batchmean')

        return {
            'gst_pred_loss': gst_pred_loss
        }

    def _calc_wsv_losses(self,
                         predictions: STLPredictorOutput,
                         batch_graph: HeteroData) -> dict[str, torch.Tensor]:

        losses: dict[str, torch.Tensor] = {}

        chosen_wsv_logits = predictions['wsv_logits'][batch_graph.has_wsv_mask]

        if self._model_output_mode == 'weights':

            losses['wsv_pred_loss'] = torch.nn.functional.mse_loss(chosen_wsv_logits,
                                                                   batch_graph.wsv_weights)

        elif self._model_output_mode == 'classification-plus-weights':

            pos_wsv_mask = batch_graph.wsv_weights > self._wsv_cl_eps

            losses['wsv_pred_loss'] = torch.nn.functional.mse_loss(
                torch.relu(chosen_wsv_logits)[pos_wsv_mask],
                batch_graph.wsv_weights[pos_wsv_mask]
            )

            losses['wsv_cl_loss'] = torch.nn.functional.binary_cross_entropy_with_logits(
                predictions['wsv_cl_logits'][batch_graph.has_wsv_mask],
                pos_wsv_mask.float(),
                pos_weight=self._wsv_cl_pos_weight
            )

        elif self._model_output_mode == 'logits':

            losses['wsv_pred_loss'] = torch.nn.functional.kl_div(
                torch.log_softmax(chosen_wsv_logits, dim=-1),
                batch_graph.wsv_weights,
                reduction='batchmean'
            )

        return losses

    def _calc_moe_losses(self, predictions: STLPredictorOutput) -> dict[str, torch.Tensor]:

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

        return {
            'moe_lb_loss': sum(load_balancing_losses.values()),
            'moe_z_loss': sum(z_losses.values())
        }


class STLPredictorMetrics(torch.nn.Module):
    """Calculates metrics w.r.t. the STL predictor's output."""

    def __init__(self, wsv_cl_eps: float | None):

        super().__init__()

        self._wsv_cl_eps = wsv_cl_eps

    def forward(self,
                predictions: STLPredictorOutput,
                batch_graph: HeteroData,
                include_moe_metrics: bool) -> dict[str, torch.Tensor]:
        """Computes metrics for the STL predictor."""

        chosen_gst_weights = predictions['final_gst_weights'][batch_graph.has_gst_mask]
        chosen_wsv_weights = predictions['final_wsv_weights'][batch_graph.has_wsv_mask]

        gst_pred_ae = torch.abs(chosen_gst_weights - batch_graph.gst_weights)
        wsv_pred_ae = torch.abs(chosen_wsv_weights - batch_graph.wsv_weights)

        metrics = {
            'gst_pred_mae': gst_pred_ae.mean(),
            'wsv_pred_mae': wsv_pred_ae.mean(),
            'gst_pred_mae_weighted': (gst_pred_ae * batch_graph.gst_weights).sum(-1).mean(),
            'wsv_pred_mae_weighted': (wsv_pred_ae * batch_graph.wsv_weights).sum(-1).mean()
        }

        if include_moe_metrics:
            metrics.update(self._calc_moe_metrics(predictions))

        if 'wsv_cl_logits' in predictions:

            assert self._wsv_cl_eps is not None
            pos_wsv_mask = batch_graph.wsv_weights > self._wsv_cl_eps
            wsv_cl_probs = torch.sigmoid(predictions['wsv_cl_logits'][batch_graph.has_wsv_mask])

            metrics.update(self._calc_wsv_cl_metrics(wsv_cl_probs > 0.5,
                                                     pos_wsv_mask))

        return metrics

    def _calc_wsv_cl_metrics(self,
                             wsv_cl_preds: torch.Tensor,
                             wsv_cl_targets: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates metrics related to WSV classification."""

        tp = ((wsv_cl_preds == 1) & (wsv_cl_targets == 1)).sum()
        tn = ((wsv_cl_preds == 0) & (wsv_cl_targets == 0)).sum()
        fp = ((wsv_cl_preds == 1) & (wsv_cl_targets == 0)).sum()
        fn = ((wsv_cl_preds == 0) & (wsv_cl_targets == 1)).sum()

        return {
            'wsv_cl_accuracy': (tp + tn) / (tp + tn + fp + fn + 1e-8),
            'wsv_cl_precision_pos': tp / (tp + fp + 1e-8),
            'wsv_cl_recall_pos': tp / (tp + fn + 1e-8),
            'wsv_cl_precision_neg': tn / (tn + fn + 1e-8),
            'wsv_cl_recall_neg': tn / (tn + fp + 1e-8)
        }

    def _calc_moe_metrics(self, predictions: STLPredictorOutput) -> dict[str, torch.Tensor]:
        """Calculates MoE-specific metrics."""

        moe_metrics = {}
        expert_usages = {}

        for node_type in ('word_emb', 'global_emb'):

            for block_idx, (topk_indices, router_logits) in enumerate(
                zip(predictions['topk_indices'][node_type],
                    predictions['router_logits'][node_type])
            ):

                expert_usages[f'{node_type}/block_{block_idx}'] = (
                    moe_expert_usage(router_logits, topk_indices)
                )

        for key, expert_usage in expert_usages.items():
            moe_metrics[f'experts_usage/{key}_max'] = expert_usage.max()
            moe_metrics[f'experts_usage/{key}_min'] = expert_usage.min()

        return moe_metrics
