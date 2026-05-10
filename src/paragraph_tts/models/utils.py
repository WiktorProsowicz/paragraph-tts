"""Contains utilities used by trainable models."""
import logging
import sys
from typing import TypedDict
from typing import Dict
from typing import Literal

import torch
from torch_geometric.data import HeteroData

from comp_trans_tts.model import loss as ctt_loss
from paragraph_tts.utils import neural as neural_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def calc_decayed_loss_weight(initial_weight: float, decay_rate: float, epoch: int) -> float:
    """Calculates decayed loss weight based on initial weight, decay rate and current epoch."""
    return initial_weight * (decay_rate ** epoch)


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


class AcousticModelLoss(torch.nn.Module):
    """Calculates losses w.r.t. the acoustic model's output."""

    def __init__(self,
                 loss_weights: Dict[str, float],
                 loss_weight_decays: Dict[str, float],
                 wsv_entropy_loss_start_epoch: int,
                 gst_entropy_loss_start_epoch: int
                 ) -> None:

        super().__init__()

        self._loss_weights = loss_weights
        self._loss_weight_decays = loss_weight_decays
        self._wsv_entropy_loss_start_epoch = wsv_entropy_loss_start_epoch
        self._gst_entropy_loss_start_epoch = gst_entropy_loss_start_epoch

    def forward(self,
                model_output: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor],
                epoch: int) -> dict[str, torch.Tensor]:
        """Computes loss components and total loss."""

        losses: dict[str, torch.Tensor] = {}

        mel_loss = torch.nn.MSELoss(reduction='none')(model_output['pred_mel_spec'],
                                                      batch['input_spec'])

        spec_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])
        spec_mask = spec_mask.unsqueeze(1).expand_as(mel_loss)

        losses['mel_loss'] = (mel_loss * spec_mask).sum() / spec_mask.sum()

        var_adap_losses = self._calculate_variance_adaptor_losses(model_output, batch)
        losses.update(var_adap_losses)

        if 'wsv_weights' in model_output:
            wsv_losses = self._calculate_wsv_losses(model_output, batch, epoch)
            losses.update(wsv_losses)

        if 'gst_weights' in model_output:
            gst_losses = self._calculate_gst_losses(model_output)
            losses.update(gst_losses)

        return {'total_loss': self._calculate_total_loss(losses, epoch), **losses}

    def get_current_loss_weights(self, epoch: int) -> dict[str, float]:
        """Calculates current loss weights based on initial weights and decay rates."""

        weight_decay_epochs = {loss_name: epoch for loss_name in self._loss_weights}

        if 'wsv_entropy_loss' in self._loss_weights:
            weight_decay_epochs['wsv_entropy_loss'] = epoch - self._wsv_entropy_loss_start_epoch

        if 'gst_entropy_loss' in self._loss_weights:
            weight_decay_epochs['gst_entropy_loss'] = epoch - self._gst_entropy_loss_start_epoch

        return {loss_name: calc_decayed_loss_weight(self._loss_weights[loss_name],
                                                    self._loss_weight_decays[loss_name],
                                                    weight_decay_epochs[loss_name])
                for loss_name in self._loss_weights}

    def _calculate_total_loss(self, losses: dict[str, torch.Tensor], epoch: int) -> torch.Tensor:
        """Calculates total loss as a weighted sum of individual losses."""

        current_loss_weights = self.get_current_loss_weights(epoch)

        return sum(loss * current_loss_weights[loss_name]
                   for loss_name, loss in losses.items())

    def _calculate_variance_adaptor_losses(self,
                                           model_output: dict[str, torch.Tensor],
                                           batch: dict[str, torch.Tensor]
                                           ) -> dict[str, torch.Tensor]:
        """Calculates losses for prosody and duration modelling outputs."""

        losses: dict[str, torch.Tensor] = {}

        prosody_mask = neural_utils.binary_mask_from_lengths(batch['prosody_features_length'])

        pitch_pred_loss = torch.nn.MSELoss(reduction='none')(
            model_output['predicted_pitch'],
            model_output['target_pitch'].detach()
        )
        losses['pitch_pred_loss'] = (pitch_pred_loss * prosody_mask).sum() / prosody_mask.sum()

        energy_pred_loss = torch.nn.MSELoss(reduction='none')(
            model_output['predicted_energy'],
            model_output['target_energy'].detach()
        )
        losses['energy_pred_loss'] = (energy_pred_loss * prosody_mask).sum() / prosody_mask.sum()

        duration_mask = neural_utils.binary_mask_from_lengths(batch['input_phonemes_length'])

        duration_loss = torch.nn.MSELoss(reduction='none')(
            model_output['predicted_durations'],
            batch['explicit_durations'].to(torch.float32).detach()
        )
        losses['duration_pred_loss'] = (duration_loss * duration_mask).sum() / duration_mask.sum()

        return losses

    def _calculate_wsv_losses(self,
                              model_output: dict[str, torch.Tensor],
                              batch: dict[str, torch.Tensor],
                              epoch: int
                              ) -> dict[str, torch.Tensor]:
        """Calculates losses for Word Level Style Variation outputs."""

        losses: dict[str, torch.Tensor] = {}

        wsv_mask = neural_utils.binary_mask_from_lengths(batch['input_word_emb_length'])

        losses['wsv_diversity_loss'] = ctt_loss.gst_diversity_loss(model_output['wsv_weights'],
                                                                   wsv_mask)

        if epoch >= self._wsv_entropy_loss_start_epoch:
            losses['wsv_entropy_loss'] = ctt_loss.gst_entropy_loss(model_output['wsv_weights'],
                                                                   wsv_mask)

        return losses

    def _calculate_gst_losses(self,
                              model_output: dict[str, torch.Tensor]
                              ) -> dict[str, torch.Tensor]:
        """Calculates losses for Global Style Token outputs."""

        losses: dict[str, torch.Tensor] = {}

        if self._gst_entropy_loss_start_epoch <= 0:
            losses['gst_entropy_loss'] = ctt_loss.gst_entropy_loss(model_output['gst_weights'])

        return losses


class AcousticModelMetrics(torch.nn.Module):
    """Calculates metrics w.r.t. the acoustic model's output."""

    def forward(self,
                model_output: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Computes metrics only."""

        metrics: dict[str, torch.Tensor] = {}

        mel_mae = torch.nn.L1Loss(reduction='none')(model_output['pred_mel_spec'],
                                                    batch['input_spec'])

        spec_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])
        spec_mask = spec_mask.unsqueeze(1).expand_as(mel_mae)

        metrics['mel_mae'] = (mel_mae * spec_mask).sum() / spec_mask.sum()

        if 'wsv_weights' in model_output:
            wsv_metrics = self._calculate_wsv_metrics(model_output, batch)
            metrics.update(wsv_metrics)

        if 'gst_weights' in model_output:
            gst_metrics = self._calculate_gst_metrics(model_output)
            metrics.update(gst_metrics)

        return metrics

    def _calculate_wsv_metrics(self,
                               model_output: dict[str, torch.Tensor],
                               batch: dict[str, torch.Tensor]
                               ) -> dict[str, torch.Tensor]:
        """Calculates metrics for Word Level Style Variation outputs."""

        metrics: dict[str, torch.Tensor] = {}

        wsv_mask = neural_utils.binary_mask_from_lengths(batch['input_word_emb_length'])

        metrics['wsv_distance_from_uniform'] = ctt_loss.gst_distance_from_uniform(
            model_output['wsv_weights'], wsv_mask)

        metrics['wsv_mean_top_weight'] = ctt_loss.gst_mean_top_weight(
            model_output['wsv_weights'], wsv_mask)

        metrics['wsv_top_3_mass'] = ctt_loss.gst_top_k_mass(
            model_output['wsv_weights'], k=3, wsv_mask=wsv_mask)

        metrics['wsv_top_7_mass'] = ctt_loss.gst_top_k_mass(
            model_output['wsv_weights'], k=7, wsv_mask=wsv_mask)

        return metrics

    def _calculate_gst_metrics(self,
                               model_output: dict[str, torch.Tensor]
                               ) -> dict[str, torch.Tensor]:
        """Calculates metrics for Global Style Token outputs."""

        metrics: dict[str, torch.Tensor] = {}

        metrics['gst_distance_from_uniform'] = ctt_loss.gst_distance_from_uniform(
            model_output['gst_weights']
        )

        metrics['gst_mean_top_weight'] = ctt_loss.gst_mean_top_weight(
            model_output['gst_weights']
        )

        metrics['gst_top_2_mass'] = ctt_loss.gst_top_k_mass(
            model_output['gst_weights'], k=2)

        metrics['gst_top_4_mass'] = ctt_loss.gst_top_k_mass(
            model_output['gst_weights'], k=4)

        return metrics


class STLPredictorOutput(TypedDict):
    """Output of the forward pass through the STL predictor."""
    wsv_logits: torch.Tensor
    gst_logits: torch.Tensor
    topk_indices: dict[str, list[torch.Tensor]]
    router_logits: dict[str, list[torch.Tensor]]


class STLPredictorLoss(torch.nn.Module):
    """Calculates losses w.r.t. the STL predictor's output."""

    def __init__(self,
                 loss_weights: dict[str, float],
                 loss_weight_decays: dict[str, float],
                 preds_loss_mode: Literal['kl-divergence', 'mse']
                 ) -> None:

        super().__init__()

        self._loss_weights = loss_weights
        self._loss_weight_decays = loss_weight_decays
        self._preds_loss_mode = preds_loss_mode

        if preds_loss_mode == 'kl-divergence':
            self._gst_loss = torch.nn.KLDivLoss(reduction='batchmean')
            self._wsv_loss = torch.nn.KLDivLoss(reduction='batchmean')

        elif preds_loss_mode == 'mae':
            self._gst_loss = torch.nn.L1Loss()
            self._wsv_loss = torch.nn.L1Loss()

        else:
            _logger().critical('Invalid preds_loss_mode: %s.', preds_loss_mode)
            sys.exit(1)

    def forward(self,
                predictions: STLPredictorOutput,
                batch_graph: HeteroData,
                epoch: int) -> dict[str, torch.Tensor]:
        """Computes loss components and total loss for the STL predictor."""

        chosen_gst_logits = predictions['gst_logits'][batch_graph.has_gst_mask]
        chosen_wsv_logits = predictions['wsv_logits'][batch_graph.has_wsv_mask]

        if self._preds_loss_mode == 'kl-divergence':
            gst_loss = self._gst_loss(torch.log_softmax(chosen_gst_logits, dim=-1),
                                      batch_graph.gst_weights)
            wsv_loss = self._wsv_loss(torch.log_softmax(chosen_wsv_logits, dim=-1),
                                      batch_graph.wsv_weights)

        else:
            gst_loss = self._gst_loss(chosen_gst_logits,
                                      batch_graph.gst_weights)
            wsv_loss = self._wsv_loss(chosen_wsv_logits,
                                      batch_graph.wsv_weights)

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
            name: calc_decayed_loss_weight(self._loss_weights[name],
                                           self._loss_weight_decays[name],
                                           epoch)
            for name in self._loss_weights
        }

        loss_components = {
            'gst_pred_loss': gst_loss,
            'wsv_pred_loss': wsv_loss,
            'moe_lb_loss': sum(load_balancing_losses.values()),
            'moe_z_loss': sum(z_losses.values())
        }

        total_loss = sum(loss * loss_weights[name] for name, loss in loss_components.items())

        return {
            'total_loss': total_loss,
            'gst_pred_loss': gst_loss,
            'wsv_pred_loss': wsv_loss,
            **load_balancing_losses,
            **z_losses
        }
