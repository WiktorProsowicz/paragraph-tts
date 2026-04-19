"""Contains utilities used by trainable models."""
import logging
import sys
from typing import Any
from typing import Dict
from typing import Iterator

import torch

from paragraph_tts.utils import neural as neural_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def optimizer_from_cfg(optimizer_cfg: Dict[str, Any],
                       parameters: Iterator[torch.nn.Parameter]) -> torch.optim.Optimizer:
    """Creates optimizer from configuration dictionary."""

    opt_name, opt_params = optimizer_cfg['name'], optimizer_cfg['params']

    if opt_name == 'adam':
        return torch.optim.Adam(parameters, **opt_params)

    _logger().critical('Unsupported optimizer type: %s', opt_name)
    sys.exit(1)


class AcousticModelLoss(torch.nn.Module):
    """Loss function incorporating all loss components for the acoustic model."""

    def __init__(self,
                 loss_weights: Dict[str, float]):

        super().__init__()

        self._loss_weights = loss_weights

    def forward(self,
                model_output: dict[str, torch.Tensor],
                batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Computes the loss components and the total loss."""

        mel_loss = torch.nn.MSELoss(reduction='none')(model_output['pred_mel_spec'],
                                                      batch['input_spec'])

        spec_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])
        spec_mask = spec_mask.unsqueeze(1).expand_as(mel_loss)

        mel_loss = (mel_loss * spec_mask).sum() / spec_mask.sum()

        losses = {
            'mel_loss': mel_loss
        }

        teacher_forcing_outputs = (
            'predicted_durations', 'target_pitch', 'target_energy'
        )

        if any(el in model_output for el in teacher_forcing_outputs):

            assert all(el in model_output for el in teacher_forcing_outputs)

            prosody_mask = neural_utils.binary_mask_from_lengths(batch['prosody_features_length'])

            pitch_pred_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_pitch'],
                model_output['target_pitch'].detach()
            )
            pitch_pred_loss = (pitch_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['pitch_pred_loss'] = pitch_pred_loss

            energy_pred_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_energy'],
                model_output['target_energy'].detach()
            )
            energy_pred_loss = (energy_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['energy_pred_loss'] = energy_pred_loss

            duration_mask = neural_utils.binary_mask_from_lengths(batch['input_phonemes_length'])

            duration_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_durations'],
                batch['explicit_durations'].to(torch.float32).detach()
            )
            duration_loss = (duration_loss * duration_mask).sum() / duration_mask.sum()
            losses['duration_pred_loss'] = duration_loss

        return {
            'total_loss': sum(loss * self._loss_weights[loss_name]
                              for loss_name, loss
                              in losses.items()),
            **losses
        }
