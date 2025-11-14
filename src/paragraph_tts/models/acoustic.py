# -*- coding: utf-8 -*-
"""Contains definition of acoustic model training/inference pipelines."""
import logging
from typing import Any
from typing import Dict
from typing import Optional

import lightning.pytorch as pl
import torch
from comp_trans_tts.model import modules as ctt_modules
from speechbrain.inference.vocoders import HIFIGAN

from paragraph_tts.layers import acoustic as acoustic_layers
from paragraph_tts.models import utils as model_utils
from paragraph_tts.utils import inference as inference_utils
from paragraph_tts.utils import neural as neural_utils
from paragraph_tts.utils import visualization as viz_utils


def _logger():
    return logging.getLogger(__name__)


class AcousticModel(pl.LightningModule):
    """Predicts mel-spectrogram from input textual features.

    The acoustic model supports explicit prosody
    """

    def __init__(self,
                 model_cfg: Dict[str, Any],
                 optim_cfg: Dict[str, Any],
                 train_cfg: Dict[str, Any],
                 data_cfg: Dict[str, Any]):

        super().__init__()

        self._encoder = acoustic_layers.encoder.Encoder(
            **model_cfg['encoder']
        )

        self._context_encoder = acoustic_layers.context_encoder.ContextEncoder(
            **model_cfg['context_encoder']
        )

        self._decoder = acoustic_layers.decoder.Decoder(
            **model_cfg['decoder']
        )

        self._var_adaptor = ctt_modules.VarianceAdaptor(
            **model_cfg['var_adaptor']
        )

        self._model_cfg = model_cfg
        self._optim_cfg = optim_cfg
        self._train_cfg = train_cfg
        self._data_cfg = data_cfg

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Sets up optimizer from config."""

        opt = model_utils.optimizer_from_cfg(self._optim_cfg,
                                             self.parameters())

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=self._optim_cfg['lr_decay']
        )

        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }

    def forward(self,  # pylint: disable=arguments-differ
                inputs: Dict[str, torch.Tensor],
                use_teacher_forcing: bool
                ) -> Dict[str, torch.Tensor]:
        """Performs forward pass of the model."""

        enc_output = self._encoder(
            inputs['input_phoneme_ids'],
            inputs['input_pos_tags'],
            inputs['input_ling_stats'],
            inputs['input_token_emb'],
            inputs['word_to_phoneme_indices'],
            inputs['sentence_pos'],
            inputs['spk_rate'],
            inputs['input_phonemes_length']
        )

        context_output = self._context_encoder(
            inputs['context_token_emb'],
            inputs['context_tokens_length'],
            inputs['context_pse'],
            inputs['context_pse_length'],
            enc_output,
            inputs['input_phonemes_length']
        )

        enc_output_enriched = enc_output + context_output

        forced_args: Dict[str, Optional[torch.Tensor]] = {
            'explicit_duration': None,
            'pitch_target': None,
            'energy_target': None
        }

        if use_teacher_forcing:

            forced_args = {
                'explicit_duration': inputs['explicit_durations'],
                'pitch_target': inputs['input_f0'],
                'energy_target': inputs['input_energy'],
            }

        var_adaptor_output = self._var_adaptor(
            phoneme_repr=enc_output_enriched,
            phoneme_lengths=inputs['input_phonemes_length'],
            pitch_possible_values=inputs['pitch_possible_values'],
            energy_possible_values=inputs['energy_possible_values'],
            speaker_embedding=inputs['spk_emb'],
            **forced_args
        )

        if use_teacher_forcing:
            mel_length = inputs['input_spec_length']

        else:
            ph_durations = inference_utils.sanitize_predicted_durations(
                var_adaptor_output['predicted_duration'],
                inputs['input_phonemes_length'])
            mel_length = ph_durations.sum(dim=1)

        pred_mel_spec = self._decoder(
            var_adaptor_output['output'],
            mel_length
        )

        return {
            'pred_mel_spec': pred_mel_spec,
            **var_adaptor_output
        }

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int) -> torch.Tensor:
        """Performs training step."""

        model_output = self.forward(batch,
                                    use_teacher_forcing=True)

        losses = self._calculate_losses(model_output, batch)

        logged_dict = {f'train/{k}': v.detach().item() for k, v in losses.items()}

        self.log_dict(logged_dict,
                      on_step=True,
                      on_epoch=False,
                      batch_size=self._data_cfg['batch_size'])

        if self._should_visualize(batch_idx, training=True):

            for sample_idx in range(min(10, self._data_cfg['batch_size'])):

                self._visualize_outputs(batch,
                                        model_output,
                                        None,
                                        'training',
                                        sample_idx)

        return sum(losses.values())  # type: ignore

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: Dict[str, torch.Tensor],
                        batch_idx: int) -> None:
        """Performs validation step."""

        model_output = self.forward(batch,
                                    use_teacher_forcing=True)

        losses = self._calculate_losses(model_output, batch)

        logged_dict = {f'val/{k}': v.detach().item() for k, v in losses.items()}

        self.log_dict(logged_dict,
                      on_step=False,
                      on_epoch=True,
                      batch_size=self._data_cfg['batch_size'])

        if self._should_visualize(batch_idx, training=False):

            hifi_gan = HIFIGAN.from_hparams(source='speechbrain/tts-hifigan-libritts-22050Hz',
                                            run_opts={'device': self.device})

            for sample_idx in range(min(10, self._data_cfg['batch_size'])):

                self._visualize_outputs(batch,
                                        model_output,
                                        hifi_gan,
                                        'teacher_forcing',
                                        sample_idx)

            model_output = self.forward(batch,
                                        use_teacher_forcing=False)

            for sample_idx in range(min(10, self._data_cfg['batch_size'])):

                self._visualize_outputs(batch,
                                        model_output,
                                        hifi_gan,
                                        'inference',
                                        sample_idx,)

    def _visualize_outputs(self,
                           batch: Dict[str, torch.Tensor],
                           model_output: Dict[str, torch.Tensor],
                           hifi_gan: Optional[HIFIGAN],
                           base_label: str,
                           sample_idx: int) -> None:
        """Visualizes model outputs (spectrograms, pitch/energy, output wav)."""

        tensorboard = self.loggers[1].experiment  # type: ignore

        _logger().debug('Visualizing outputs for sample %d (label=%s).',
                        sample_idx, base_label)

        spec_length = int(batch['input_spec_length'][sample_idx].item())

        fig = viz_utils.plot_spectrograms(
            model_output['pred_mel_spec'][sample_idx].detach()[:, :spec_length],
            batch['input_spec'][sample_idx].detach()[:, :spec_length])

        tensorboard.add_figure(f'{base_label}/spectrograms/{sample_idx}',
                               fig,
                               self.trainer.global_step)

        if 'target_pitch_quant' in model_output:

            cont_len = int(batch['input_spec_length'][sample_idx].item())

            fig = viz_utils.plot_contours(
                model_output['predicted_pitch'][sample_idx].detach()[:cont_len],
                model_output['target_pitch_quant'][sample_idx].detach()[:cont_len],
                'Pitch'
            )

            tensorboard.add_figure(f'{base_label}/pitch/{sample_idx}',
                                   fig,
                                   self.trainer.global_step)

        if 'target_energy_quant' in model_output:

            cont_len = int(batch['input_spec_length'][sample_idx].item())

            fig = viz_utils.plot_contours(
                model_output['predicted_energy'][sample_idx].detach()[:cont_len],
                model_output['target_energy_quant'][sample_idx].detach()[:cont_len],
                'Energy'
            )

            tensorboard.add_figure(f'{base_label}/energy/{sample_idx}',
                                   fig,
                                   self.trainer.global_step)

        if 'duration_rounded' in model_output:

            cont_len = int(batch['input_phonemes_length'][sample_idx].item())

            fig = viz_utils.plot_contours(
                model_output['predicted_duration'][sample_idx].detach()[:cont_len],
                model_output['duration_rounded'][sample_idx].detach()[:cont_len],
                'Duration'
            )

            tensorboard.add_figure(f'{base_label}/durations/{sample_idx}',
                                   fig,
                                   self.trainer.global_step)

        if hifi_gan is None:
            return

        if base_label != 'inference':
            mel_len = int(batch['input_spec_length'][sample_idx].item())

        else:
            san_dur = inference_utils.sanitize_predicted_durations(
                model_output['predicted_duration'],
                batch['input_phonemes_length'])[sample_idx]
            mel_len = int(san_dur.sum().item())

        wav = inference_utils.transform_mel_to_wav(
            model_output['pred_mel_spec'][sample_idx][:, :mel_len],
            hifi_gan.decode_batch,
            split_spec_by_silences=True
        )

        if wav is not None:
            tensorboard.add_audio(f'{base_label}/wav/{sample_idx}/generated',
                                  wav.squeeze(0),
                                  self.trainer.global_step,
                                  sample_rate=22050)

        mel_len = int(batch['input_spec_length'][sample_idx])

        wav = inference_utils.transform_mel_to_wav(
            batch['input_spec'][sample_idx][:, :mel_len],
            hifi_gan.decode_batch,
            split_spec_by_silences=True
        )

        if wav is not None:
            tensorboard.add_audio(f'{base_label}/wav/{sample_idx}/target',
                                  wav.squeeze(0),
                                  self.trainer.global_step,
                                  sample_rate=22050)

    def _should_visualize(self,
                          batch_idx: int,
                          training: bool) -> bool:
        """Decides whether to visualize outputs on the current train/val step."""

        if (self.trainer.current_epoch + 1) % self._train_cfg['visualize_every_n_epochs'] != 0:
            return False

        if not training and batch_idx != 0:
            return False

        if training:
            n_viz = self._train_cfg['visualize_n_times_during_training']
            viz_interval = self.trainer.num_training_batches // (n_viz + 1)
            proper_indices = [(i + 1) * viz_interval for i in range(n_viz)]

            if batch_idx not in proper_indices:
                return False

        return True

    def _calculate_losses(self,
                          model_output: Dict[str, torch.Tensor],
                          batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculates losses from model output and target features."""

        mel_loss = torch.nn.MSELoss(reduction='none')(model_output['pred_mel_spec'],
                                                      batch['input_spec'])

        spec_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])
        spec_mask = spec_mask.unsqueeze(1).expand_as(mel_loss)

        mel_loss = (mel_loss * spec_mask).sum() / spec_mask.sum()

        losses = {
            'mel_loss': mel_loss
        }

        teacher_forcing_outputs = (
            'duration_rounded', 'target_pitch_quant', 'target_energy_quant'
        )

        if any(el in model_output for el in teacher_forcing_outputs):

            assert all(el in model_output for el in teacher_forcing_outputs)

            prosody_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])

            pitch_pred_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_pitch'],
                model_output['target_pitch_quant'].detach()
            )
            pitch_pred_loss = (pitch_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['pitch_pred_loss'] = pitch_pred_loss

            energy_pred_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_energy'],
                model_output['target_energy_quant'].detach()
            )
            energy_pred_loss = (energy_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['energy_pred_loss'] = energy_pred_loss

            duration_mask = neural_utils.binary_mask_from_lengths(batch['input_phonemes_length'])

            duration_loss = torch.nn.MSELoss(reduction='none')(
                model_output['predicted_duration'],
                model_output['duration_rounded'].to(torch.float32).detach()
            )
            duration_loss = (duration_loss * duration_mask).sum() / duration_mask.sum()
            losses['duration_pred_loss'] = duration_loss

        for l_name in losses:
            losses[l_name] /= self._train_cfg['loss_est_max'][l_name]
            losses[l_name] *= self._train_cfg['loss_weights'][l_name]

        return losses
