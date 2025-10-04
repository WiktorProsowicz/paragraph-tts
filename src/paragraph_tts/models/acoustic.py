"""Contains definition of acoustic model training/inference pipelines."""

from typing import Dict, Any

import lightning.pytorch as pl
import torch
from comp_trans_tts.model import modules as ctt_modules
from comp_trans_tts.model import loss as ctt_loss

from paragraph_tts.layers import acoustic as acoustic_layers
from paragraph_tts.models import utils as model_utils
from paragraph_tts.utils import neural as neural_utils


class AcousticModel(pl.LightningModule):
    """Predicts mel-spectrogram from input textual features.

    The acoustic model supports explicit prosody
    """

    def __init__(self,
                 model_cfg: Dict[str, Any],
                 optim_cfg: Dict[str, Any],
                 train_cfg: Dict[str, Any]):

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

        self.save_hyperparameters()

    def configure_optimizers(self):
        """Sets up optimizer from config."""

        return model_utils.optimizer_from_cfg(self._optim_cfg,
                                              self.parameters())

    def forward(self,  # pylint: disable=arguments-differ
                inputs: Dict[str, torch.Tensor],
                use_teacher_forcing: bool,
                ) -> Dict[str, torch.Tensor]:
        """Performs forward pass of the model."""

        enc_output = self._encoder(
            inputs['input_phoneme_ids'],
            inputs['input_pos_tags'],
            inputs['input_ling_stats'],
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

        if use_teacher_forcing:

            binarize_alignment = False

            if self.trainer.global_step > self._train_cfg['binarize_alignment_start_step']:
                binarize_alignment = True

            forced_args = {
                'binarize_alignment': binarize_alignment,
                'mel': inputs['input_spec'],
                'mel_lengths': inputs['input_spec_length'],
                'pitch_target': inputs['input_f0'],
                'energy_target': inputs['input_energy'],
                'attn_prior': inputs['align_att_prior']
            }

        else:
            forced_args = {
                'binarize_alignment': None,
                'mel': None,
                'mel_lengths': None,
                'pitch_target': None,
                'energy_target': None,
                'attn_prior': None
            }

        var_adaptor_output = self._var_adaptor(
            phoneme_repr=enc_output_enriched,
            phoneme_lengths=inputs['input_phonemes_length'],
            phoneme_mask=neural_utils.binary_mask_from_lengths(inputs['input_phonemes_length']),
            pitch_possible_values=inputs['pitch_possible_values'],
            energy_possible_values=inputs['energy_possible_values'],
            speaker_embedding=inputs['spk_emb'],
            **forced_args
        )

        if use_teacher_forcing:
            mel_length = inputs['input_spec_length']

        else:
            predicted_dur = var_adaptor_output['predicted_durations']
            duration_rounded = torch.clamp(torch.round(predicted_dur), min=0)
            mel_length = duration_rounded.sum(dim=1).long()

        pred_mel_spec = self._decoder(
            var_adaptor_output['output'],
            mel_length
        )

        return {
            'pred_mel_spec': pred_mel_spec,
            **var_adaptor_output
        }

    def training_step(self, # pylint: disable=arguments-differ
                      batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs training step."""

        model_output = self.forward(batch, use_teacher_forcing=True)

        losses = self._calculate_losses(model_output, batch)

        for loss_name, loss_t in losses.items():
            self.log(f'train/{loss_name}',
                     loss_t.item(),
                     on_step=True,
                     on_epoch=False,
                     batch_size=self._train_cfg['batch_size'])

        return sum(losses.values())

    def validation_step(self, # pylint: disable=arguments-differ
                        batch: Dict[str, torch.Tensor]) -> None:
        """Performs validation step."""

        model_output = self.forward(batch, use_teacher_forcing=True)

        losses = self._calculate_losses(model_output, batch)

        for loss_name, loss_t in losses.items():
            self.log(f'val/{loss_name}',
                     loss_t.item(),
                     on_step=False,
                     on_epoch=True,
                     batch_size=self._train_cfg['batch_size'])

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
            'attn_soft', 'attn_hard', 'attn_logprob', 'duration_rounded',
            'target_pitch_quant', 'target_energy_quant'
        )

        if any(el in model_output for el in teacher_forcing_outputs):

            assert all(el in model_output for el in teacher_forcing_outputs)

            ctc_loss = ctt_loss.ForwardSumLoss()(attn_logprob=model_output['attn_logprob'],
                                                 in_lens=batch['input_phonemes_length'],
                                                 out_lens=batch['input_spec_length'])

            losses['ctc_loss'] = ctc_loss

            train_step = self.trainer.global_step

            if train_step > self._train_cfg['binarize_alignment_start_step']:
                bin_loss_weight = 0.0

            else:
                bin_warmup_steps = self._train_cfg['binarization_loss_warmup_steps']
                bin_loss_weight = min((train_step - bin_warmup_steps) / bin_warmup_steps, 1.0)

            bin_loss = ctt_loss.BinLoss()(hard_attention=model_output['attn_hard'],
                                          soft_attention=model_output['attn_soft'])
            bin_loss *= bin_loss_weight
            losses['bin_loss'] = bin_loss

            prosody_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])

            pitch_pred_loss = torch.nn.L1Loss(reduction='none')(
                model_output['predicted_pitch'],
                model_output['target_pitch_quant']
            )
            pitch_pred_loss = (pitch_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['pitch_pred_loss'] = pitch_pred_loss

            energy_pred_loss = torch.nn.L1Loss(reduction='none')(
                model_output['predicted_energy'],
                model_output['target_energy_quant']
            )
            energy_pred_loss = (energy_pred_loss * prosody_mask).sum() / prosody_mask.sum()
            losses['energy_pred_loss'] = energy_pred_loss

        return losses
