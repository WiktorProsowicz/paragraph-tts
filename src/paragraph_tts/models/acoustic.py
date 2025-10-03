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
            speaker_embedding=inputs['spk_embedding'],
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
            mel_length=mel_length
        )

        return {
            'pred_mel_spec': pred_mel_spec,
            **var_adaptor_output
        }

    def training_step(self,
                      batch: Dict[str, torch.Tensor],
                      batch_idx: int):

        pass

    def _calculate_losses(self,
                          model_output: Dict[str, torch.Tensor],
                          batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Calculates losses from model output and target features."""

        mel_loss = torch.nn.MSELoss(reduction='none')(model_output['pred_mel_spec'],
                                                      batch['input_spec'])

        spec_mask = neural_utils.binary_mask_from_lengths(batch['input_spec_length'])
        spec_mask = spec_mask.unsqueeze(1).expand_as(mel_loss)

        mel_loss = (mel_loss * spec_mask).sum() / spec_mask.sum()
