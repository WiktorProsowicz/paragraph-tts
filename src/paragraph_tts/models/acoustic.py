"""Contains definition of acoustic model training/inference pipelines."""
import logging
import pathlib
from typing import Annotated

import mlflow
import lightning.pytorch as pl
import pydantic
import torch
from comp_trans_tts.model import modules as ctt_modules
from torch_dev_utils.tts import visualization
from pydantic import Field
from speechbrain.inference.vocoders import HIFIGAN
import soundfile

from paragraph_tts.layers.acoustic import encoder as acoustic_encoder
from paragraph_tts.layers.acoustic import decoder as acoustic_decoder
from paragraph_tts.layers.acoustic import context_encoder as acoustic_context_encoder
from paragraph_tts.models import utils as model_utils
from paragraph_tts.utils import inference as inference_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ModelConfiguration(pydantic.BaseModel):
    """Configuration of model architecture components."""

    encoder: Annotated[acoustic_encoder.Encoder.Configuration, Field(
        description='Configuration of acoustic encoder.')]

    decoder: Annotated[acoustic_decoder.Decoder.Configuration, Field(
        description='Configuration of acoustic decoder.')]

    context_encoder: Annotated[acoustic_context_encoder.ContextEncoder.Configuration | None, Field(
        description=('Configuration of context encoder.'
                     'If none, context encoder is not used.'))]

    var_adaptor: Annotated[ctt_modules.VarianceAdaptor.Configuration, Field(
        description='Configuration of variance adaptor module.')]

    prosody_encoder: Annotated[ctt_modules.HierarchicalProsodyEncoder.Configuration | None, Field(
        description=('Configuration of hierarchical prosody encoder module.'
                     'If none, hierarchical prosody encoder is not used.'))]


class OptimizerConfiguration(pydantic.BaseModel):
    """Optimizer parameters."""

    lr: Annotated[float, Field(description='Learning rate.')]

    betas: Annotated[tuple[float, float], Field(
        description='Adam beta coefficients.')]

    eps: Annotated[float, Field(description='Adam epsilon value.')]

    weight_decay: Annotated[float, Field(description='Weight decay coefficient.')]

    lr_scheduler_gamma: Annotated[float, Field(
        description='Exponential decay factor of learning rate scheduler.')]


class TrainConfiguration(pydantic.BaseModel):
    """Training loop and objective configuration."""

    loss_weights: Annotated[dict[str, float], Field(
        description='Relative weights of different loss components used for optimization.')]

    loss_weight_decays: Annotated[dict[str, float], Field(
        description='Decay factors for loss weights.')]

    gradient_clip_val: Annotated[float, Field(
        description='Gradient clipping value used by trainer.')]

    accumulate_grad_batches: Annotated[int, Field(
        description='Number of batches to accumulate gradients over.')]

    stl_bin_init_temperature: Annotated[float, Field(
        description='Initial temperature for STL binarization in prosody encoder.')]

    stl_bin_temperature_decay: Annotated[float, Field(
        description='Decay factor for temperature of STL binarization in prosody encoder.')]

    stl_bin_loss_start_epoch: Annotated[int, Field(
        description='Epoch to start applying binarization loss for prosody encoder.')]

    stl_bin_hard_start_epoch: Annotated[int, Field(
        description='Epoch to start using hard binarization in prosody encoder.')]


class AcousticModel(pl.LightningModule):
    """Predicts mel-spectrogram from input textual features.

    The acoustic model supports explicit prosody
    """

    def __init__(self,
                 model_cfg: ModelConfiguration,
                 optim_cfg: OptimizerConfiguration,
                 train_cfg: TrainConfiguration,
                 vocoder: HIFIGAN) -> None:

        super().__init__()

        self._encoder = acoustic_encoder.Encoder(
            model_cfg.encoder
        )

        if model_cfg.context_encoder is not None:
            self._context_encoder = acoustic_context_encoder.ContextEncoder(
                model_cfg.context_encoder
            )
        else:
            self._context_encoder = None

        if model_cfg.prosody_encoder is not None:
            self._prosody_encoder = ctt_modules.HierarchicalProsodyEncoder(
                model_cfg.prosody_encoder
            )
        else:
            self._prosody_encoder = None

        self._decoder = acoustic_decoder.Decoder(
            model_cfg.decoder
        )
        self._var_adaptor = ctt_modules.VarianceAdaptor(
            model_cfg.var_adaptor
        )

        self._model_cfg = model_cfg
        self._optim_cfg = optim_cfg
        self._train_cfg = train_cfg
        self._vocoder = vocoder

        self.save_hyperparameters(logger=False, ignore=['vocoder'])

        self._visualize_n_batches = 3
        self._visualize_n_samples_per_batch = 3

        self._loss = model_utils.AcousticModelLoss(
            train_cfg.loss_weights,
            train_cfg.loss_weight_decays,
            train_cfg.stl_bin_loss_start_epoch
        )
        self._metrics = model_utils.AcousticModelMetrics()

    def configure_optimizers(self):  # type: ignore
        """Sets up optimizer from config."""

        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self._optim_cfg.lr,
            betas=self._optim_cfg.betas,
            eps=self._optim_cfg.eps,
            weight_decay=self._optim_cfg.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            opt, gamma=self._optim_cfg.lr_scheduler_gamma
        )

        return {
            'optimizer': opt,
            'lr_scheduler': scheduler
        }

    def forward(self,  # pylint: disable=arguments-differ
                inputs: dict[str, torch.Tensor],
                use_teacher_forcing: bool
                ) -> dict[str, torch.Tensor]:
        """Performs forward pass of the model."""

        enc_output = self._encoder(
            inputs['input_phoneme_ids'],
            inputs['input_pos_tags'],
            inputs['input_ling_stats'],
            inputs['input_word_emb'],
            inputs['word_to_phoneme_indices'],
            inputs['sentence_pos'],
            inputs['spk_rate'],
            inputs['input_phonemes_length']
        )

        if self._context_encoder is not None:

            context_output = self._context_encoder(
                inputs['context_token_emb'],
                inputs['context_tokens_length'],
                inputs['context_pse'],
                inputs['context_pse_length'],
                enc_output,
                inputs['input_phonemes_length']
            )

            enc_output = enc_output + context_output

        prosody_enc_outputs: dict[str, torch.Tensor] = {}

        if self._prosody_encoder is not None:

            if self.current_epoch < self._train_cfg.stl_bin_hard_start_epoch:
                bin_params = ctt_modules.StlBinarizationParams(
                    hard=False,
                    temperature=model_utils.calc_decayed_loss_weight(
                        self._train_cfg.stl_bin_init_temperature,
                        self._train_cfg.stl_bin_temperature_decay,
                        self.current_epoch
                    )
                )
            else:
                bin_params = ctt_modules.StlBinarizationParams(
                    hard=True,
                    temperature=model_utils.calc_decayed_loss_weight(
                        self._train_cfg.stl_bin_init_temperature,
                        self._train_cfg.stl_bin_temperature_decay,
                        self._train_cfg.stl_bin_hard_start_epoch
                    )
                )

            ((gst_emb, gst_weights), (wsv_emb, wsv_weights)) = self._prosody_encoder(
                spectrogram=inputs['input_spec'],
                spectrogram_length=inputs['input_spec_length'],
                spec_word_pool_matrix=inputs['spec_to_word_pool_matrix'],
                phoneme_ids=inputs['input_phoneme_ids'],
                linguistic_features=inputs['input_ling_stats'],
                phoneme_spec_indices=inputs['phone_to_spec_indices'],
                local_stl_binarization_params=bin_params)

            gst_emb = gst_emb.unsqueeze(1).expand_as(enc_output)
            wsv_emb = wsv_emb[torch.arange(enc_output.size(0)).unsqueeze(1),
                              inputs['word_to_phoneme_indices']]

            if bin_params.hard:
                gst_emb = gst_emb.detach()
                wsv_emb = wsv_emb.detach()
                gst_weights = gst_weights.detach()
                wsv_weights = wsv_weights.detach()

            prosody_enc_outputs['gst_weights'] = gst_weights
            prosody_enc_outputs['wsv_weights'] = wsv_weights

            enc_output = enc_output + gst_emb + wsv_emb

        forced_args: dict[str, torch.Tensor | None] = {
            'explicit_durations': None,
            'pitch_target': None,
            'energy_target': None
        }

        if use_teacher_forcing:

            forced_args = {
                'explicit_durations': inputs['explicit_durations'],
                'pitch_target': inputs['input_f0'],
                'energy_target': inputs['input_energy'],
            }

        var_adaptor_inputs = ctt_modules.VarianceAdaptor.ForwardInput(
            phoneme_repr=enc_output,
            phonemes_length=inputs['input_phonemes_length'],
            pitch_possible_values=inputs.get('pitch_possible_values'),
            energy_possible_values=inputs.get('energy_possible_values'),
            speaker_embedding=inputs['spk_emb'],
            **forced_args
        )

        var_adaptor_output = self._var_adaptor(var_adaptor_inputs)

        if use_teacher_forcing:
            mel_length = inputs['input_spec_length']

        else:
            mel_length = var_adaptor_output['predicted_durations'].sum(dim=1).long()

        pred_mel_spec = self._decoder(
            var_adaptor_output['output'],
            mel_length
        )

        return {
            'pred_mel_spec': pred_mel_spec,
            **var_adaptor_output,
            **prosody_enc_outputs
        }

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs training step."""

        model_output = self.forward(batch,
                                    use_teacher_forcing=True)

        losses = self._loss(model_output, batch, self.current_epoch)

        with torch.no_grad():
            metrics = self._metrics(model_output, batch)

        logged_dict = {f'train/{k}': v.detach().item() for k, v in {**losses, **metrics}.items()}

        self.log_dict(logged_dict,
                      on_step=True,
                      on_epoch=False,
                      batch_size=batch['input_phonemes_length'].size(0))

        return losses['total_loss']

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: dict[str, torch.Tensor],
                        batch_idx: int) -> None:
        """Performs validation step."""

        model_output = self(batch,
                            use_teacher_forcing=True)

        losses = self._loss(model_output, batch, self.current_epoch)
        metrics = self._metrics(model_output, batch)

        logged_dict = {f'val/{k}': v.detach().item() for k, v in {**losses, **metrics}.items()}

        self.log_dict(logged_dict,
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch['input_phonemes_length'].size(0))

        if batch_idx < self._visualize_n_batches:

            model_output_inference = self(batch,
                                          use_teacher_forcing=False)

            for sample_idx in range(min(self._visualize_n_samples_per_batch,
                                        batch['input_phonemes_length'].size(0))):

                sample = {k: v[sample_idx].cpu() for k, v in batch.items()}

                self._visualize_outputs(
                    sample,
                    {k: v[sample_idx].cpu() for k, v in model_output.items()},
                    self._vocoder,
                    save_target_wav=(self.current_epoch == 0),
                    output_dir=(pathlib.Path(mlflow.get_artifact_uri())
                                .joinpath('viz')
                                .joinpath(f'epoch_{self.current_epoch}')
                                .joinpath(f'batch_{batch_idx}')
                                .joinpath(f'sample_{sample_idx}')
                                .joinpath('teacher_forcing'))
                )

                self._visualize_outputs(
                    sample,
                    {k: v[sample_idx].cpu() for k, v in model_output_inference.items()},
                    self._vocoder,
                    save_target_wav=False,
                    output_dir=(pathlib.Path(mlflow.get_artifact_uri())
                                .joinpath('viz')
                                .joinpath(f'epoch_{self.current_epoch}')
                                .joinpath(f'batch_{batch_idx}')
                                .joinpath(f'sample_{sample_idx}')
                                .joinpath('inference'))
                )

    def _visualize_outputs(self,
                           sample: dict[str, torch.Tensor],
                           model_output: dict[str, torch.Tensor],
                           hifi_gan: HIFIGAN,
                           save_target_wav: bool,
                           output_dir: pathlib.Path) -> None:
        """Visualizes model outputs (spectrograms, pitch/energy, output wav)."""

        _logger().debug('Visualizing and saving outputs to %s.', output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        spec_length = int(sample['input_spec_length'].item())
        predicted_spec_length = int(model_output['predicted_durations'].sum().long().item())
        prosody_features_length = int(sample['prosody_features_length'].item())
        phonemes_length = int(sample['input_phonemes_length'].item())

        if prosody_features_length == spec_length:
            predicted_prosody_features_length = predicted_spec_length

        else:
            predicted_prosody_features_length = phonemes_length

        visualization.plot_and_save_spectrograms(
            model_output['pred_mel_spec'][:, :spec_length].numpy(),
            sample['input_spec'][:, :spec_length].numpy(),
            sr=22050,
            hop_length=256,
            output_path=output_dir.joinpath('spectrograms_target_length.svg')
        )

        visualization.plot_and_save_spectrograms(
            model_output['pred_mel_spec'][:, :predicted_spec_length].numpy(),
            sample['input_spec'][:, :spec_length].numpy(),
            sr=22050,
            hop_length=256,
            output_path=output_dir.joinpath('spectrograms_predicted_length.svg')
        )

        if 'target_pitch' in model_output:

            visualization.plot_and_save_contours(
                model_output['predicted_pitch'][:prosody_features_length],
                model_output['target_pitch'][:prosody_features_length],
                'Pitch',
                output_dir.joinpath('pitch_contours.svg')
            )

        else:

            visualization.plot_and_save_contour(
                sample['input_f0'][:predicted_prosody_features_length],
                'Pitch',
                output_dir.joinpath('pitch_contour.svg')
            )

        if 'target_energy' in model_output:

            visualization.plot_and_save_contours(
                model_output['predicted_energy'][:prosody_features_length],
                model_output['target_energy'][:prosody_features_length],
                'Energy',
                output_dir.joinpath('energy_contour.svg')
            )

        else:

            visualization.plot_and_save_contour(
                sample['input_energy'][:predicted_prosody_features_length],
                'Energy',
                output_dir.joinpath('energy_contour.svg')
            )

        visualization.plot_and_save_contours(
            model_output['predicted_durations'][:phonemes_length],
            sample['explicit_durations'][:phonemes_length],
            'Duration',
            output_dir.joinpath('duration_contour.svg')
        )

        wav = inference_utils.transform_mel_to_wav(
            model_output['pred_mel_spec'][:, :predicted_spec_length],
            hifi_gan.decode_batch,
            split_spec_by_silences=True
        )

        if wav is not None:
            soundfile.write(output_dir.joinpath('predicted.wav'),
                            wav.squeeze(0).numpy(), 22050)

        if not save_target_wav:
            return

        wav = inference_utils.transform_mel_to_wav(
            sample['input_spec'][:, :spec_length],
            hifi_gan.decode_batch,
            split_spec_by_silences=True
        )

        if wav is not None:
            soundfile.write(output_dir.joinpath('target.wav'),
                            wav.squeeze(0).numpy(), 22050)
