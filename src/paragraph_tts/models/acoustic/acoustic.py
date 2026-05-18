"""Contains definition of acoustic model training/inference pipelines."""
import logging
import pathlib
from typing import Annotated, Callable
import itertools

import mlflow
import lightning.pytorch as pl
import pydantic
import torch
from comp_trans_tts.model import modules as ctt_modules
from pydantic import Field
from speechbrain.inference.vocoders import HIFIGAN

from paragraph_tts.layers.acoustic import encoder as acoustic_encoder
from paragraph_tts.layers.acoustic import decoder as acoustic_decoder
from paragraph_tts.layers.acoustic import context_encoder as acoustic_context_encoder
from paragraph_tts.models.acoustic import utils as model_utils
from paragraph_tts.utils import neural as neural_utils
from paragraph_tts.utils import visualization


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

    base_lr: Annotated[float, Field(description='Learning rate for base parameters.')]

    prosody_enc_lr: Annotated[float, Field(
        description='Learning rate for prosody encoder parameters.')]

    betas: Annotated[tuple[float, float], Field(
        description='Adam beta coefficients.')]

    eps: Annotated[float, Field(description='Adam epsilon value.')]

    base_weight_decay: Annotated[float, Field(description='Weight decay for base parameters.')]

    prosody_enc_weight_decay: Annotated[float, Field(
        description='Weight decay for prosody encoder parameters.')]

    lr_scheduler_gamma: Annotated[float, Field(
        description='Exponential decay factor of LR scheduler.')]


class TrainConfiguration(pydantic.BaseModel):
    """Training loop and objective configuration."""

    loss_weights: Annotated[dict[str, float], Field(
        description='Relative weights of different loss components used for optimization.')]

    loss_weight_decays: Annotated[dict[str, float], Field(
        description='Decay factors for loss weights.')]

    wsv_bin_init_temperature: Annotated[float, Field(
        description='Initial temperature for WSV binarization in prosody encoder.')]

    gst_bin_init_temperature: Annotated[float, Field(
        description='Initial temperature for GST binarization in prosody encoder.')]

    wsv_bin_temperature_decay: Annotated[float, Field(
        description='Decay factor for temperature of WSV binarization in prosody encoder.')]

    gst_bin_temperature_decay: Annotated[float, Field(
        description='Decay factor for temperature of GST binarization in prosody encoder.')]

    wsv_bin_loss_start_epoch: Annotated[int, Field(
        description='Epoch to start applying binarization loss for prosody encoder.')]

    wsv_bin_hard_start_epoch: Annotated[int, Field(
        description='Epoch to start using hard binarization for WSV in prosody encoder.')]

    gst_bin_loss_start_epoch: Annotated[int, Field(
        description='Epoch to start applying binarization loss for GST in prosody encoder.')]


class AcousticModel(pl.LightningModule):
    """Predicts mel-spectrogram from input textual features.

    The acoustic model supports explicit prosody
    """

    def __init__(self,
                 model_cfg: ModelConfiguration,
                 optim_cfg: OptimizerConfiguration,
                 train_cfg: TrainConfiguration,
                 visualize_n_batches: int,
                 visualize_n_samples_per_batch: int) -> None:

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
        self._vocoder: Callable[[], HIFIGAN] | None = None
        self._visualize_n_batches = visualize_n_batches
        self._visualize_n_samples_per_batch = visualize_n_samples_per_batch
        self.strict_loading = False

        self.save_hyperparameters(
            logger=False,
            ignore=['visualize_n_batches', 'visualize_n_samples_per_batch']
        )

        self._loss = model_utils.AcousticModelLoss(
            train_cfg.loss_weights,
            train_cfg.loss_weight_decays,
            train_cfg.wsv_bin_loss_start_epoch,
            train_cfg.gst_bin_loss_start_epoch

        )
        self._metrics = model_utils.AcousticModelMetrics()

    def configure_optimizers(self):  # type: ignore
        """Sets up optimizer from config."""

        param_groups = [{
            'params': itertools.chain(
                self._encoder.parameters(),
                self._decoder.parameters(),
                self._var_adaptor.parameters(),
                *[self._context_encoder.parameters()] if self._context_encoder is not None else [],
            ),
            'lr': self._optim_cfg.base_lr,
            'weight_decay': self._optim_cfg.base_weight_decay
        }]

        if self._prosody_encoder is not None:

            param_groups.append({
                'params': self._prosody_encoder.parameters(),
                'lr': self._optim_cfg.prosody_enc_lr,
                'weight_decay': self._optim_cfg.prosody_enc_weight_decay
            })

        base_opt = torch.optim.AdamW(
            param_groups,
            betas=self._optim_cfg.betas,
            eps=self._optim_cfg.eps,
        )

        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            base_opt, gamma=self._optim_cfg.lr_scheduler_gamma
        )

        return {
            'optimizer': base_opt,
            'lr_scheduler': scheduler,
        }

    def forward(self,  # pylint: disable=arguments-differ
                inputs: dict[str, torch.Tensor],
                use_teacher_forcing: bool,
                wsv_bin_params: ctt_modules.StlBinarizationParams | None,
                gst_bin_params: ctt_modules.StlBinarizationParams | None
                ) -> dict[str, torch.Tensor]:
        """Performs forward pass of the model."""

        enc_output = self._obtain_encoder_outputs(inputs)

        prosody_enc_outputs: dict[str, torch.Tensor] = {}

        if self._prosody_encoder is not None:

            assert wsv_bin_params is not None

            prosody_enc_outputs = self.obtain_prosody_encoder_outputs(inputs,
                                                                      wsv_bin_params,
                                                                      gst_bin_params)

            gst_emb = prosody_enc_outputs['gst_emb'].unsqueeze(1).expand_as(enc_output)
            wsv_emb = prosody_enc_outputs['wsv_emb'][torch.arange(enc_output.size(0)).unsqueeze(1),
                                                     inputs['word_to_phoneme_indices']]

            enc_output = enc_output + gst_emb + wsv_emb

        dec_outputs = self._obtain_decoder_outputs(inputs, use_teacher_forcing, enc_output)

        return {**dec_outputs,
                **prosody_enc_outputs}

    def inference(self,
                  inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Performs inference with the model.

        If prosody encoder is used, the `inputs` dict is supposed to contain 'gst_weights' and
        'wsv_weights' tensors, as well as 'gst_to_phone_indices' for upsampling prosody embeddings
        to phoneme level.
        """

        enc_output = self._obtain_encoder_outputs(inputs)

        if self._prosody_encoder is not None:

            gst_emb = self._prosody_encoder.global_stl.get_emb_from_weights(inputs['gst_weights'])
            wsv_emb = self._prosody_encoder.local_stl.get_emb_from_weights(inputs['input_word_emb'])

            gst_emb = gst_emb[torch.arange(enc_output.size(0)).unsqueeze(1),
                              inputs['gst_to_phone_indices']]
            wsv_emb = wsv_emb[torch.arange(enc_output.size(0)).unsqueeze(1),
                              inputs['word_to_phoneme_indices']]

            enc_output = enc_output + gst_emb + wsv_emb

        return self._obtain_decoder_outputs(inputs,
                                            use_teacher_forcing=False,
                                            enc_output=enc_output)

    def obtain_prosody_encoder_outputs(self,
                                       inputs: dict[str, torch.Tensor],
                                       wsv_bin_params: ctt_modules.StlBinarizationParams,
                                       gst_bin_params: ctt_modules.StlBinarizationParams
                                       ) -> dict[str, torch.Tensor]:
        """Calculates outputs of hierarchical prosody encoder."""

        ((gst_emb, gst_weights), (wsv_emb, wsv_weights)) = self._prosody_encoder(
            spectrogram=inputs['input_spec'],
            spectrogram_length=inputs['input_spec_length'],
            spec_word_pool_matrix=inputs['spec_to_word_pool_matrix'],
            phoneme_ids=inputs['input_phoneme_ids'],
            linguistic_features=inputs['input_ling_stats'],
            phoneme_spec_indices=inputs['phone_to_spec_indices'],
            local_stl_binarization_params=wsv_bin_params,
            global_stl_binarization_params=gst_bin_params
        )

        if wsv_bin_params.hard:
            gst_emb = gst_emb.detach()
            wsv_emb = wsv_emb.detach()
            gst_weights = gst_weights.detach()
            wsv_weights = wsv_weights.detach()

        outputs = {
            'gst_weights': gst_weights,
            'wsv_weights': wsv_weights,
            'gst_emb': gst_emb,
            'wsv_emb': wsv_emb
        }

        return outputs

    def _obtain_wsv_binarization_params(self) -> ctt_modules.StlBinarizationParams | None:
        """Calculates current binarization parameters for WSV in hierarchical prosody encoder."""

        if self._prosody_encoder is None:
            return None

        if self.current_epoch < self._train_cfg.wsv_bin_hard_start_epoch:
            return ctt_modules.StlBinarizationParams(
                hard=False,
                temperature=neural_utils.calc_decayed_loss_weight(
                    self._train_cfg.wsv_bin_init_temperature,
                    self._train_cfg.wsv_bin_temperature_decay,
                    self.current_epoch
                )
            )

        return ctt_modules.StlBinarizationParams(
            hard=True,
            temperature=neural_utils.calc_decayed_loss_weight(
                self._train_cfg.wsv_bin_init_temperature,
                self._train_cfg.wsv_bin_temperature_decay,
                self._train_cfg.wsv_bin_hard_start_epoch
            )
        )

    def _obtain_gst_binarization_params(self) -> ctt_modules.StlBinarizationParams | None:
        """Calculates current binarization parameters for GST in hierarchical prosody encoder."""

        if self._prosody_encoder is None:
            return None

        return ctt_modules.StlBinarizationParams(
            hard=False,
            temperature=neural_utils.calc_decayed_loss_weight(
                self._train_cfg.gst_bin_init_temperature,
                self._train_cfg.gst_bin_temperature_decay,
                self.current_epoch
            )
        )

    def _obtain_decoder_outputs(self,
                                inputs: dict[str, torch.Tensor],
                                use_teacher_forcing: bool,
                                enc_output: torch.Tensor) -> dict[str, torch.Tensor]:
        """Calculates decoder outputs (mel-spectrogram and intermediate representations)."""

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
            mel_length = var_adaptor_output['durations_rounded'].sum(dim=1)

        pred_mel_spec = self._decoder(
            var_adaptor_output['output'],
            mel_length
        )

        return {
            'pred_mel_spec': pred_mel_spec,
            **var_adaptor_output
        }

    def _obtain_encoder_outputs(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Calculates encoder outputs from input textual features."""

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

        return enc_output

    def on_fit_start(self):
        """Fit start hook."""

        self.logger.log_metrics(
            {'model_size': sum(p.numel() for p in self.parameters() if p.requires_grad)}
        )

        vocoder: torch.nn.Module = HIFIGAN.from_hparams(
            source='speechbrain/tts-hifigan-libritts-22050Hz',
            run_opts={'device': str(self.device)}
        )
        vocoder.eval()
        for param in vocoder.parameters():
            param.requires_grad = False

        self._vocoder = lambda: vocoder

    def training_step(self,  # pylint: disable=arguments-differ
                      batch: dict[str, torch.Tensor]) -> torch.Tensor:
        """Performs training step."""

        model_output = self.forward(batch,
                                    use_teacher_forcing=True,
                                    wsv_bin_params=self._obtain_wsv_binarization_params(),
                                    gst_bin_params=self._obtain_gst_binarization_params())

        losses = self._loss(model_output, batch, self.current_epoch)

        with torch.no_grad():
            metrics = self._metrics(model_output, batch)

        self.log_dict({f'train/{k}': v for k, v in {**losses, **metrics}.items()},
                      on_step=True,
                      on_epoch=False,
                      batch_size=batch['input_phonemes_length'].size(0))

        return losses['total_loss']

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch: dict[str, torch.Tensor],
                        batch_idx: int) -> None:
        """Performs validation step."""

        model_output = self(batch,
                            use_teacher_forcing=True,
                            wsv_bin_params=self._obtain_wsv_binarization_params(),
                            gst_bin_params=self._obtain_gst_binarization_params())

        losses = self._loss(model_output, batch, self.current_epoch)
        metrics = self._metrics(model_output, batch)

        self.log_dict({f'val/{k}': v for k, v in {**losses, **metrics}.items()},
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch['input_phonemes_length'].size(0))

        if batch_idx < self._visualize_n_batches:

            model_output_inference = self(batch,
                                          use_teacher_forcing=False,
                                          wsv_bin_params=self._obtain_wsv_binarization_params(),
                                          gst_bin_params=self._obtain_gst_binarization_params())

            for sample_idx in range(min(self._visualize_n_samples_per_batch,
                                        batch['input_phonemes_length'].size(0))):

                sample = {k: v[sample_idx].cpu() for k, v in batch.items()}

                visualization.visualize_acoustic_model_outputs(
                    sample,
                    {k: v[sample_idx].cpu() for k, v in model_output.items()},
                    self._vocoder(),
                    save_target_wav=(self.current_epoch == 0),
                    output_dir=(pathlib.Path(mlflow.get_artifact_uri())
                                .joinpath('viz')
                                .joinpath(f'epoch_{self.current_epoch}')
                                .joinpath(f'batch_{batch_idx}')
                                .joinpath(f'sample_{sample_idx}')
                                .joinpath('val_teacher_forcing'))
                )

                visualization.visualize_acoustic_model_outputs(
                    sample,
                    {k: v[sample_idx].cpu() for k, v in model_output_inference.items()},
                    self._vocoder(),
                    save_target_wav=False,
                    output_dir=(pathlib.Path(mlflow.get_artifact_uri())
                                .joinpath('viz')
                                .joinpath(f'epoch_{self.current_epoch}')
                                .joinpath(f'batch_{batch_idx}')
                                .joinpath(f'sample_{sample_idx}')
                                .joinpath('val_inference'))
                )

    def on_validation_epoch_end(self):
        """Validation epoch end hook."""

        self.log_dict({f'loss_weights/{k}': v
                       for k, v in self._loss.get_current_loss_weights(self.current_epoch).items()},
                      on_step=False,
                      on_epoch=True)
