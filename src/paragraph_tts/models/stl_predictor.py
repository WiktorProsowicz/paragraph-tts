"""Contains implementation of the STL predictor model."""
from typing import Annotated, Literal
import pathlib

import pydantic
from pydantic import Field
from lightning import pytorch as pl
import torch
from torch_geometric.data import HeteroData
import mlflow

from paragraph_tts.layers import stl_predictor as predictor_layers
from paragraph_tts.models import utils as model_utils
from paragraph_tts.data.loading import stl_predictor_ds
from paragraph_tts.utils.path import stl_predictor_ds_handler
from paragraph_tts.utils import visualization as viz_utils


class OptimizerConfig(pydantic.BaseModel):
    """Configuration for the STL predictor optimizer."""

    learning_rate: float
    weight_decay: float
    betas: tuple[float, float]
    eps: float


class TrainConfig(pydantic.BaseModel):
    """Configuration for training the STL predictor."""

    loss_init_weights: Annotated[dict[str, float], Field(
        description='Initial weights for each loss component.')]

    loss_weight_decays: Annotated[dict[str, float], Field(
        description='Decay rates for each loss component weights.')]

    wsv_cl_pos_weight: Annotated[float | None, Field(
        description='Positive class weight for the WSV classification loss.')]


class ModelConfig(pydantic.BaseModel):
    """Configuration for the STL predictor model."""

    encoder: predictor_layers.Encoder.Configuration

    output_mode: Annotated[Literal['logits', 'weights', 'classification-plus-weights'], Field(
        description='Determines the output mode for the predictor.')]


class STLPredictor(pl.LightningModule):
    """Predicts GST/WSV weights for the sentences in a given paragraph.

    The weights are predicted based on the input textual embeddings and are used during inference
    to compute the GST/WSV embeddings without the reference audio.
    """

    def __init__(self,
                 model_cfg: ModelConfig,
                 optimizer_cfg: OptimizerConfig,
                 train_cfg: TrainConfig,
                 viz_n_batches: int,
                 viz_n_samples_per_batch: int) -> None:

        super().__init__()

        self._encoder = predictor_layers.Encoder(model_cfg.encoder)

        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self._train_cfg = train_cfg
        self._viz_n_batches = viz_n_batches
        self._viz_n_samples_per_batch = viz_n_samples_per_batch

        self.save_hyperparameters(logger=False,
                                  ignore=['viz_n_batches', 'viz_n_samples_per_batch'])

        self._loss = model_utils.STLPredictorLoss(
            loss_weights=train_cfg.loss_init_weights,
            loss_weight_decays=train_cfg.loss_weight_decays,
            model_output_mode=model_cfg.output_mode,
            wsv_cl_pos_weight=train_cfg.wsv_cl_pos_weight
        )

    def configure_optimizers(self):  # type: ignore

        return torch.optim.AdamW(self.parameters(),
                                 lr=self._optimizer_cfg.learning_rate,
                                 weight_decay=self._optimizer_cfg.weight_decay,
                                 betas=self._optimizer_cfg.betas,
                                 eps=self._optimizer_cfg.eps)

    def forward(self,  # pylint: disable=arguments-differ
                batch_graph: HeteroData) -> model_utils.STLPredictorOutput:
        """Predicts the GST/WSV weights for the sentences in the input graph."""

        outputs = self._encoder(batch_graph)

        if self._model_cfg.output_mode == 'classification-plus-weights':

            pos_wsv_mask = (torch.nn.functional.sigmoid(outputs['wsv_cl_logits']) > 0.5).float()
            outputs['final_wsv_weights'] = outputs['wsv_logits'] * pos_wsv_mask
            outputs['final_gst_weights'] = torch.softmax(outputs['gst_logits'], dim=-1)

        elif self._model_cfg.output_mode == 'logits':

            outputs['final_wsv_weights'] = torch.softmax(outputs['wsv_logits'], dim=-1)
            outputs['final_gst_weights'] = torch.softmax(outputs['gst_logits'], dim=-1)

        else:
            outputs['final_wsv_weights'] = outputs['wsv_logits']
            outputs['final_gst_weights'] = outputs['gst_logits']

        return outputs

    def training_step(self,  # pylint: disable=arguments-differ
                      batch_graph: HeteroData) -> torch.Tensor:
        """Training step."""

        predictions = self(batch_graph)
        losses = self._loss(predictions, batch_graph, epoch=self.current_epoch)

        with torch.no_grad():
            metrics = self._calculate_metrics(predictions, batch_graph)

        self.log_dict({f'train/{key}': value for key, value in losses.items()},
                      on_step=True,
                      on_epoch=False,
                      batch_size=batch_graph.batch_size)

        self.log_dict({f'train/{key}': value for key, value in metrics.items()},
                      on_step=True,
                      on_epoch=False,
                      batch_size=batch_graph.batch_size)

        return losses['total_loss']

    def validation_step(self,  # pylint: disable=arguments-differ
                        batch_graph: HeteroData,
                        batch_idx: int) -> None:
        """Validation step."""

        predictions = self(batch_graph)
        losses = self._loss(predictions, batch_graph, epoch=self.current_epoch)

        with torch.no_grad():
            metrics = self._calculate_metrics(predictions, batch_graph)

        self.log_dict({f"val/{key}": value for key, value in losses.items()},
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_graph.batch_size)

        self.log_dict({f"val/{key}": value for key, value in metrics.items()},
                      on_step=False,
                      on_epoch=True,
                      batch_size=batch_graph.batch_size)

        original_ds: stl_predictor_ds.STLPredictorDataset = self.trainer.datamodule.val_ds

        if batch_idx < self._viz_n_batches:

            sample_graphs = [
                batch_graph.get_example(sample_idx)
                for sample_idx in range(min(self._viz_n_samples_per_batch, batch_graph.batch_size))
            ]

            cum_global_length = 0
            cum_local_length = 0

            for graph_idx, graph in enumerate(sample_graphs):

                global_len = int(graph['global_emb'].x.size(0))
                local_len = int(graph['word_emb'].x.size(0))

                gst_weights = predictions['final_gst_weights'][cum_global_length:
                                                               cum_global_length + global_len]
                wsv_weights = predictions['final_wsv_weights'][cum_local_length:
                                                               cum_local_length + local_len]

                cum_global_length += global_len
                cum_local_length += local_len

                self._visualize_predictions(
                    graph.cpu(),
                    gst_weights.cpu(),
                    wsv_weights.cpu(),
                    original_ds.get_sample_metadata(batch_idx * batch_graph.batch_size + graph_idx),
                    (pathlib.Path(mlflow.get_artifact_uri())
                     .joinpath('viz')
                     .joinpath(f'epoch_{self.current_epoch}')
                     .joinpath(f'batch_{batch_idx}')
                     .joinpath(f'sample_{graph_idx}')))

    def _calculate_metrics(self,
                           predictions: model_utils.STLPredictorOutput,
                           batch_graph: HeteroData) -> dict[str, torch.Tensor]:
        """Calculates the metrics for the given predictions and batch graph."""

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

        moe_expert_usage = {}

        for node_type in ('word_emb', 'global_emb'):

            for block_idx, (topk_indices, router_logits) in enumerate(
                zip(predictions['topk_indices'][node_type],
                    predictions['router_logits'][node_type])
            ):

                moe_expert_usage[f'{node_type}/block_{block_idx}'] = (
                    model_utils.moe_expert_usage(router_logits, topk_indices)
                )

        for key, expert_usage in moe_expert_usage.items():
            metrics[f'experts_usage/{key}_max'] = expert_usage.max()
            metrics[f'experts_usage/{key}_min'] = expert_usage.min()

        return metrics

    def _visualize_predictions(self,
                               graph: HeteroData,
                               gst_weights: torch.Tensor,
                               wsv_weights: torch.Tensor,
                               sample_metadata: stl_predictor_ds_handler.ProcessedParagraph,
                               output_dir: pathlib.Path
                               ) -> None:
        """Visualizes the predictions for a single sample graph."""

        output_dir.mkdir(parents=True, exist_ok=True)
        cum_length = 0

        with output_dir.joinpath('raw_paragraph.json').open('w') as f:
            f.write(sample_metadata.raw_paragraph.model_dump_json(indent=4))

        utterances = [
            utt for utt in sample_metadata.utterances
            if utt.stl_weights is not None
        ]

        pred_gst_weights = gst_weights[graph.has_gst_mask]
        pred_wsv_weights = wsv_weights[graph.has_wsv_mask]
        sentence_lengths = graph.sentence_lengths[graph.has_gst_mask]

        for utt_idx, utt_meta in enumerate(utterances):

            utt_length = int(sentence_lengths[utt_idx].item())
            wsv_pred_utt = pred_wsv_weights[cum_length:cum_length + utt_length]
            gst_pred_utt = pred_gst_weights[utt_idx]

            viz_utils.plot_and_save_wsv_prediction(
                wsv_pred_utt,
                graph.wsv_weights[cum_length:cum_length + utt_length],
                output_dir.joinpath(f'utt_{utt_meta.raw_utterance.utt_id}_wsv_prediction.svg')
            )

            viz_utils.plot_and_save_gst_prediction(
                gst_pred_utt,
                graph.gst_weights[utt_idx],
                output_dir.joinpath(f'utt_{utt_meta.raw_utterance.utt_id}_gst_prediction.svg')
            )

            cum_length += utt_length
