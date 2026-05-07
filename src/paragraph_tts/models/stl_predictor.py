"""Contains implementation of the STL predictor model."""
from typing import Annotated, TypedDict
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

    gst_pred_loss_weight: Annotated[float, Field(
        description="The weight of the GST prediction loss in the total loss.")]

    wsv_pred_loss_weight: Annotated[float, Field(
        description="The weight of the WSV prediction loss in the total loss.")]

    moe_lb_loss_weight: Annotated[float, Field(
        description="The weight of the MoE load balancing loss in the total loss.")]

    moe_z_loss_weight: Annotated[float, Field(
        description="The weight of the MoE router z-loss in the total loss.")]


class STLPredictor(pl.LightningModule):
    """Predicts GST/WSV weights for the sentences in a given paragraph.

    The weights are predicted based on the input textual embeddings and are used during inference
    to compute the GST/WSV embeddings without the reference audio.
    """

    class ForwardOutput(TypedDict):
        """Output of the forward pass through the STL predictor."""

        wsv_logits: torch.Tensor
        gst_logits: torch.Tensor
        topk_indices: dict[str, list[torch.Tensor]]
        router_logits: dict[str, list[torch.Tensor]]

    def __init__(self,
                 model_cfg: predictor_layers.Encoder.Configuration,
                 optimizer_cfg: OptimizerConfig,
                 train_cfg: TrainConfig,
                 viz_n_batches: int,
                 viz_n_samples_per_batch: int) -> None:

        super().__init__()

        self._encoder = predictor_layers.Encoder(model_cfg)

        self._model_cfg = model_cfg
        self._optimizer_cfg = optimizer_cfg
        self._train_cfg = train_cfg
        self._viz_n_batches = viz_n_batches
        self._viz_n_samples_per_batch = viz_n_samples_per_batch

        self.save_hyperparameters(logger=False,
                                  ignore=['viz_n_batches', 'viz_n_samples_per_batch'])

        self._gst_loss = torch.nn.KLDivLoss(reduction='batchmean')
        self._wsv_loss = torch.nn.KLDivLoss(reduction='batchmean')

    def configure_optimizers(self):  # type: ignore

        return torch.optim.AdamW(self.parameters(),
                                 lr=self._optimizer_cfg.learning_rate,
                                 weight_decay=self._optimizer_cfg.weight_decay,
                                 betas=self._optimizer_cfg.betas,
                                 eps=self._optimizer_cfg.eps
                                 )

    def forward(self,  # pylint: disable=arguments-differ
                batch_graph: HeteroData) -> ForwardOutput:
        """Predicts the GST/WSV weights for the sentences in the input graph."""

        return self._encoder(batch_graph)

    def training_step(self,  # pylint: disable=arguments-differ
                      batch_graph: HeteroData) -> torch.Tensor:
        """Training step."""

        predictions = self(batch_graph)
        losses = self._calculate_loss(predictions, batch_graph)

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
        losses = self._calculate_loss(predictions, batch_graph)

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

                gst_logits = predictions['gst_logits'][cum_global_length:
                                                       cum_global_length + global_len]
                wsv_logits = predictions['wsv_logits'][cum_local_length:
                                                       cum_local_length + local_len]

                cum_global_length += global_len
                cum_local_length += local_len

                self._visualize_predictions(
                    graph.cpu(),
                    gst_logits.cpu(),
                    wsv_logits.cpu(),
                    original_ds.get_sample_metadata(batch_idx * batch_graph.batch_size + graph_idx),
                    (pathlib.Path(mlflow.get_artifact_uri())
                     .joinpath('viz')
                     .joinpath(f'epoch_{self.current_epoch}')
                     .joinpath(f'batch_{batch_idx}')
                     .joinpath(f'sample_{graph_idx}')))

    def _calculate_loss(self,
                        predictions: ForwardOutput,
                        batch_graph: HeteroData) -> dict[str, torch.Tensor]:
        """Calculates the loss for the given predictions and batch graph."""

        chosen_gst_logits = predictions['gst_logits'][batch_graph.has_gst_mask]
        chosen_wsv_logits = predictions['wsv_logits'][batch_graph.has_wsv_mask]

        gst_loss = self._gst_loss(torch.log_softmax(chosen_gst_logits, dim=-1),
                                  batch_graph.gst_weights)
        wsv_loss = self._wsv_loss(torch.log_softmax(chosen_wsv_logits, dim=-1),
                                  batch_graph.wsv_weights)

        load_balancing_losses = {}
        z_losses = {}

        for node_type in ('word_emb', 'global_emb'):

            for block_idx, (topk_indices, router_logits) in enumerate(
                zip(predictions['topk_indices'][node_type],
                    predictions['router_logits'][node_type])
            ):

                load_balancing_losses[f'moe_lb_loss/{node_type}_block_{block_idx}'] = (
                    model_utils.moe_load_balancing_loss(router_logits, topk_indices)
                )

                z_losses[f'moe_z_loss/{node_type}_block_{block_idx}'] = (
                    model_utils.moe_router_z_loss(router_logits)
                )

        total_loss = (self._train_cfg.gst_pred_loss_weight * gst_loss +
                      self._train_cfg.wsv_pred_loss_weight * wsv_loss +
                      self._train_cfg.moe_lb_loss_weight * sum(load_balancing_losses.values()) +
                      self._train_cfg.moe_z_loss_weight * sum(z_losses.values()))

        return {
            'total_loss': total_loss,
            'gst_pred_loss': gst_loss,
            'wsv_pred_loss': wsv_loss,
            **load_balancing_losses,
            **z_losses
        }

    def _calculate_metrics(self,
                           predictions: ForwardOutput,
                           batch_graph: HeteroData) -> dict[str, torch.Tensor]:
        """Calculates the metrics for the given predictions and batch graph."""

        chosen_gst_logits = predictions['gst_logits'][batch_graph.has_gst_mask]
        chosen_wsv_logits = predictions['wsv_logits'][batch_graph.has_wsv_mask]

        gst_pred_ae = torch.abs(torch.softmax(chosen_gst_logits, dim=-1) - batch_graph.gst_weights)
        wsv_pred_ae = torch.abs(torch.softmax(chosen_wsv_logits, dim=-1) - batch_graph.wsv_weights)

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
                               gst_logits: torch.Tensor,
                               wsv_logits: torch.Tensor,
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

        pred_gst_logits = gst_logits[graph.has_gst_mask]
        pred_wsv_logits = wsv_logits[graph.has_wsv_mask]

        for utt_idx, utt_meta in enumerate(utterances):

            if utt_meta.stl_weights is None:
                continue

            utt_length = int(graph.sentence_lengths[utt_idx].item())

            viz_utils.plot_and_save_wsv_prediction(
                torch.softmax(pred_wsv_logits[cum_length:cum_length + utt_length], dim=-1),
                graph.wsv_weights[cum_length:cum_length + utt_length],
                output_dir.joinpath(f'utt_{utt_meta.raw_utterance.utt_id}_wsv_prediction.svg')
            )

            viz_utils.plot_and_save_gst_prediction(
                torch.softmax(pred_gst_logits[utt_idx], dim=-1),
                graph.gst_weights[utt_idx],
                output_dir.joinpath(f'utt_{utt_meta.raw_utterance.utt_id}_gst_prediction.svg')
            )

            cum_length += utt_length
