"""Runs training pipeline for acoustic model.

The script supports logging MLFlow experiment parameters and saves checkpoints.
"""
import logging
import os
from typing import List
import pathlib

import hydra
import lightning.pytorch as pl
import mlflow
import omegaconf
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import profilers as pl_profilers
from speechbrain.inference.vocoders import HIFIGAN
import torch

from paragraph_tts.data.loading import processed_librittsr as data_loading
from paragraph_tts.models import acoustic as acoustic_model
from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import processed_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


def _obtain_mlflow_run_id(mlflow_client: mlflow.tracking.MlflowClient,
                          experiment_name: str,
                          run_name: str) -> str | None:
    """Checks if an MLFlow run with the given name exists and returns its ID."""

    experiment = mlflow.get_experiment_by_name(experiment_name)

    runs = mlflow_client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        max_results=1
    )

    if not runs:
        return None

    return runs[0].info.run_id


@hydra.main(version_base=None, config_path='cfg', config_name='train_acoustic_model')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Runs training pipeline for acoustic model."""

    pl.seed_everything(script_cfg.run_cfg.seed, workers=True)

    ds_cfg = omegaconf.OmegaConf.to_container(script_cfg.ds_cfg)
    model_cfg = omegaconf.OmegaConf.to_container(script_cfg.model_cfg)
    train_cfg = omegaconf.OmegaConf.to_container(script_cfg.train_cfg)
    optim_cfg = omegaconf.OmegaConf.to_container(script_cfg.optimizer_cfg)

    processed_ds_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(
        pathlib.Path(script_cfg.run_cfg.ds_path)
    )

    data_module = data_loading.ProcessedLibriTTSRDataModule(
        ds_cfg=data_loading.ProcessedLibriTTSRDataset.Configuration.model_validate(ds_cfg),
        processed_ds_handler=processed_ds_handler,
        batch_size=script_cfg.run_cfg.batch_size,
        num_workers=script_cfg.run_cfg.num_workers,
        train_val_split=script_cfg.run_cfg.train_val_split
    )

    vocoder = HIFIGAN.from_hparams(
        source='speechbrain/tts-hifigan-libritts-22050Hz',
        run_opts={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    if script_cfg.run_cfg.continue_training_from_checkpoint is None:
        model = acoustic_model.AcousticModel(
            model_cfg=acoustic_model.ModelConfiguration.model_validate(model_cfg),
            optim_cfg=acoustic_model.OptimizerConfiguration.model_validate(optim_cfg),
            train_cfg=acoustic_model.TrainConfiguration.model_validate(train_cfg),
            vocoder=vocoder
        )
    else:
        model = acoustic_model.AcousticModel.load_from_checkpoint(
            script_cfg.run_cfg.continue_training_from_checkpoint,
            vocoder=vocoder
        )

    mlflow.set_tracking_uri(script_cfg.run_cfg.mlflow_server_uri)
    mlflow.set_experiment(script_cfg.run_cfg.mlflow_experiment)
    run_id = _obtain_mlflow_run_id(
        mlflow.tracking.MlflowClient(script_cfg.run_cfg.mlflow_server_uri),
        script_cfg.run_cfg.mlflow_experiment,
        script_cfg.run_cfg.mlflow_run)

    with mlflow.start_run(run_name=script_cfg.run_cfg.mlflow_run, run_id=run_id) as run:

        mlflow_logger = pl_loggers.MLFlowLogger(
            experiment_name=script_cfg.run_cfg.mlflow_experiment,
            run_name=script_cfg.run_cfg.mlflow_run,
            tracking_uri=script_cfg.run_cfg.mlflow_server_uri,
            run_id=run.info.run_id)

        mlflow_logger.log_hyperparams({
            'model_cfg': model_cfg,
            'train_cfg': train_cfg,
            'optim_cfg': optim_cfg,
            'ds_cfg': ds_cfg
        })

        mlflow_logger.log_metrics({'model_size':
                                  sum(p.numel() for p in model.parameters() if p.requires_grad)})

        logging_utils.setup_logging(
            'train_acoustic_model',
            output_dir=os.path.join(mlflow.get_artifact_uri(), 'script_logs'))

        _logger().info('Script configuration:\n%s', omegaconf.OmegaConf.to_yaml(script_cfg))
        logging.getLogger('speechbrain.utils.parameter_transfer').setLevel(logging.CRITICAL)

        callbacks = [
            pl_callbacks.DeviceStatsMonitor(cpu_stats=True),
            pl_callbacks.EarlyStopping(
                monitor='val/mel_loss', min_delta=0.0,
                patience=3,
                mode='min')
        ]

        if script_cfg.run_cfg.save_checkpoints:
            callbacks.append(
                pl_callbacks.ModelCheckpoint(
                    dirpath=os.path.join(mlflow.get_artifact_uri(), 'checkpoints'),
                    monitor='val/mel_loss',
                    mode='min',
                    save_top_k=3,
                    every_n_epochs=1)
            )

        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            max_epochs=script_cfg.run_cfg.max_epochs,
            logger=mlflow_logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            # profiler=pl_profilers.PyTorchProfiler(
            #     dirpath=os.path.join(mlflow.get_artifact_uri(),
            #                          'torch_profiler'),
            #     filename=f'profile_{run.info.run_id}',
            #     row_limit=-1,
            #     profiler_kwargs={
            #         'with_stack': True,
            #         'with_modules': True,
            #         'profile_memory': True,
            #     }
            # ),
            enable_checkpointing=script_cfg.run_cfg.save_checkpoints,
            check_val_every_n_epoch=1,
            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            log_every_n_steps=25,
            accumulate_grad_batches=train_cfg['accumulate_grad_batches'],
            gradient_clip_val=train_cfg['gradient_clip_val'],
            enable_model_summary=True
        )

        _logger().info('Starting training...')

        trainer.fit(model,
                    datamodule=data_module,
                    ckpt_path=script_cfg.run_cfg.continue_training_from_checkpoint)


if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter
