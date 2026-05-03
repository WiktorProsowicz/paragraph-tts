"""Runs training pipeline for acoustic model.

The script supports logging MLFlow experiment parameters and saves checkpoints.
"""
import logging
import os
import pathlib

import hydra
import mlflow
import lightning.pytorch as pl
import omegaconf
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import profilers as pl_profilers
import torch

from paragraph_tts.data.loading import processed_librittsr as data_loading
from paragraph_tts.models import acoustic as acoustic_model
from paragraph_tts.utils.path import processed_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


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
        train_val_split=script_cfg.run_cfg.train_val_split,
        seed=script_cfg.run_cfg.seed,
    )

    model = acoustic_model.AcousticModel(
        model_cfg=acoustic_model.ModelConfiguration.model_validate(model_cfg),
        optim_cfg=acoustic_model.OptimizerConfiguration.model_validate(optim_cfg),
        train_cfg=acoustic_model.TrainConfiguration.model_validate(train_cfg),
        visualize_n_batches=script_cfg.run_cfg.visualize_n_batches,
        visualize_n_samples_per_batch=script_cfg.run_cfg.visualize_n_samples_per_batch
    )

    if script_cfg.run_cfg.load_from_checkpoint is not None:

        _logger().info('Loading model weights from checkpoint: %s',
                       script_cfg.run_cfg.load_from_checkpoint)

        loaded_state_dict = torch.load(script_cfg.run_cfg.load_from_checkpoint,
                                       map_location='cpu',
                                       weights_only=False)['state_dict']

        model.load_state_dict(loaded_state_dict, strict=False)

    mlflow.set_tracking_uri(script_cfg.run_cfg.mlflow_server_uri)
    mlflow.set_experiment(script_cfg.run_cfg.mlflow_experiment)

    with mlflow.start_run(run_name=script_cfg.run_cfg.mlflow_run,
                          run_id=script_cfg.run_cfg.mlflow_run_id) as run:

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

        _logger().info('Script configuration:\n%s', omegaconf.OmegaConf.to_yaml(script_cfg))
        logging.getLogger('speechbrain.utils.parameter_transfer').setLevel(logging.CRITICAL)

        callbacks: list[pl_callbacks.Callback] = []

        if script_cfg.run_cfg.save_checkpoints:
            callbacks.append(
                pl_callbacks.ModelCheckpoint(
                    dirpath=os.path.join(mlflow.get_artifact_uri(), 'checkpoints'),
                    monitor='epoch',
                    mode='max',
                    save_top_k=5,
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
            accumulate_grad_batches=script_cfg.run_cfg['accumulate_grad_batches'],
            gradient_clip_val=None,
            enable_model_summary=True
        )

        _logger().info('Starting training...')

        trainer.fit(model,
                    datamodule=data_module,
                    ckpt_path=script_cfg.run_cfg.continue_training_from_checkpoint,
                    weights_only=False)


if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter
