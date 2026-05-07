"""Runs the training pipeline for the STL predictor."""

import logging
import os
import pathlib

import hydra
import omegaconf
import lightning.pytorch as pl
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import loggers as pl_loggers
import mlflow

from paragraph_tts.utils import logging_utils
from paragraph_tts.models import stl_predictor
from paragraph_tts.layers import stl_predictor as predictor_layers
from paragraph_tts.data.loading import stl_predictor_ds
from paragraph_tts.utils.path import stl_predictor_ds_handler


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='train_stl_predictor')
def main(config: omegaconf.DictConfig) -> None:
    """Runs the training pipeline for the STL predictor."""

    ds_cfg = stl_predictor_ds.STLPredictorDataset.Configuration.model_validate(config.ds_cfg)
    train_cfg = stl_predictor.TrainConfig.model_validate(config.train_cfg)
    optimizer_cfg = stl_predictor.OptimizerConfig.model_validate(config.optimizer_cfg)
    model_cfg = predictor_layers.Encoder.Configuration.model_validate(config.model_cfg)

    ds_path_handler = stl_predictor_ds_handler.STLPredictorDatasetHandler(
        pathlib.Path(config.run_cfg.ds_path)
    )

    data_module = stl_predictor_ds.STLPredictorDataModule(ds_path_handler,
                                                          ds_cfg,
                                                          batch_size=config.run_cfg.batch_size,
                                                          num_workers=config.run_cfg.num_workers,
                                                          seed=config.run_cfg.seed)

    model = stl_predictor.STLPredictor(
        model_cfg=model_cfg,
        optimizer_cfg=optimizer_cfg,
        train_cfg=train_cfg,
        viz_n_batches=config.run_cfg.viz_n_batches,
        viz_n_samples_per_batch=config.run_cfg.viz_n_samples_per_batch)

    mlflow.set_tracking_uri(config.run_cfg.mlflow_server_uri)
    mlflow.set_experiment(config.run_cfg.mlflow_experiment)

    with mlflow.start_run(run_name=config.run_cfg.mlflow_run,
                          run_id=config.run_cfg.mlflow_run_id) as run:

        mlflow_logger = pl_loggers.MLFlowLogger(
            experiment_name=config.run_cfg.mlflow_experiment,
            run_name=config.run_cfg.mlflow_run,
            tracking_uri=config.run_cfg.mlflow_server_uri,
            run_id=run.info.run_id)

        mlflow_logger.log_hyperparams({
            'model_cfg': model_cfg.model_dump(),
            'train_cfg': train_cfg.model_dump(),
            'optim_cfg': optimizer_cfg.model_dump(),
            'ds_cfg': ds_cfg.model_dump()
        })

        logging_utils.setup_logging('train_stl_predictor')
        _logger().info('Script configuration:\n%s', omegaconf.OmegaConf.to_yaml(config))

        callbacks: list[pl_callbacks.Callback] = []

        if config.run_cfg.save_checkpoints:
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
            max_epochs=config.run_cfg.max_epochs,
            logger=mlflow_logger,
            callbacks=callbacks,
            num_sanity_val_steps=0,
            enable_checkpointing=config.run_cfg.save_checkpoints,
            check_val_every_n_epoch=1,
            limit_train_batches=None,
            limit_val_batches=None,
            limit_test_batches=None,
            log_every_n_steps=25,
            gradient_clip_val=config.run_cfg.grad_clip_val,
            enable_model_summary=True,
            accumulate_grad_batches=config.run_cfg.accumulate_grad_batches
        )

        _logger().info('Starting training...')

        trainer.fit(model,
                    datamodule=data_module,
                    ckpt_path=config.run_cfg.continue_training_from_checkpoint,
                    weights_only=False)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
