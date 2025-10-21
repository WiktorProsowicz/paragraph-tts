"""Runs training pipeline for acoustic model.

The script supports logging MLFlow experiment parameters and saves checkpoints.
"""

import logging
import hydra
import omegaconf
import os

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch import callbacks as pl_callbacks
from lightning.pytorch import profilers as pl_profilers
import mlflow

from paragraph_tts.data import loading as data_loading
from paragraph_tts.models import acoustic as acoustic_model
from paragraph_tts.utils import logging_utils


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='train_acoustic_model')
def main(script_cfg: omegaconf.DictConfig):
    """Runs training pipeline for acoustic model."""

    # Set seed for reproducibility
    if script_cfg.train_cfg.seed is not None:
        pl.seed_everything(script_cfg.train_cfg.seed, workers=True)

    data_cfg = script_cfg.data_cfg

    data_module = data_loading.processed_librittsr.ProcessedLibriTTSR(
        ds_path=data_cfg.processed_ds_path,
        batch_size=data_cfg.batch_size,
        num_workers=data_cfg.num_workers,
        num_test_samples=data_cfg.num_test_samples,
        train_val_split=data_cfg.train_val_split,
        n_pitch_bins=data_cfg.explicit_prosody_quant.n_pitch_bins,
        pitch_bounds=(data_cfg.explicit_prosody_quant.pitch_bounds_min,
                      data_cfg.explicit_prosody_quant.pitch_bounds_max),
        n_energy_bins=data_cfg.explicit_prosody_quant.n_energy_bins,
        energy_bounds=(data_cfg.explicit_prosody_quant.energy_bounds_min,
                       data_cfg.explicit_prosody_quant.energy_bounds_max)
    )

    if script_cfg.train_cfg.load_model_from_checkpoint is None:

        model = acoustic_model.AcousticModel(
            model_cfg=omegaconf.OmegaConf.to_container(script_cfg.model_cfg),
            optim_cfg=omegaconf.OmegaConf.to_container(script_cfg.optim_cfg),
            train_cfg=omegaconf.OmegaConf.to_container(script_cfg.train_cfg),
            data_cfg=omegaconf.OmegaConf.to_container(script_cfg.data_cfg)
        )

    else:
        model = acoustic_model.AcousticModel.load_from_checkpoint(
            script_cfg.train_cfg.load_model_from_checkpoint
        )

    experiment = mlflow.set_experiment(script_cfg.run_cfg.mlflow_experiment)

    client = mlflow.tracking.MlflowClient()

    run_id = None
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string=f"tags.mlflow.runName = '{script_cfg.run_cfg.mlflow_run_name}'",
        max_results=1
    )

    if runs:
        run_id = runs[0].info.run_id

    with mlflow.start_run(run_name=script_cfg.run_cfg.mlflow_run_name,
                          run_id=run_id) as run:

        run_path = os.path.join('mlruns', experiment.experiment_id, run.info.run_id)

        logging_utils.setup_logging('train_acoustic_model',
                                    output_dir=os.path.join(run_path, 'script_logs'))

        _logger().info('Script configuration:\n%s', omegaconf.OmegaConf.to_yaml(script_cfg))
        logging.getLogger('speechbrain.utils.parameter_transfer').setLevel(logging.CRITICAL)

        profiler = None
        limit_train_batches = None
        limit_val_batches = None
        limit_test_batches = None

        callbacks = [pl_callbacks.ModelSummary(max_depth=2),]

        if script_cfg.run_cfg.dev_run:
            profiler = pl_profilers.PyTorchProfiler(
                dirpath=os.path.join('tensorboard',
                                     f'{experiment.name}/{run.info.run_name}',
                                     'version_0'),
                filename=f'profile_{run.info.run_id}',
                row_limit=-1,
                profiler_kwargs={
                    'with_stack': True,
                    'with_modules': True,
                    'profile_memory': True,
                }
            )

            # profiler = pl_profilers.AdvancedProfiler(
            #     os.path.join(run_path, 'profiling'),
            #     'advanced_profiler',
            #     dump_stats=False)

            limit_train_batches = 50
            limit_val_batches = 5
            limit_test_batches = 5

        if script_cfg.train_cfg.save_checkpoints:
            callbacks.append(
                pl_callbacks.ModelCheckpoint(
                    dirpath=os.path.join(run_path, 'checkpoints'),
                    monitor='val/mel_loss',
                    mode='min',
                    save_top_k=3,
                    every_n_epochs=1)
            )

        if script_cfg.run_cfg.dev_run:
            callbacks.append(
                pl_callbacks.DeviceStatsMonitor(cpu_stats=True)
            )

        if not script_cfg.run_cfg.dev_run:
            callbacks.append(
                pl_callbacks.EarlyStopping(
                    monitor='val/mel_loss', min_delta=0.0,
                    patience=3,
                    mode='min')
            )

        trainer = pl.Trainer(
            accelerator='auto',
            devices='auto',
            max_epochs=script_cfg.train_cfg.max_epochs,
            logger=[
                pl_loggers.MLFlowLogger(
                    experiment_name=script_cfg.run_cfg.mlflow_experiment,
                    run_name=script_cfg.run_cfg.mlflow_run_name,
                    run_id=run.info.run_id),
                pl_loggers.TensorBoardLogger(
                    save_dir='tensorboard',
                    name=f'{experiment.name}/{run.info.run_name}',
                    default_hp_metric=False,
                    version=0
                )
            ],
            callbacks=callbacks,
            num_sanity_val_steps=0,
            profiler=profiler,
            enable_checkpointing=True,
            check_val_every_n_epoch=script_cfg.train_cfg.val_every_n_epochs,
            limit_train_batches=limit_train_batches,
            limit_val_batches=limit_val_batches,
            limit_test_batches=limit_test_batches,
            log_every_n_steps=25,
            accumulate_grad_batches=script_cfg.train_cfg.accumulate_grad_batches,
            gradient_clip_val=script_cfg.train_cfg.gradient_clip_val
        )

        _logger().info('Starting training...')

        trainer.fit(model,
                    datamodule=data_module,
                    ckpt_path=script_cfg.run_cfg.continue_training_from_checkpoint)


if __name__ == '__main__':

    main()  # pylint: disable=no-value-for-parameter
