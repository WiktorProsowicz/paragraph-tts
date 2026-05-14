"""Runs inference on evaluation dataset and saves the results."""

import logging
import pathlib
import os

import hydra
import omegaconf
import mlflow

from paragraph_tts.utils import logging_utils
from paragraph_tts.models.acoustic import acoustic as acoustic_module
from paragraph_tts.models.stl_predictor import stl_predictor as stl_predictor_module
from paragraph_tts.eval import inference_engine


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


@hydra.main(version_base=None, config_path='cfg', config_name='run_inference')
def main(cfg: omegaconf.DictConfig) -> None:
    """Runs inference on evaluation dataset and saves the results."""

    logging_utils.setup_logging('prepare_test_dataset')

    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    experiment = mlflow.set_experiment(cfg.acoustic_model_cfg.experiment_name)
    ckpt_path = os.path.join(cfg.mlflow_artifacts_root,
                             experiment.experiment_id,
                             cfg.acoustic_model_cfg.run_id,
                             'artifacts',
                             'checkpoints',
                             cfg.acoustic_model_cfg.ckpt_name)

    acoustic_model = acoustic_module.AcousticModel.load_from_checkpoint(
        ckpt_path,
        weights_only=False,
        visualize_n_batches=0,
        visualize_n_samples_per_batch=0,
        train_cfg=acoustic_module.TrainConfiguration(
            loss_weights={},
            loss_weight_decays={},
            wsv_bin_init_temperature=0.0,
            gst_bin_init_temperature=0.0,
            wsv_bin_temperature_decay=0.0,
            gst_bin_temperature_decay=0.0,
            wsv_bin_loss_start_epoch=0,
            wsv_bin_hard_start_epoch=0,
            gst_bin_loss_start_epoch=0
        ),
        optim_cfg=acoustic_module.OptimizerConfiguration(
            base_lr=0.0,
            prosody_enc_lr=0.0,
            betas=(0.0, 0.0),
            eps=0.0,
            base_weight_decay=0.0,
            prosody_enc_weight_decay=0.0,
            lr_scheduler_gamma=0.0
        )
    )

    if cfg.stl_predictor_cfg is not None:
        experiment = mlflow.set_experiment(cfg.stl_predictor_cfg.experiment_name)
        ckpt_path = os.path.join(cfg.mlflow_artifacts_root,
                                 experiment.experiment_id,
                                 cfg.stl_predictor_cfg.run_id,
                                 'artifacts',
                                 'checkpoints',
                                 cfg.stl_predictor_cfg.ckpt_name)

        stl_predictor_model = stl_predictor_module.STLPredictor.load_from_checkpoint(
            ckpt_path,
            visualize_n_batches=0,
            weights_only=False,
            visualize_n_samples_per_batch=0
        )
    else:
        stl_predictor_model = None

    eval_runner = inference_engine.InferenceEngine(
        acoustic_model=acoustic_model,
        stl_predictor_model=stl_predictor_model,
        energy_quantization_params=tuple(cfg.energy_quantization_params),
        pitch_quantization_params=tuple(cfg.pitch_quantization_params),
        inference_label=cfg.inference_label,
        inference_device=cfg.inference_device
    )

    eval_runner.run_inference(
        eval_ds_path=pathlib.Path(cfg.eval_ds_path),
        root_output_dir=pathlib.Path(cfg.root_output_dir)
    )


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
