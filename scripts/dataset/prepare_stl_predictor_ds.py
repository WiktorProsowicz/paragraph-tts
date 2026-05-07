"""Processes and saves the STL predictor dataset."""

import logging
import pathlib
import os

import hydra
import mlflow
import omegaconf

from comp_trans_tts.model import modules as ctt_modules

from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.models import acoustic as acoustic_models
from paragraph_tts.data.preprocessing import stl_predictor_ds_processor


def _logger() -> logging.Logger:
    """Returns the logger for the module."""
    return logging.getLogger('paragraph_tts')


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_stl_predictor_ds')
def main(cfg: omegaconf.DictConfig) -> None:
    """The main function for preparing the STL predictor dataset."""

    logging_utils.setup_logging('prepare_stl_predictor_ds')

    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)

    acoustic_ds_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(
        pathlib.Path(cfg.acoustic_ds_path))

    inference_resources: dict[str, stl_predictor_ds_processor.AcousticInferenceResources] = {}

    for resource_name, resource_cfg in cfg.inference_resources.items():

        experiment = mlflow.set_experiment(resource_cfg.experiment_name)

        ckpt_path = os.path.join(cfg.mlflow_artifacts_root,
                                 experiment.experiment_id,
                                 resource_cfg.run_id,
                                 'artifacts',
                                 'checkpoints',
                                 resource_cfg.ckpt_name)

        acoustic_model = acoustic_models.AcousticModel.load_from_checkpoint(
            ckpt_path,
            visualize_n_batches=0,
            visualize_n_samples_per_batch=0,
            weights_only=False
        )
        acoustic_model.eval()

        inference_resources[resource_name] = stl_predictor_ds_processor.AcousticInferenceResources(
            acoustic_model=acoustic_model,
            gst_bin_params=ctt_modules.StlBinarizationParams(**resource_cfg.gst_bin_params),
            wsv_bin_params=ctt_modules.StlBinarizationParams(**resource_cfg.wsv_bin_params)
        )

    processor = stl_predictor_ds_processor.STLPredictorDatasetProcessor(
        acoustic_ds_handler=acoustic_ds_handler,
        inference_resources=inference_resources,
        embedder_device=cfg.embedder_device
    )

    processor.process(pathlib.Path(cfg.output_dir))

    with pathlib.Path(cfg.output_dir).joinpath('cfg.yaml').open('w', encoding='utf-8') as f:
        omegaconf.OmegaConf.save(config=cfg, f=f)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
