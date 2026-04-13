"""Runs analysis of the processed dataset."""

import json
import logging
import pathlib
import yaml

import hydra
import omegaconf

from paragraph_tts.utils import logging_utils
from paragraph_tts.data.eda import processed_ds_analysis
from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.data.preprocessing import processor as ds_processor


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


@hydra.main(version_base=None, config_path='cfg', config_name='perform_processed_ds_analysis')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Runs analysis of the processed LibriTTS-R dataset."""

    logging_utils.setup_logging('perform_processed_ds_analysis')

    _logger().info('Script configuration:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    output_dir = pathlib.Path(script_cfg.output_dir)

    ds_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(
        pathlib.Path(script_cfg.processed_ds_dir)
    )

    with open(script_cfg.processor_cfg_path, 'r', encoding='utf-8') as f:
        processor_cfg = ds_processor.LibriTTSRProcessor.Configuration(**yaml.safe_load(f))

    analyzer = processed_ds_analysis.ProcessedDSAnalyzer(
        ds_handler=ds_handler,
        processor_cfg=processor_cfg
    )

    _logger().info('Collecting speakers stats...')
    analyzer.save_stats(output_dir)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
