"""Runs Exploratory Data Analysis on raw LibriTTS-R dataset."""

import json
import logging
import pathlib

import hydra
import omegaconf

from paragraph_tts.data.eda import raw_eda
from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


@hydra.main(version_base=None, config_path='cfg', config_name='perform_eda')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Runs LibriTTS-R Exploratory Data Analysis."""

    logging_utils.setup_logging('perform_eda')

    _logger().info('Script configuration:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    output_dir = pathlib.Path(script_cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_ds_handler = raw_libri_dir_handler.RawLibriDirHandler(
        script_cfg.raw_ds_path,
        choose_splits=script_cfg.choose_splits
    )

    eda_processor = raw_eda.RawEDA(raw_ds_handler)

    _logger().info('Collecting speakers stats...')
    eda_processor.save_speakers_stats(output_dir / 'speakers_stats')


if __name__ == '__main__':
    main()  # pylint: disable=E1120
