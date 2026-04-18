"""Runs preprocessing on raw LibriTTS-R dataset and saves the preprocessed files."""
import json
import logging

import hydra
import omegaconf
import pathlib

from paragraph_tts.data.preprocessing import processor
from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import alignments_dir_handler
from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_librittsr_ds')
def main(script_cfg: omegaconf.DictConfig) -> None:
    """Runs LibriTTS-R preprocessing."""

    logging_utils.setup_logging('prepare_librittsr_ds')
    _logger().info('Config:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    raw_ds_path_hand = raw_libri_dir_handler.RawLibriDirHandler(
        script_cfg.raw_ds_path,
        choose_splits=script_cfg.choose_splits)
    alignments_path_hand = alignments_dir_handler.AlignmentsDirHandler(
        script_cfg.alignments_path)

    ds_processor = processor.LibriTTSRProcessor(
        processor.LibriTTSRProcessor.Configuration.model_validate(
            omegaconf.OmegaConf.to_container(script_cfg.processor_cfg)
        )
    )

    ds_processor.process_dataset(
        raw_ds_handler=raw_ds_path_hand,
        alignments_handler=alignments_path_hand,
        output_dir=pathlib.Path(script_cfg.output_dir)
    )


if __name__ == '__main__':
    main()  # pylint: disable=E1120
