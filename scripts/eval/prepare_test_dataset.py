"""Processes and saves the test dataset for evaluation."""

import logging
import pathlib
import os

import hydra
import omegaconf

from comp_trans_tts import deepspeaker
from paragraph_tts.utils import logging_utils
from paragraph_tts.eval import test_ds_processor
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.utils.path import alignments_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger('paragraph_tts')


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_test_dataset')
def main(cfg: omegaconf.DictConfig) -> None:
    """Processes and saves the test dataset for evaluation."""

    logging_utils.setup_logging('prepare_test_dataset')

    _logger().info('Script cfg:\n%s', omegaconf.OmegaConf.to_yaml(cfg))

    alignments_handler = alignments_dir_handler.AlignmentsDirHandler(cfg.alignments_dir)
    raw_ds_handler = raw_libri_dir_handler.RawLibriDirHandler(cfg.raw_ds_path,
                                                              choose_splits=cfg.choose_splits)
    spk_embedder = deepspeaker.embedder.DeepSpeakerEmbedder(cfg.processor_cfg.embedder_device)

    processor = test_ds_processor.TestDsProcessor(
        test_ds_processor.TestDsProcessor.Configuration(
            bert_model_tag=cfg.processor_cfg.bert_model_tag,
            embedder_device=cfg.processor_cfg.embedder_device,
            spec_frames_per_second=cfg.processor_cfg.spec_frames_per_second
        ),
        raw_ds_handler=raw_ds_handler,
        alignments_handler=alignments_handler,
        spk_embedder=spk_embedder,
        paragraphs_spec=[test_ds_processor.ParagraphSpec(**para_spec)
                         for para_spec in cfg.paragraphs_spec]
    )

    processor.prepare_dataset(pathlib.Path(cfg.output_dir))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
