# -*- coding: utf-8 -*-
"""Runs preprocessing on raw LibriTTS-R dataset and saves the preprocessed files."""

import json
import logging
import os

import tqdm
import hydra
import omegaconf

from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import (
    raw_libri_dir_handler,
    enriched_context_dir_handler,
    alignments_dir_handler)
from paragraph_tts import data


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_librittsr_ds')
def main(script_cfg: omegaconf.DictConfig):
    """Runs LibriTTS-R preprocessing."""

    logging_utils.setup_logging('prepare_librittsr_ds')
    _logger().info('Config:\n%s', json.dumps(dict(script_cfg), indent=4))

    raw_ds_path_hand = raw_libri_dir_handler.RawLibriDirHandler(script_cfg.raw_ds_path)
    alignments_path_hand = alignments_dir_handler.AlignmentsDirHandler(
        script_cfg.alignments_path)

    if script_cfg.enriched_contexts_path is not None:
        enriched_contexts_path_hand = enriched_context_dir_handler.EnrichedContextDirHandler(
            script_cfg.enriched_contexts_path)

    else:
        enriched_contexts_path_hand = None

    os.makedirs(script_cfg.processed_ds_output_path, exist_ok=True)

    preprocessor = data.processor.LibriTTSRPreprocessor(raw_ds_path_hand,
                                                        enriched_contexts_path_hand,
                                                        alignments_path_hand,
                                                        script_cfg.processed_ds_output_path,
                                                        script_cfg.prepare_speaker_embeddings,
                                                        script_cfg.embedders_device)

    all_speakers = list(raw_ds_path_hand.iter_speakers())

    _logger().info('Processing %d speakers', len(all_speakers))

    for speaker_id in tqdm.tqdm(all_speakers, desc='Speakers', unit='spk'):
        preprocessor.run_for_speaker(speaker_id)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
