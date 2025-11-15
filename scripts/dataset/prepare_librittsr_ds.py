# -*- coding: utf-8 -*-
"""Runs preprocessing on raw LibriTTS-R dataset and saves the preprocessed files."""
import json
import logging
import os

import hydra
import omegaconf
import tqdm

from paragraph_tts.data.preprocessing import (processor, utils as prep_utils)
from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import alignments_dir_handler
from paragraph_tts.utils.path import enriched_context_dir_handler
from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_librittsr_ds')
def main(script_cfg: omegaconf.DictConfig):
    """Runs LibriTTS-R preprocessing."""

    logging_utils.setup_logging('prepare_librittsr_ds')
    _logger().info('Config:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    raw_ds_path_hand = raw_libri_dir_handler.RawLibriDirHandler(
        script_cfg.raw_ds_path,
        choose_splits=script_cfg.filters.choose_splits)
    alignments_path_hand = alignments_dir_handler.AlignmentsDirHandler(
        script_cfg.alignments_path)

    if script_cfg.enriched_contexts_path is not None:
        enriched_contexts_path_hand = enriched_context_dir_handler.EnrichedContextDirHandler(
            script_cfg.enriched_contexts_path)

    else:
        enriched_contexts_path_hand = None

    os.makedirs(script_cfg.processed_ds_output_path, exist_ok=True)

    sample_filter_cfg = processor.SampleFilterCfg(
        max_words_in_utterance=script_cfg.filters.max_words_in_utterance,
        min_words_in_utterance=script_cfg.filters.min_words_in_utterance,
        allow_fragmented_sentences=script_cfg.filters.allow_fragmented_sentences,
        max_paragraph_length=script_cfg.filters.max_context_length,
        min_paragraph_length=script_cfg.filters.min_context_length,
        min_utterance_duration=script_cfg.filters.min_utterance_duration,
        max_utterance_duration=script_cfg.filters.max_utterance_duration,
        min_words_in_context=script_cfg.filters.min_words_in_context,
        max_words_in_context=script_cfg.filters.max_words_in_context
    )

    preprocessor = processor.LibriTTSRPreprocessor(raw_ds_path_hand,
                                                        enriched_contexts_path_hand,
                                                        alignments_path_hand,
                                                        script_cfg.processed_ds_output_path,
                                                        script_cfg.prepare_speaker_embeddings,
                                                        script_cfg.embedders_device,
                                                        sample_filter_cfg)

    speakers = list(raw_ds_path_hand.iter_speakers())

    _logger().info('Processing %d speakers', len(speakers))

    for speaker_id in tqdm.tqdm(speakers, desc='Speakers', unit='spk'):
        preprocessor.run_for_speaker(speaker_id)

    preprocessor.save_metadata(
        {
            'prepare_speaker_embeddings': script_cfg.prepare_speaker_embeddings,
            'embedders_device': script_cfg.embedders_device,
            'filters': omegaconf.OmegaConf.to_container(script_cfg.filters),
            'ds_stats': prep_utils.compose_processed_ds_stats(
                script_cfg.processed_ds_output_path
            )
        }
    )


if __name__ == '__main__':
    main()  # pylint: disable=E1120
