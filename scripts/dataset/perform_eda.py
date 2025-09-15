"""Runs Exploratory Data Analysis on raw LibriTTS-R dataset."""

import os
import logging
import yaml  # type: ignore

import hydra
import omegaconf
from paragraph_tts import (data, utils)

RESULTS_DESC = """
This directory contains results of EDA on LibriTTS-R dataset:

- overall_stats.yaml: Overall statistics collected during EDA.
- figures/: Directory with various figures visualizing the dataset.

Notes:
    - A paragraph is considered complete if it contains all utterances from 0 to N
      (where N is the highest utterance ID in the paragraph).
    - A paragraph is considered valid if it is complete and contains at least 3 utterances
      (begin, middle, end).
"""

@hydra.main(version_base=None, config_path='cfg', config_name='perform_eda')
def main(script_cfg: omegaconf.DictConfig):
    """Runs LibriTTS-R Exploratory Data Analysis."""

    utils.logging_utils.setup_logging()
    logging.getLogger('utils.path').setLevel(logging.INFO)

    if not os.path.exists(script_cfg.raw_ds_path):
        logging.critical('Cannot load raw dataset from a non-existing path: %s',
                         script_cfg.raw_ds_path)

    os.makedirs(script_cfg.output_dir, exist_ok=True)

    overall_stats = {}

    feature_extractor = data.eda.FeaturesExtractor(script_cfg.raw_ds_path)

    logging.info('Collecting speakers stats...')
    overall_stats['speakers_stats'] = feature_extractor.get_speakers_stats()

    logging.info('Collecting chapters stats...')
    overall_stats['chapters_stats'] = feature_extractor.get_chapters_stats()

    logging.info('Collecting paragraphs stats...')
    overall_stats['paragraphs_stats'] = feature_extractor.get_paragraphs_stats()

    logging.info('Collecting utterances stats...')
    overall_stats['utterances_stats'] = feature_extractor.get_utterances_stats()

    with open(os.path.join(script_cfg.output_dir, 'overall_stats.yaml'), 'w', encoding='utf-8') as stats_f:
        yaml.dump(overall_stats, stats_f)

    with open(os.path.join(script_cfg.output_dir, 'README.txt'), 'w', encoding='utf-8') as readme_f:
        readme_f.write(RESULTS_DESC)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
