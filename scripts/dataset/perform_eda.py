"""Runs Exploratory Data Analysis on raw LibriTTS-R dataset."""

import os
import logging
import datetime
import yaml  # type: ignore

import soundfile
import hydra
import omegaconf
from paragraph_tts import (data, utils)

RESULTS_DESC = """
Generated at {time}.
This directory contains results of EDA on LibriTTS-R dataset:

- overall_stats.yaml: Overall statistics collected during EDA.
- figures/: Directory with various figures visualizing the dataset.

Notes:
    - A paragraph is considered complete if it contains all utterances from 0 to N
      (where N is the highest utterance ID in the paragraph).
    - A paragraph is considered valid if it is complete and contains at least 3 utterances
      (begin, middle, end).
"""


def _save_example_paragraphs(feature_extractor: data.eda.FeaturesExtractor,
                             output_dir: str):

    logging.info('Saving example paragraphs...')
    examples_dir = os.path.join(output_dir, 'example_paragraphs')
    os.makedirs(examples_dir, exist_ok=True)

    for para_idx, para_info in enumerate(feature_extractor.get_example_paragraphs()):
        para_path = os.path.join(examples_dir, f'paragraph_{para_idx:03d}.txt')
        with open(para_path, 'w', encoding='utf-8') as para_f:
            para_f.write(f'Paragraph ID: {para_info.para_id}\n')
            para_f.write(f'Chapter ID: {para_info.chap_id}\n')
            para_f.write(f'Is valid: {data.eda.is_paragraph_valid(para_info)}\n')
            para_f.write(f'Is complete: {para_info.is_complete}\n')

            for line in feature_extractor.paragraph_as_lines(para_info):
                para_f.write(f'{line}\n')

            if data.eda.is_paragraph_valid(para_info):
                wav = feature_extractor.paragraph_as_waveform(para_info)
                wav_path = os.path.join(examples_dir, f'paragraph_{para_idx:03d}.wav')
                soundfile.write(wav_path, wav, samplerate=22050)


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

    logging.info('Saving figures...')
    figures_dir = os.path.join(script_cfg.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for name, fig in feature_extractor.get_figures().items():
        fig_path = os.path.join(figures_dir, f'{name}.png')
        fig.savefig(fig_path, format='png', bbox_inches='tight')
        logging.info('Saved figure: %s', fig_path)

    _save_example_paragraphs(feature_extractor, script_cfg.output_dir)

    with open(os.path.join(script_cfg.output_dir, 'README.txt'), 'w', encoding='utf-8') as readme_f:
        readme_f.write(RESULTS_DESC.format(
            time=datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        ))


if __name__ == '__main__':
    main()  # pylint: disable=E1120
