"""Runs Exploratory Data Analysis on raw LibriTTS-R dataset."""
import datetime
import json
import logging
import os

import hydra
import omegaconf
import soundfile
import yaml
from torch_dev_utils.tts import text_prep

from paragraph_tts import data
from paragraph_tts import utils
from paragraph_tts.data import librittsr_helpers

RESULTS_DESC = """
Generated at {time}.
This directory contains results of EDA on LibriTTS-R dataset:

- overall_stats.yaml: Overall statistics collected during EDA.
- figures/: Directory with various figures visualizing the dataset.
- example_paragraphs/: Directory with example paragraphs from the dataset.
- outlier_utterances/: Directory with outlier utterances based on various statistics (e.g word count)
- outlier_paragraphs/: Directory with outlier paragraphs based on various statistics (e.g word count)

Notes:
    - A paragraph is considered complete if it contains all utterances from 0 to N
      (where N is the highest utterance ID in the paragraph).
    - A paragraph is considered valid if it is complete and contains at least 3 utterances
      (begin, middle, end).
"""


def _logger():
    return logging.getLogger(__name__)


def _save_example_paragraphs(feature_extractor: data.eda.FeaturesExtractor,
                             output_dir: str):

    _logger().info('Saving example paragraphs...')

    libri_metadata = librittsr_helpers.LibriTTSRMetadata()

    for example_type, paragraphs in feature_extractor.get_example_paragraphs().items():

        examples_dir = os.path.join(
            output_dir, 'example_paragraphs', example_type)
        os.makedirs(examples_dir, exist_ok=True)

        for para_idx, para_info in enumerate(paragraphs):

            speaker, book = libri_metadata.get_speaker_and_book(
                para_info.spk_id, para_info.chap_id)

            para_path = os.path.join(
                examples_dir, f'paragraph_{para_idx:03d}.txt')
            with open(para_path, 'w', encoding='utf-8') as para_f:
                para_f.write(f'Paragraph ID: {para_info.para_id}\n')
                para_f.write(f'Chapter ID: {para_info.chap_id}\n')
                para_f.write(f'Speaker: {speaker.name}\n')
                para_f.write(f'Book: {book.title}\n')
                para_f.write(
                    f'Is valid: {data.eda.is_paragraph_valid(para_info)}\n')
                para_f.write(f'Is complete: {para_info.is_complete}\n')

                for line in feature_extractor.paragraph_as_lines(para_info):
                    para_f.write(f'{line}\n')

                if data.eda.is_paragraph_valid(para_info):
                    wav = feature_extractor.paragraph_as_waveform(para_info)
                    wav_path = os.path.join(
                        examples_dir, f'paragraph_{para_idx:03d}.wav')
                    soundfile.write(wav_path, wav, samplerate=22050)


def _save_outlier_utterances(feature_extractor: data.eda.FeaturesExtractor,
                             output_dir: str):

    _logger().info('Saving outlier utterances...')
    outliers = feature_extractor.get_outlier_utterances()

    for stat_type in outliers:
        for outlier_type, utterances in outliers[stat_type].items():

            outliers_dir = os.path.join(
                output_dir, 'outlier_utterances', stat_type)
            os.makedirs(outliers_dir, exist_ok=True)

            for utt_idx, utt_info in enumerate(utterances):
                utt_path = os.path.join(
                    outliers_dir, f'{outlier_type}_{utt_idx:03d}.txt')
                wav_path = os.path.join(
                    outliers_dir, f'{outlier_type}_{utt_idx:03d}.wav')

                text = text_prep.TextProcessor.load_text(utt_info.text_path)

                with open(utt_info.wav_path, 'rb') as src_wav_f:
                    wav_data = src_wav_f.read()

                with open(utt_path, 'w', encoding='utf-8') as utt_f:
                    utt_f.write(f'Utterance ID: {utt_info.utt_id}\n')
                    utt_f.write(f'Text: {text}\n')

                with open(wav_path, 'wb') as dst_wav_f:
                    dst_wav_f.write(wav_data)


def _save_outlier_paragraphs(feature_extractor: data.eda.FeaturesExtractor,
                             output_dir: str):

    _logger().info('Saving outlier paragraphs...')

    outliers = feature_extractor.get_outliers_original_paragraphs()

    for stat_type, paragraphs in outliers.items():

        outliers_dir = os.path.join(
            output_dir, 'outlier_paragraphs', stat_type)
        os.makedirs(outliers_dir, exist_ok=True)

        for para_idx, para_info in enumerate(paragraphs):

            para_path = os.path.join(
                outliers_dir, f'paragraph_{para_idx:03d}.txt')
            with open(para_path, 'w', encoding='utf-8') as para_f:

                para_f.write(f'Paragraph ID: {para_info.para_id}\n')
                para_f.write(f'Chapter ID: {para_info.chap_id}\n')
                para_f.write(f'Speaker ID: {para_info.spk_id}\n')

                for sent_idx, sentence in para_info.sentences.items():
                    para_f.write(f'{sent_idx}: {sentence}\n')


@hydra.main(version_base=None, config_path='cfg', config_name='perform_eda')
def main(script_cfg: omegaconf.DictConfig):
    """Runs LibriTTS-R Exploratory Data Analysis."""

    utils.logging_utils.setup_logging('perform_eda')

    _logger().info('Script configuration:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    if not os.path.exists(script_cfg.raw_ds_path):
        _logger().critical('Cannot load raw dataset from a non-existing path: %s',
                           script_cfg.raw_ds_path)

    os.makedirs(script_cfg.output_dir, exist_ok=True)

    overall_stats = {}

    feature_extractor = data.eda.FeaturesExtractor(script_cfg.raw_ds_path,
                                                   choose_splits=script_cfg.choose_splits)

    _logger().info('Collecting speakers stats...')
    overall_stats['speakers_stats'] = feature_extractor.get_speakers_stats()

    _logger().info('Collecting chapters stats...')
    overall_stats['chapters_stats'] = feature_extractor.get_chapters_stats()

    _logger().info('Collecting paragraphs stats...')
    overall_stats['paragraphs_stats'] = feature_extractor.get_paragraphs_stats()

    _logger().info('Collecting utterances stats...')
    overall_stats['utterances_stats'] = feature_extractor.get_utterances_stats()

    _logger().info('Collecting original paragraphs stats...')
    overall_stats['original_paragraphs_stats'] = feature_extractor.get_original_paragraphs_stats()

    overall_stats_path = os.path.join(
        script_cfg.output_dir, 'overall_stats.yaml')
    with open(overall_stats_path, 'w', encoding='utf-8') as stats_f:
        yaml.dump(overall_stats, stats_f)

    _logger().info('Saving figures...')
    figures_dir = os.path.join(script_cfg.output_dir, 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    for name, fig in feature_extractor.get_figures().items():
        fig_path = os.path.join(figures_dir, f'{name}.png')
        fig.savefig(fig_path, format='png', bbox_inches='tight')
        _logger().info('Saved figure: %s', fig_path)

    _save_example_paragraphs(feature_extractor, script_cfg.output_dir)

    _save_outlier_utterances(feature_extractor, script_cfg.output_dir)

    _save_outlier_paragraphs(feature_extractor, script_cfg.output_dir)

    with open(os.path.join(script_cfg.output_dir, 'README.txt'), 'w', encoding='utf-8') as readme_f:
        readme_f.write(RESULTS_DESC.format(
            time=datetime.datetime.now(
                datetime.timezone.utc).strftime('%Y-%m-%d %H:%M:%S %Z')
        ))


if __name__ == '__main__':
    main()  # pylint: disable=E1120
