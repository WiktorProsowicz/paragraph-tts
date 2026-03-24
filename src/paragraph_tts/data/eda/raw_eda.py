"""Contains utilities for performing Exploratory Data Analysis of LibriTTS-R DS."""

from typing import Any
from typing import Iterator
import pathlib

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from paragraph_tts.data import librittsr_helpers
from paragraph_tts.data.preprocessing import audio as audio_prep
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.data.eda import utils as eda_utils
from paragraph_tts.utils.path import alignments_dir_handler


class RawEDA:
    """Extracts various features from the dataset."""

    def __init__(self,
                 raw_ds_handler: raw_libri_dir_handler.RawLibriDirHandler,
                 alignments_handler: alignments_dir_handler.AlignmentsDirHandler):
        """Inits the extractor."""

        self._raw_path_handler = raw_ds_handler
        self._alignments_handler = alignments_handler
        self._raw_ds_meta = librittsr_helpers.LibriTTSRMetadata()

        sns.set_theme(style='darkgrid')

    def save_speakers_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics related to speakers to an output directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        utt_df = pd.DataFrame(self._get_utterances_df())

        per_speaker_stats = {
            'Is male': utt_df.groupby('spk_id')['spk_gender'].first() == 'M',

            'Num. utterances': utt_df.groupby('spk_id')['utt_id'].count(),

            'Num. paragraphs': (utt_df[['spk_id', 'chap_id', 'para_id']].drop_duplicates()
                                .groupby('spk_id').size()),

            'Total length (sec)': utt_df.groupby('spk_id')['length_sec'].sum(),

            'Num. utterances with audio': (utt_df[utt_df['length_sec'].notna()]
                                           .groupby('spk_id')['utt_id'].count()),

            'Num. paragraphs with any audio': (
                utt_df[['spk_id', 'chap_id', 'para_id', 'paragraph_has_any_wav']].drop_duplicates()
                .groupby('spk_id')['paragraph_has_any_wav'].sum()),

            'Num. paragraphs with all audio': (
                utt_df[['spk_id', 'chap_id', 'para_id', 'paragraph_has_all_wav']].drop_duplicates()
                .groupby('spk_id')['paragraph_has_all_wav'].sum())
        }

        other_stats = {
            'n_speakers': self._raw_path_handler.num_speakers,
            'n_books': utt_df['book_id'].nunique(),
            'n_male_speakers': utt_df[utt_df['spk_gender'] == 'M']['spk_id'].nunique(),
        }

        with output_dir.joinpath('primary_stats.json').open('w') as f:
            json.dump({
                **{k: v.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
                   for k, v in per_speaker_stats.items()},
                **other_stats
            },
                f, indent=4, ensure_ascii=False)

        self._save_per_speaker_distributions(per_speaker_stats, output_dir)

    def save_paragraph_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics related to paragraphs to an output directory."""

        utt_df = pd.DataFrame(self._get_utterances_df())

        self._save_paragraph_stats_for_df(utt_df, output_dir / 'all_paragraphs')

        self._save_paragraph_stats_for_df(utt_df[utt_df['paragraph_has_any_wav']],
                                          output_dir / 'paragraphs_with_any_audio')

        self._save_paragraph_stats_for_df(utt_df[utt_df['paragraph_has_all_wav']],
                                          output_dir / 'paragraphs_with_all_audio')

    def save_alignments_stats(self, output_dir: pathlib.Path) -> None:
        """Dumps statistics related to phoneme alignments to an output directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        align_df = pd.DataFrame(self._get_alignments_df())

        primary_stats = {
            'Num. phonemes': align_df['n_phonemes'],
            'Min. phoneme length (sec)': align_df['minimum_phoneme_length'],
            'Max. phoneme length (sec)': align_df['maximum_phoneme_length'],
            'Avg. phoneme length (sec)': align_df['avg_phoneme_length']
        }

        other_stats = {
            'Total num. utterances': len(align_df),
            'Total num. utterances with audio': int(align_df['has_sound'].sum()),
            'Total length (sec) of aligned audio': float(
                align_df[align_df['has_sound']]['length_sec'].sum()),
            'Num. utterances without alignment': int(align_df['n_phonemes'].isna().sum()),
            'Num. utterances with audio but without alignment': int(
                align_df[align_df['has_sound'] & align_df['n_phonemes'].isna()].shape[0]),
        }

        with output_dir.joinpath('primary_stats.json').open('w') as f:
            json.dump({
                **{k: v.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
                   for k, v in primary_stats.items()},
                **other_stats
            },
                f, indent=4, ensure_ascii=False)

    def _save_paragraph_stats_for_df(self,
                                     utt_df: pd.DataFrame,
                                     output_dir: pathlib.Path) -> None:
        """Dumps statistics related to paragraphs for a given DataFrame to an output directory."""

        output_dir.mkdir(parents=True, exist_ok=True)

        per_paragraph_stats = pd.DataFrame({
            'Length (sec)': utt_df.groupby(['spk_id', 'chap_id', 'para_id'])['length_sec'].sum(),
            'Num. utterances': utt_df.groupby(['spk_id', 'chap_id', 'para_id'])['utt_id'].count(),
            'Num. words': utt_df.groupby(['spk_id', 'chap_id', 'para_id'])['n_words'].sum(),
        })

        other_stats = {
            'Total num. utterances': len(utt_df),
            'Total length (sec)': float(utt_df['length_sec'].sum()),
            'Total num. utterances with audio': int(utt_df['length_sec'].notna().sum()),
        }

        with output_dir.joinpath('primary_stats.json').open('w') as f:
            json.dump({
                **per_paragraph_stats.describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99]).to_dict(),
                **other_stats
            },
                f, indent=4, ensure_ascii=False)

        output_dir.joinpath('distributions').mkdir(exist_ok=True)
        output_dir.joinpath('distributions_no_outliers').mkdir(exist_ok=True)
        output_dir.joinpath('only_outliers').mkdir(exist_ok=True)

        for col in ['Length (sec)']:
            self._save_per_paragraph_distributions_violin(per_paragraph_stats,
                                                          col,
                                                          output_dir / 'distributions')

            self._save_per_paragraph_distributions_violin(
                per_paragraph_stats[~eda_utils.is_outlier(per_paragraph_stats[col])],
                col,
                output_dir / 'distributions_no_outliers')

            self._save_per_paragraph_distributions_violin(
                per_paragraph_stats[eda_utils.is_outlier(per_paragraph_stats[col])],
                col,
                output_dir / 'only_outliers')

        for col in ['Num. utterances', 'Num. words']:
            self._save_per_paragraph_distributions_hist(per_paragraph_stats,
                                                        col,
                                                        output_dir / 'distributions')

            self._save_per_paragraph_distributions_hist(
                per_paragraph_stats[~eda_utils.is_outlier(per_paragraph_stats[col])],
                col,
                output_dir / 'distributions_no_outliers')

            self._save_per_paragraph_distributions_hist(
                per_paragraph_stats[eda_utils.is_outlier(per_paragraph_stats[col])],
                col,
                output_dir / 'only_outliers')

    def _save_per_speaker_distributions(self,
                                        per_speaker_stats: pd.DataFrame,
                                        output_dir: pathlib.Path) -> None:
        """Saves several per-speaker distributions as figures."""

        for col in ['Num. utterances', 'Num. paragraphs', 'Total length (sec)',
                    'Num. utterances with audio', 'Num. paragraphs with any audio',
                    'Num. paragraphs with all audio']:

            fig, axes = plt.subplots(figsize=(15, 6), ncols=2, sharey=True)

            sns.violinplot(data=per_speaker_stats,
                           y=col,
                           hue='Is male',
                           split=True,
                           inner='quart',
                           palette=eda_utils.get_gradient_palette(2),
                           fill=False,
                           ax=axes[0],
                           cut=0)

            axes[0].set_ylabel(col)

            sns.violinplot(data=per_speaker_stats,
                           y=col,
                           inner='quart',
                           fill=True,
                           color=eda_utils.MIDDLE_COLOR_STD,
                           ax=axes[1],
                           cut=0)

            axes[0].xaxis.grid(True)
            axes[0].set_axisbelow(True)

            fig.suptitle(f'Distribution of {col} per speaker')
            fig.subplots_adjust(top=.95)
            fig.savefig(output_dir.joinpath(f'{col}.svg'))

    def _save_per_paragraph_distributions_violin(self,
                                                 per_paragraph_stats: pd.DataFrame,
                                                 column: str,
                                                 output_dir: pathlib.Path) -> None:
        """Saves distribution of per-paragraph feature in a violin plot."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.violinplot(data=per_paragraph_stats,
                       y=column,
                       inner='quart',
                       fill=True,
                       color=eda_utils.MIDDLE_COLOR_STD,
                       ax=ax,
                       cut=0)

        ax.set_title(
            f'Distribution of {column} per paragraph ({len(per_paragraph_stats)} samples)')
        ax.set_ylabel(column)
        ax.xaxis.grid(True)
        ax.set_axisbelow(True)

        fig.subplots_adjust(top=.95)
        fig.savefig(output_dir.joinpath(f'{column}.svg'))

    def _save_per_paragraph_distributions_hist(self,
                                               per_paragraph_stats: pd.DataFrame,
                                               column: str,
                                               output_dir: pathlib.Path) -> None:
        """Saves distribution of per-paragraph feature in a histogram."""

        fig, ax = plt.subplots(figsize=(10, 6))

        sns.histplot(data=per_paragraph_stats, x=column, ax=ax,
                     discrete=True, color=eda_utils.MIDDLE_COLOR_STD)

        ax.set_title(f'Distribution of {column} per paragraph ({len(per_paragraph_stats)} samples)')
        ax.set_xlabel(column)
        ax.xaxis.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
        ax.set_axisbelow(True)

        fig.subplots_adjust(top=.95)
        fig.savefig(output_dir.joinpath(f'{column}.svg'))

    def _get_utterances_df(self) -> Iterator[dict[str, Any]]:
        """Returns a DataFrame with utterance-level data."""

        for spk_id in self._raw_path_handler.iter_speakers():
            for para_info in self._raw_path_handler.iter_paragraphs(spk_id):

                spk_info, book_info = self._raw_ds_meta.get_speaker_and_book(spk_id,
                                                                             para_info.chap_id)

                for utt in para_info.utterances:

                    row: dict[str, Any] = {
                        'spk_id': spk_id,
                        'chap_id': para_info.chap_id,
                        'para_id': para_info.para_id,
                        'utt_id': utt.utt_id,
                        'n_words': len(utt.normalized_text.split(' ')),
                        'spk_gender': spk_info.gender,
                        'book_id': book_info.id,
                        'paragraph_has_any_wav': any(u.wav_path is not None
                                                     for u in para_info.utterances),
                        'paragraph_has_all_wav': all(u.wav_path is not None
                                                     for u in para_info.utterances)
                    }

                    if utt.wav_path is not None:
                        row['length_sec'] = audio_prep.length_in_sec_of_file(utt.wav_path)

                    yield row

    def _get_alignments_df(self) -> Iterator[dict[str, Any]]:
        """Returns a DataFrame with information about phoneme alignments."""

        for spk_id in self._raw_path_handler.iter_speakers():
            for utt_info in self._raw_path_handler.iter_utterances_for_spk(spk_id):

                row = {
                    'spk_id': spk_id,
                    'chap_id': utt_info.chap_id,
                    'para_id': utt_info.para_id,
                    'utt_id': utt_info.utt_id,
                    'has_sound': utt_info.wav_path is not None
                }

                if self._alignments_handler.has_alignment_for(utt_info):

                    alignment = self._alignments_handler.get_alignment_for(utt_info)
                    intervals = alignment.get_tier_by_name('phones').intervals

                    row['n_phonemes'] = len(intervals)
                    row['length_sec'] = intervals[-1].end_time
                    row['minimum_phoneme_length'] = min(interval.end_time - interval.start_time
                                                        for interval in intervals)
                    row['maximum_phoneme_length'] = max(interval.end_time - interval.start_time
                                                        for interval in intervals)
                    row['avg_phoneme_length'] = row['length_sec'] / row['n_phonemes']

                yield row
