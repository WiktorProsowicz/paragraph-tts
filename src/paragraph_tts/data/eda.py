"""Contains utilities for performing Exploratory Data Analysis of LibriTTS-R DS."""
import itertools
import random
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import TypeAlias

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from torch_dev_utils.tts import text_prep

from paragraph_tts.data import librittsr_helpers
from paragraph_tts.data.preprocessing import audio as audio_prep
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.utils.path.raw_libri_dir_handler import OriginalParagraph
from paragraph_tts.utils.path.raw_libri_dir_handler import ParagraphInfo
from paragraph_tts.utils.path.raw_libri_dir_handler import UtteranceInfo

_FixedLengthArrayType: TypeAlias = List[float] | List[int] | np.ndarray


def is_paragraph_valid(para_info: ParagraphInfo) -> bool:
    """Returns True if the paragraph is valid."""
    return para_info.is_complete and len(para_info.utterances) >= 3


def _calculate_basic_numerical_stats(data: _FixedLengthArrayType) -> Dict[str, float]:
    """Calculates basic numerical statistics for a list of numbers."""

    return {
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'avg': float(np.mean(data)),
        'std': float(np.std(data)),
        'q1': float(np.percentile(data, 25)),
        'q2': float(np.percentile(data, 50)),
        'q3': float(np.percentile(data, 75)),
        'count': int(len(data))
    }


def _is_outlier_lower(value: float | int, q1: float, q3: float) -> bool:
    """Tells if the value is a lower outlier based on IQR rule."""
    return value < (q1 - 1.5 * (q3 - q1))


def _is_upper_outlier(value: float | int, q1: float, q3: float) -> bool:
    """Tells if the value is an upper outlier based on IQR rule."""
    return value > (q3 + 1.5 * (q3 - q1))


def _is_outlier(value: float | int, q1: float, q3: float) -> bool:
    """Tells if the value is an outlier based on IQR rule."""
    return _is_outlier_lower(value, q1, q3) or _is_upper_outlier(value, q1, q3)


class FeaturesExtractor:
    """Extracts various features from the dataset."""

    def __init__(self, raw_ds_path: str, choose_splits: List[str]):
        """Inits the extractor.

        Args:
            raw_ds_path: Path to LibriTTS-R raw dataset.
        """

        self._raw_ds_path = raw_ds_path
        self._raw_path_handler = raw_libri_dir_handler.RawLibriDirHandler(
            raw_ds_path,
            choose_splits=choose_splits
        )
        self._audio_processor = audio_prep.AudioProcessor(
            sr=22050,
            hop_length=256,
            win_length=1025,
            n_mels=80,
            fmin=0,
            fmax=8000,
            trim_top_db=23
        )
        self._text_processor = text_prep.TextProcessor()

    def paragraph_as_lines(self, para_info: ParagraphInfo) -> List[str]:
        """Returns the paragraph as a list of lines (strings)."""

        lines = []

        for utt in para_info.utterances:
            text = self._text_processor.load_text(utt.text_path)
            text = self._text_processor.clean_text(text)
            lines.append(f'{utt.utt_id}: {text}')

        return lines

    def paragraph_as_waveform(self, para_info: ParagraphInfo) -> np.ndarray:
        """Returns the paragraph as a concatenated waveform."""

        waveforms = []

        for utt in para_info.utterances:
            wav = self._audio_processor.load_wav_raw(utt.wav_path)
            waveforms.append(wav)

        return np.concatenate(waveforms, axis=0)

    def get_speakers_stats(self) -> Dict[str, Any]:
        """Returns stats related to speakers."""

        speakers_stats: Dict[str, Any] = {}

        speakers_stats['num_speakers'] = self._raw_path_handler.num_speakers

        utt_stats_per_speaker = self._get_stats_per_speaker()

        speakers_stats['utterances_per_speaker'] = _calculate_basic_numerical_stats(
            utt_stats_per_speaker['num_utt']
        )

        speakers_stats['total_duration_per_speaker (minutes)'] = _calculate_basic_numerical_stats(
            np.array(utt_stats_per_speaker['total_dur']) / 60.0
        )

        speakers_stats['paragraphs_per_speaker'] = _calculate_basic_numerical_stats(
            utt_stats_per_speaker['num_paragraphs']
        )

        return speakers_stats

    def get_chapters_stats(self) -> Dict[str, Any]:
        """Returns stats related to chapters."""

        chapters_stats: Dict[str, Any] = {}

        chapters_stats['num_chapters'] = len(
            list(self._raw_path_handler.iter_chapters()))

        n_paragraphs_in_chapter = {}
        n_utterances_in_chapter = {}

        for spk_id in self._raw_path_handler.iter_speakers():
            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                n_paragraphs_in_chapter[chap_id] = 0
                n_utterances_in_chapter[chap_id] = 0

                for para_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_paragraphs_in_chapter[chap_id] += 1
                    n_utterances_in_chapter[chap_id] += len(
                        para_info.utterances)

        chapters_stats['paragraphs_per_chapter'] = _calculate_basic_numerical_stats(
            list(n_paragraphs_in_chapter.values())
        )

        chapters_stats['utterances_per_chapter'] = _calculate_basic_numerical_stats(
            list(n_utterances_in_chapter.values())
        )

        return chapters_stats

    def get_paragraphs_stats(self) -> Dict[str, Any]:
        """Returns stats related to paragraphs."""

        paragraphs_stats: Dict[str, Any] = {}

        n_utterances_in_paragraph = []
        n_utterances_in_valid_paragraph = []
        n_utterances_per_complete_paragraph = []
        n_incomplete_paragraphs = 0
        n_paragraphs = 0
        n_valid_paragraphs = 0  # Is complete and contains at least 3 utterances

        for spk_id in self._raw_path_handler.iter_speakers():
            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                for para_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_utterances_in_paragraph.append(len(para_info.utterances))

                    if is_paragraph_valid(para_info):
                        n_utterances_in_valid_paragraph.append(
                            len(para_info.utterances))

                    if para_info.is_complete:
                        n_utterances_per_complete_paragraph.append(
                            len(para_info.utterances))

                    if not para_info.is_complete:
                        n_incomplete_paragraphs += 1

                    if is_paragraph_valid(para_info):
                        n_valid_paragraphs += 1

                    n_paragraphs += 1

        paragraphs_stats['utterances_per_paragraph'] = _calculate_basic_numerical_stats(
            n_utterances_in_paragraph
        )

        paragraphs_stats['utterances_per_valid_paragraph'] = _calculate_basic_numerical_stats(
            n_utterances_in_valid_paragraph
        )

        paragraphs_stats['utterances_per_complete_paragraph'] = _calculate_basic_numerical_stats(
            n_utterances_per_complete_paragraph
        )

        paragraphs_stats['num_incomplete_paragraphs'] = n_incomplete_paragraphs
        paragraphs_stats['num_paragraphs'] = n_paragraphs
        paragraphs_stats['num_valid_paragraphs'] = n_valid_paragraphs

        return paragraphs_stats

    def get_original_paragraphs_stats(self) -> Dict[str, Any]:
        """Returns stats related to original paragraphs (entire paragraphs from books)."""

        stats_dict: Dict[str, Any] = {}

        numerical_stats = self._get_stats_per_original_paragraph()

        stats_dict['num_original_paragraphs'] = len(
            numerical_stats['n_words_in_paragraph'])

        stats_dict['num_words_per_paragraph'] = _calculate_basic_numerical_stats(
            numerical_stats['n_words_in_paragraph']
        )

        stats_dict['num_sentences_per_paragraph'] = _calculate_basic_numerical_stats(
            numerical_stats['n_sentences_in_paragraph']
        )

        stats_dict['num_words_per_sentence'] = _calculate_basic_numerical_stats(
            numerical_stats['n_words_in_sentence']
        )

        stats_dict['num_paras_with_1_sentence'] = sum(
            1 for n_sent in numerical_stats['n_sentences_in_paragraph'] if n_sent == 1
        )

        stats_dict['num_paras_with_2_sentences'] = sum(
            1 for n_sent in numerical_stats['n_sentences_in_paragraph'] if n_sent == 2
        )

        return stats_dict

    def get_utterances_stats(self) -> Dict[str, Any]:
        """Returns stats related to utterances."""

        utterances_stats: Dict[str, Any] = {}

        stats_per_utterance = self._get_stats_per_utterance()

        utterances_stats['num_utterances'] = len(
            stats_per_utterance['word_counts'])

        utterances_stats['words_per_utterance'] = _calculate_basic_numerical_stats(
            stats_per_utterance['word_counts'])

        utterances_stats['length_sec_per_utterance'] = _calculate_basic_numerical_stats(
            stats_per_utterance['lengths_sec']
        )

        utterances_stats['num_utterances_with_isolated_punctuations'] = sum(
            stats_per_utterance['has_single_punctuations'])

        utterances_stats['num_utterances_which_are_whole_sentences'] = sum(
            stats_per_utterance['is_whole_sentence'])

        total_dur = np.sum(stats_per_utterance['lengths_sec']) / 3600.0
        utterances_stats['total_duration (hours)'] = float(total_dur)

        n_outliers_word_counts = sum(
            _is_outlier(word_count,
                        utterances_stats['words_per_utterance']['q1'],
                        utterances_stats['words_per_utterance']['q3'])
            for word_count in stats_per_utterance['word_counts']
        )

        n_outliers_lengths_sec = sum(
            _is_outlier(length_sec,
                        utterances_stats['length_sec_per_utterance']['q1'],
                        utterances_stats['length_sec_per_utterance']['q3'])
            for length_sec in stats_per_utterance['lengths_sec']
        )

        utterances_stats['num_outliers'] = {
            'word_count': n_outliers_word_counts,
            'length_sec': n_outliers_lengths_sec
        }

        return utterances_stats

    def get_figures(self) -> Dict[str, Any]:
        """Returns figures related to the dataset."""

        figures = {}

        figures.update(self._get_speaker_figures())

        stats_per_utterance = self._get_stats_per_utterance()

        bins = min(100, len(stats_per_utterance['word_counts']))

        fig, ax = plt.subplots()
        ax.hist(stats_per_utterance['word_counts'], bins=bins)
        ax.set_title('Number of words per utterance')
        ax.set_xlabel('Number of words')
        ax.set_ylabel('Number of utterances')
        ax.axvline(x=np.mean(stats_per_utterance['word_counts']), color='red', linestyle='--',
                   label='Mean')
        ax.legend()

        figures['num_words_per_utterance'] = fig

        fig, ax = plt.subplots()
        ax.hist(stats_per_utterance['lengths_sec'], bins=bins)
        ax.set_title('Length (seconds) per utterance')
        ax.set_xlabel('Length (seconds)')
        ax.set_ylabel('Number of utterances')
        ax.axvline(x=np.mean(stats_per_utterance['lengths_sec']), color='red', linestyle='--',
                   label='Mean')
        ax.legend()

        figures['length_sec_per_utterance'] = fig

        stats_per_orig_para = self._get_stats_per_original_paragraph()

        bins = min(
            100, len(set(stats_per_orig_para['n_sentences_in_paragraph'])))
        fig, ax = plt.subplots()
        ax.hist(stats_per_orig_para['n_sentences_in_paragraph'], bins=bins)
        ax.set_title('Number of sentences per original paragraph')
        ax.set_xlabel('Number of sentences')
        ax.set_ylabel('Number of original paragraphs')
        ax.axvline(x=np.mean(stats_per_orig_para['n_sentences_in_paragraph']),
                   color='red', linestyle='--', label='Mean')
        ax.legend()

        figures['num_sentences_per_original_paragraph'] = fig

        bins = min(100, len(set(stats_per_orig_para['n_words_in_paragraph'])))
        fig, ax = plt.subplots()
        ax.hist(stats_per_orig_para['n_words_in_paragraph'], bins=bins)
        ax.set_title('Number of words per original paragraph')
        ax.set_xlabel('Number of words')
        ax.set_ylabel('Number of original paragraphs')
        ax.axvline(x=np.mean(stats_per_orig_para['n_words_in_paragraph']),
                   color='red', linestyle='--', label='Mean')
        ax.legend()

        figures['num_words_per_original_paragraph'] = fig

        return figures

    def get_example_paragraphs(self) -> Dict[str, Iterator[ParagraphInfo]]:
        """Returns example paragraphs from the dataset.

        The paragraphs are selected from the following groups:
            - Valid paragraphs
            - Invalid paragraphs
            - Paragraphs containing at least one non-whole sentence utterance
        """

        def contains_non_whole_sentence(para_info: ParagraphInfo) -> bool:
            for utt in para_info.utterances:
                text = self._text_processor.load_text(utt.text_path)
                text = self._text_processor.clean_text(text)

                if not librittsr_helpers.is_sentence_whole(text):
                    return True

            return False

        all_paragraphs = list(self._raw_path_handler.iter_all_paragraphs())
        random.shuffle(all_paragraphs)

        valid_paras = filter(is_paragraph_valid, all_paragraphs)
        invalid_paras = itertools.filterfalse(
            is_paragraph_valid, all_paragraphs)
        paras_with_non_whole_sentence = filter(
            contains_non_whole_sentence, all_paragraphs)

        return {
            'valid_paragraphs': itertools.islice(valid_paras, 10),
            'invalid_paragraphs': itertools.islice(invalid_paras, 10),
            'paragraphs_with_non_whole_sentence': itertools.islice(paras_with_non_whole_sentence,
                                                                   20)
        }

    def get_outlier_utterances(self) -> Dict[str, Dict[str,
                                                       Iterator[UtteranceInfo]]]:
        """Returns outlier utterances from the dataset.

        Outliers are either extremely short or extremely long utterances.
        """

        word_count_lower_outliers = []
        length_sec_lower_outliers = []
        word_count_upper_outliers = []
        length_sec_upper_outliers = []

        stats_per_utterance = self._get_stats_per_utterance()

        q1_word_counts = np.percentile(stats_per_utterance['word_counts'], 25)
        q3_word_counts = np.percentile(stats_per_utterance['word_counts'], 75)

        q1_lengths_sec = np.percentile(stats_per_utterance['lengths_sec'], 25)
        q3_lengths_sec = np.percentile(stats_per_utterance['lengths_sec'], 75)

        for spk_id in self._raw_path_handler.iter_speakers():
            for utterance in self._raw_path_handler.iter_utterances_for_spk(spk_id):

                text = self._text_processor.load_text(utterance.text_path)
                text_norm = self._text_processor.clean_text(text)

                n_words = len(text_norm.split())
                length_sec = self._audio_processor.length_in_sec_of_file(
                    utterance.wav_path)

                if _is_outlier_lower(n_words, q1_word_counts, q3_word_counts):
                    word_count_lower_outliers.append(utterance)
                elif _is_upper_outlier(n_words, q1_word_counts, q3_word_counts):
                    word_count_upper_outliers.append(utterance)

                if _is_outlier_lower(length_sec, q1_lengths_sec, q3_lengths_sec):
                    length_sec_lower_outliers.append(utterance)
                elif _is_upper_outlier(length_sec, q1_lengths_sec, q3_lengths_sec):
                    length_sec_upper_outliers.append(utterance)

        random.shuffle(word_count_lower_outliers)
        random.shuffle(word_count_upper_outliers)
        random.shuffle(length_sec_lower_outliers)
        random.shuffle(length_sec_upper_outliers)

        return {
            'word_count': {
                'lower_outliers': itertools.islice(word_count_lower_outliers, 5),
                'upper_outliers': itertools.islice(word_count_upper_outliers, 5)
            },
            'length_sec': {
                'lower_outliers': itertools.islice(length_sec_lower_outliers, 5),
                'upper_outliers': itertools.islice(length_sec_upper_outliers, 5)
            }
        }

    def get_outliers_original_paragraphs(self) -> Dict[str, Iterator[OriginalParagraph]]:
        """Returns outlier original paragraphs from the dataset.

        Outliers are either extremely short or extremely long original paragraphs.
        """

        word_count_outliers = []
        sentence_count_outliers = []

        stats_per_orig_para = self._get_stats_per_original_paragraph()

        q1_word_counts = np.percentile(
            stats_per_orig_para['n_words_in_paragraph'], 25)
        q3_word_counts = np.percentile(
            stats_per_orig_para['n_words_in_paragraph'], 75)

        q1_sentence_counts = np.percentile(
            stats_per_orig_para['n_sentences_in_paragraph'], 25)
        q3_sentence_counts = np.percentile(
            stats_per_orig_para['n_sentences_in_paragraph'], 75)

        for para_info in self._raw_path_handler.iter_all_paragraphs():

            original_paragraph = self._raw_path_handler.get_original_paragraph(
                para_info)

            if original_paragraph is None:
                continue

            n_sentences = len(original_paragraph.sentences)
            n_words = sum(len(self._text_processor.clean_text(sentence).split())
                          for sentence in original_paragraph.sentences.values())

            if _is_outlier(n_words, q1_word_counts, q3_word_counts):
                word_count_outliers.append(original_paragraph)

            if _is_outlier(n_sentences, q1_sentence_counts, q3_sentence_counts):
                sentence_count_outliers.append(original_paragraph)

        random.shuffle(word_count_outliers)
        random.shuffle(sentence_count_outliers)

        return {
            'word_count': itertools.islice(word_count_outliers, 5),
            'sentence_count': itertools.islice(sentence_count_outliers, 5)
        }

    def _get_speaker_figures(self) -> Dict[str, Figure]:

        figures: Dict[str, Figure] = {}

        utt_stats_per_speaker = self._get_stats_per_speaker()
        bins = min(100, len(utt_stats_per_speaker['num_utt']))

        fig, ax = plt.subplots()
        ax.hist(utt_stats_per_speaker['num_utt'], bins=bins)
        ax.set_title('Number of utterances per speaker')
        ax.set_xlabel('Number of utterances')
        ax.set_ylabel('Number of speakers')

        figures['num_utterances_per_speaker'] = fig

        fig, ax = plt.subplots()
        ax.hist(np.array(utt_stats_per_speaker['total_dur']) / 60.0, bins=bins)
        ax.set_title('Total duration (minutes) of utterances per speaker')
        ax.set_xlabel('Total duration (minutes)')
        ax.set_ylabel('Number of speakers')

        figures['total_duration_minutes_per_speaker'] = fig

        fig, ax = plt.subplots()
        ax.hist(utt_stats_per_speaker['num_paragraphs'], bins=bins)
        ax.set_title('Number of paragraphs per speaker')
        ax.set_xlabel('Number of paragraphs')
        ax.set_ylabel('Number of speakers')

        figures['num_paragraphs_per_speaker'] = fig

        return figures

    def _get_stats_per_original_paragraph(self) -> Dict[str, Any]:

        n_words_in_paragraph = []
        n_sentences_in_paragraph = []
        n_words_in_sentence = []

        for para_info in self._raw_path_handler.iter_all_paragraphs():

            original_paragraph = self._raw_path_handler.get_original_paragraph(
                para_info)

            if original_paragraph is None:
                continue

            n_sentences_in_paragraph.append(len(original_paragraph.sentences))

            word_counts = [len(self._text_processor.clean_text(sentence).split())
                           for sentence in original_paragraph.sentences.values()]

            n_words_in_paragraph.append(sum(word_counts))
            n_words_in_sentence.extend(word_counts)

        return {
            'n_words_in_paragraph': n_words_in_paragraph,
            'n_sentences_in_paragraph': n_sentences_in_paragraph,
            'n_words_in_sentence': n_words_in_sentence
        }

    def _get_stats_per_speaker(self) -> Dict[str, Any]:
        """Returns stats related to utterances per speaker."""

        stats: Dict[str, Any] = {
            'num_utt': [],
            'total_dur': [],
            'num_paragraphs': []
        }

        for spk_id in self._raw_path_handler.iter_speakers():

            utterances_lengths = []
            n_paragraphs = 0

            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                for paragraph_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):

                    utterances_lengths.extend([
                        self._audio_processor.length_in_sec_of_file(
                            utt.wav_path)
                        for utt in paragraph_info.utterances
                    ])

                    n_paragraphs += 1

            stats['num_utt'].append(len(utterances_lengths))
            stats['total_dur'].append(np.sum(utterances_lengths))
            stats['num_paragraphs'].append(n_paragraphs)

        return stats

    def _get_stats_per_utterance(self) -> Dict[str, Any]:
        """Returns stats related to utterances."""

        stats: Dict[str, Any] = {
            'word_counts': [],
            'lengths_sec': [],
            'has_single_punctuations': [],
            'is_whole_sentence': []
        }

        for spk_id in self._raw_path_handler.iter_speakers():
            for utterance in self._raw_path_handler.iter_utterances_for_spk(spk_id):

                text = self._text_processor.load_text(utterance.text_path)

                if any(p in text for p in self._text_processor.single_puncts_replace):
                    stats['has_single_punctuations'].append(1)
                else:
                    stats['has_single_punctuations'].append(0)

                text_norm = self._text_processor.clean_text(text)

                if librittsr_helpers.is_sentence_whole(text_norm):
                    stats['is_whole_sentence'].append(1)
                else:
                    stats['is_whole_sentence'].append(0)

                n_words = len(text_norm.split())
                stats['word_counts'].append(n_words)

                stats['lengths_sec'].append(
                    self._audio_processor.length_in_sec_of_file(utterance.wav_path))

        return stats
