"""Contains utilities for performing Exploratory Data Analysis of LibriTTS-R DS."""
from typing import Dict, Any, Iterator, List
import itertools
import random

import numpy as np
import matplotlib.pyplot as plt  # type: ignore

from paragraph_tts import utils
from paragraph_tts.data import preprocessing


def is_paragraph_valid(para_info: utils.path.ParagraphInfo) -> bool:
    """Returns True if the paragraph is valid."""
    return para_info.is_complete and len(para_info.utterances) >= 3

class FeaturesExtractor:
    """Extracts various features from the dataset."""

    def __init__(self, raw_ds_path: str):
        """Inits the extractor.

        Args:
            raw_ds_path: Path to LibriTTS-R raw dataset.
        """

        self._raw_ds_path = raw_ds_path
        self._raw_path_handler = utils.path.RawLibriDirHandler(
            raw_ds_path
        )
        self._audio_processor = preprocessing.audio.AudioProcessor(
            sr=22050,
            hop_length=256,
            win_length=1025,
            n_mels=80,
            fmin=0,
            fmax=8000,
            trim_top_db=23
        )
        self._text_processor = preprocessing.text.TextProcessor()

    def paragraph_as_lines(self, para_info: utils.path.ParagraphInfo) -> List[str]:
        """Returns the paragraph as a list of lines (strings)."""
    
        lines = []

        for utt in para_info.utterances:
            with open(utt.text_path, 'r', encoding='utf-8') as text_f:
                text = self._text_processor.normalize_text(text_f.read())
            lines.append(f'{utt.utt_id}: {text}')

        return lines
    
    def paragraph_as_waveform(self, para_info: utils.path.ParagraphInfo) -> np.ndarray:
        """Returns the paragraph as a concatenated waveform."""
    
        waveforms = []

        for utt in para_info.utterances:
            wav = self._audio_processor.load_wav_raw(utt.wav_path)
            waveforms.append(wav)

        return np.concatenate(waveforms, axis=0)      

    def get_speakers_stats(self) -> Dict[str, Any]:
        """Returns stats related to speakers."""

        speakers_stats = {}

        speakers_stats['num_speakers'] = self._raw_path_handler.num_speakers

        utt_stats_per_speaker = self._get_stats_per_speaker()

        speakers_stats['utterances_per_speaker'] = {
            'min': int(np.min(utt_stats_per_speaker['num_utt'])),
            'max': int(np.max(utt_stats_per_speaker['num_utt'])),
            'avg': float(np.mean(utt_stats_per_speaker['num_utt'])),
        }

        speakers_stats['total_duration_per_speaker (minutes)'] = {
            'min': float(np.min(utt_stats_per_speaker['total_dur']) / 60.0),
            'max': float(np.max(utt_stats_per_speaker['total_dur']) / 60.0),
            'avg': float(np.mean(utt_stats_per_speaker['total_dur']) / 60.0),
        }

        speakers_stats['paragraphs_per_speaker'] = {
            'min': int(np.min(utt_stats_per_speaker['num_paragraphs'])),
            'max': int(np.max(utt_stats_per_speaker['num_paragraphs'])),
            'avg': float(np.mean(utt_stats_per_speaker['num_paragraphs'])),
        }

        return speakers_stats

    def get_chapters_stats(self) -> Dict[str, Any]:
        """Returns stats related to chapters."""

        chapters_stats = {}

        chapters_stats['num_chapters'] = len(list(self._raw_path_handler.iter_chapters()))

        n_paragraphs_in_chapter = {}
        n_utterances_in_chapter = {}

        for spk_id in self._raw_path_handler.iter_speakers():
            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                n_paragraphs_in_chapter[chap_id] = 0
                n_utterances_in_chapter[chap_id] = 0

                for para_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_paragraphs_in_chapter[chap_id] += 1
                    n_utterances_in_chapter[chap_id] += len(para_info.utterances)

        chapters_stats['paragraphs_per_chapter'] = {
            'min': int(np.min(list(n_paragraphs_in_chapter.values()))),
            'max': int(np.max(list(n_paragraphs_in_chapter.values()))),
            'avg': float(np.mean(list(n_paragraphs_in_chapter.values()))),
        }

        chapters_stats['utterances_per_chapter'] = {
            'min': int(np.min(list(n_utterances_in_chapter.values()))),
            'max': int(np.max(list(n_utterances_in_chapter.values()))),
            'avg': float(np.mean(list(n_utterances_in_chapter.values()))),
        }

        return chapters_stats

    def get_paragraphs_stats(self) -> Dict[str, Any]:
        """Returns stats related to paragraphs."""

        paragraphs_stats = {}

        n_utterances_in_paragraph = []
        n_incomplete_paragraphs = 0
        n_paragraphs = 0
        n_valid_paragraphs = 0  # Is complete and contains at least 3 utterances

        for spk_id in self._raw_path_handler.iter_speakers():
            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                for para_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_utterances_in_paragraph.append(len(para_info.utterances))

                    if not para_info.is_complete:
                        n_incomplete_paragraphs += 1

                    if is_paragraph_valid(para_info):
                        n_valid_paragraphs += 1

                    n_paragraphs += 1

        paragraphs_stats['utterances_per_paragraph'] = {
            'min': int(np.min(n_utterances_in_paragraph)),
            'max': int(np.max(n_utterances_in_paragraph)),
            'avg': float(np.mean(n_utterances_in_paragraph)),
        }

        paragraphs_stats['num_incomplete_paragraphs'] = n_incomplete_paragraphs
        paragraphs_stats['num_paragraphs'] = n_paragraphs
        paragraphs_stats['num_valid_paragraphs'] = n_valid_paragraphs

        return paragraphs_stats

    def get_utterances_stats(self) -> Dict[str, Any]:
        """Returns stats related to utterances."""

        utterances_stats = {}

        stats_per_utterance = self._get_stats_per_utterance()

        utterances_stats['num_utterances'] = len(stats_per_utterance['word_counts'])

        utterances_stats['words_per_utterance'] = {
            'min': int(np.min(stats_per_utterance['word_counts'])),
            'max': int(np.max(stats_per_utterance['word_counts'])),
            'avg': float(np.mean(stats_per_utterance['word_counts'])),
        }

        utterances_stats['length_sec_per_utterance'] = {
            'min': float(np.min(stats_per_utterance['lengths_sec'])),
            'max': float(np.max(stats_per_utterance['lengths_sec'])),
            'avg': float(np.mean(stats_per_utterance['lengths_sec'])),
        }

        utterances_stats['num_utterances_with_isolated_punctuations'] = sum(
            stats_per_utterance['has_single_punctuations'])

        total_dur = np.sum(stats_per_utterance['lengths_sec']) / 3600.0
        utterances_stats['total_duration (hours)'] = float(total_dur)

        return utterances_stats

    def get_figures(self) -> Dict[str, Any]:
        """Returns figures related to the dataset."""

        figures = {}

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

        return figures

    def get_example_paragraphs(self) -> Iterator[utils.path.ParagraphInfo]:
        """Returns example paragraphs from the dataset.

        The paragraphs are selected from valid and invalid ones.
        """
        
        valid_paras = filter(
            is_paragraph_valid,
            self._raw_path_handler.iter_all_paragraphs())
        invalid_paras = itertools.filterfalse(
            is_paragraph_valid,
            self._raw_path_handler.iter_all_paragraphs())
        
        valid_paras = list(valid_paras)
        invalid_paras = list(invalid_paras)

        random.shuffle(valid_paras)
        random.shuffle(invalid_paras)
        
        num_paras_needed = min(5, len(valid_paras), len(invalid_paras))

        return itertools.chain(
            itertools.islice(valid_paras, num_paras_needed),
            itertools.islice(invalid_paras, num_paras_needed)
        )

    def _get_stats_per_speaker(self) -> Dict[str, Any]:
        """Returns stats related to utterances per speaker."""

        stats = {
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
                        self._audio_processor.length_in_sec_of_file(utt.wav_path)
                        for utt in paragraph_info.utterances
                    ])

                    n_paragraphs += 1

            stats['num_utt'].append(len(utterances_lengths))
            stats['total_dur'].append(np.sum(utterances_lengths))
            stats['num_paragraphs'].append(n_paragraphs)

        return stats

    def _get_stats_per_utterance(self) -> Dict[str, Any]:
        """Returns stats related to utterances."""

        stats = {
            'word_counts': [],
            'lengths_sec': [],
            'has_single_punctuations': []
        }

        for spk_id in self._raw_path_handler.iter_speakers():
            for utterance in self._raw_path_handler.iter_utterances_for_spk(spk_id):

                with open(utterance.text_path, 'r', encoding='utf-8') as text_f:
                    text = text_f.read()

                if any(p in text for p in self._text_processor.single_puncts_replace):
                    stats['has_single_punctuations'].append(1)

                text_norm = self._text_processor.normalize_text(text)

                n_words = len(text_norm.split())
                stats['word_counts'].append(n_words)

                stats['lengths_sec'].append(
                    self._audio_processor.length_in_sec_of_file(utterance.wav_path))

        return stats
