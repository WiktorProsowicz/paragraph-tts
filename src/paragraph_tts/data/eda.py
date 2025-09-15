"""Contains utilities for performing Exploratory Data Analysis of LibriTTS-R DS."""
from typing import Dict, Any

import numpy as np

from paragraph_tts import utils
from paragraph_tts.data import preprocessing


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

    def get_speakers_stats(self) -> Dict[str, Any]:
        """Returns stats related to speakers."""

        speakers_stats = {}

        speakers_stats['num_speakers'] = self._raw_path_handler.num_speakers

        spk_utter_stats = {}

        for spk_id in self._raw_path_handler.iter_speakers():

            utterances_lengths = []

            for utterance in self._raw_path_handler.iter_utterances_for_spk(spk_id):

                wav = self._audio_processor.load_wav(utterance.wav_path)
                utterances_lengths.append(self._audio_processor.length_in_sec(wav))

            spk_utter_stats[spk_id] = {
                'min_length (s)': float(np.min(utterances_lengths)),
                'max_length (s)': float(np.max(utterances_lengths)),
                'avg_length (s)': float(np.mean(utterances_lengths)),
                'num_utterances': len(utterances_lengths)
            }

        n_utterances_per_speaker = [stats['num_utterances'] for stats in spk_utter_stats.values()]
        
        speakers_stats['utterances_per_speaker'] = {
            'min': int(np.min(n_utterances_per_speaker)),
            'max': int(np.max(n_utterances_per_speaker)),
            'avg': float(np.mean(n_utterances_per_speaker)),
        }

        speakers_stats['utterances_stats_for_speakers'] = spk_utter_stats

        n_paragraphs_per_speaker = []

        for spk_id in self._raw_path_handler.iter_speakers():
            n_paragraphs = 0

            for chap_id in self._raw_path_handler.iter_chapters(spk_id):
                for _ in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_paragraphs += 1

            n_paragraphs_per_speaker.append(n_paragraphs)

        speakers_stats['paragraphs_per_speaker'] = {
            'min': int(np.min(n_paragraphs_per_speaker)),
            'max': int(np.max(n_paragraphs_per_speaker)),
            'avg': float(np.mean(n_paragraphs_per_speaker)),
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
        n_valid_paragraphs = 0 # Is complete and contains at least 3 utterances 

        for spk_id in self._raw_path_handler.iter_speakers():
            for chap_id in self._raw_path_handler.iter_chapters(spk_id):

                for para_info in self._raw_path_handler.iter_paragraphs(spk_id, chap_id):
                    n_utterances_in_paragraph.append(len(para_info.utterances))
                    if not para_info.is_complete:
                        n_incomplete_paragraphs += 1
                    elif len(para_info.utterances) > 3:
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

        utterances_stats['num_utterances'] = 0
        utt_word_counts = []
        utt_phoneme_counts = []

        for spk_id in self._raw_path_handler.iter_speakers():
            for utterance in self._raw_path_handler.iter_utterances_for_spk(spk_id):
                
                with open(utterance.text_path, 'r', encoding='utf-8') as text_f:
                    text = text_f.read()
                text_norm = self._text_processor.normalize_text(text)
                
                n_words = len(text_norm.split())

                utt_word_counts.append(n_words)

                utterances_stats['num_utterances'] += 1

        utterances_stats['words_per_utterance'] = {
            'min': int(np.min(utt_word_counts)),
            'max': int(np.max(utt_word_counts)),
            'avg': float(np.mean(utt_word_counts)),
        }

        return utterances_stats

        