# -*- coding: utf-8 -*-
"""Contains classes for processing/reading LibriTTS-R dataset."""
from typing import Dict, List, Optional
import logging
import os
import time

import numpy as np
import torch
import tqdm  # type: ignore
from comp_trans_tts import (deepspeaker)  # type: ignore
from sklearn.preprocessing import StandardScaler

from paragraph_tts import data
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.utils.path.raw_libri_dir_handler import RawLibriDirHandler
from paragraph_tts.utils.path.alignments_dir_handler import AlignmentsDirHandler
from paragraph_tts.utils.path.enriched_context_dir_handler import EnrichedContextDirHandler
from paragraph_tts.data.preprocessing import alignment as alignment_prep
from paragraph_tts.data.preprocessing import text as text_prep
from paragraph_tts.data.preprocessing import audio as audio_prep


def _logger():
    return logging.getLogger(__name__)


class LibriTTSRPreprocessor:
    """Runs preprocessing on raw dataset."""

    def __init__(
            self,
            raw_path_handler: RawLibriDirHandler,
            enriched_contexts_path_hand: Optional[EnrichedContextDirHandler],
            alignments_dir_hand: AlignmentsDirHandler,
            output_path: str,
            multi_speaker: bool,
            embedders_device: str):
        """
        Args:
            raw_path_handler: Handler for accessing raw dataset files.
        """

        self._raw_path_handler = raw_path_handler
        self._enriched_contexts_path_hand = enriched_contexts_path_hand
        self._alignments_path_hand = alignments_dir_hand

        self._output_path = output_path
        self._multi_speaker = multi_speaker
        self._spk_embedder = deepspeaker.embedder.DeepSpeakerEmbedder(embedders_device)
        self._audio_processor = audio_prep.AudioProcessor(
            sr=22050,
            hop_length=256,
            win_length=1025,
            n_mels=80,
            fmin=0,
            fmax=8000,
            trim_top_db=23
        )
        self._text_processor = text_prep.TextProcessor(embedders_device)

    def run_for_speaker(self, speaker_id: int):
        """Runs preprocessing of samples for given speaker."""

        if self._multi_speaker:
            _logger().debug('Preparing speaker embedding for spk %d', speaker_id)
            os.makedirs(os.path.join(self._output_path, 'spk_embeddings'), exist_ok=True)
            self._prepare_spk_embedding(speaker_id)

        for chap_id in self._raw_path_handler.iter_chapters(speaker_id):
            for para_info in self._raw_path_handler.iter_paragraphs(speaker_id, chap_id):

                _logger().debug('Processing paragraph %d of speaker %d',
                                para_info.para_id,
                                speaker_id)

                self._process_paragraph(para_info)

        self._normalize_contours_for_speaker(speaker_id)

    def _process_paragraph(self, para_info: raw_libri_dir_handler.ParagraphInfo):

        dst_dir = os.path.join(self._output_path,
                               'samples',
                               str(para_info.spk_id),
                               f'{para_info.chap_id}_{para_info.para_id}')

        context_embeddings_dir = os.path.join(dst_dir, 'context_embeddings')
        os.makedirs(context_embeddings_dir, exist_ok=True)

        original_paragraph = self._raw_path_handler.get_original_paragraph(para_info)

        self._prepare_context_embeddings(list(original_paragraph.sentences.values()),
                                         os.path.join(context_embeddings_dir, 'original'))

        for utt_info in para_info.utterances:

            self._process_utterance(utt_info, dst_dir, context_embeddings_dir)

    def _process_utterance(self,
                           utt_info: raw_libri_dir_handler.UtteranceInfo,
                           dst_dir: str,
                           context_embeddings_dir: str):

        if not self._alignments_path_hand.has_alignment_for(utt_info):
            _logger().debug('No alignment found for utterance %s, skipping.',
                            utt_info)
            return

        inputs_path = os.path.join(dst_dir, 'input_data', str(utt_info.utt_id))
        if os.path.exists(inputs_path):
            _logger().debug('Input data for utterance %s already exists, skipping.',
                            utt_info)
            return

        inputs = self._obtain_input_for_utterance(utt_info)

        if not inputs:
            return

        os.makedirs(inputs_path)

        for file_name, tensor in inputs.items():
            torch.save(tensor, os.path.join(inputs_path, f'{file_name}.pt'))

            if self._enriched_contexts_path_hand:

                if not self._enriched_contexts_path_hand.contains_contexts_for(utt_info):
                    continue

                contexts = self._enriched_contexts_path_hand.get_contexts_for(utt_info)

                for context_idx, context in enumerate(contexts):
                    self._prepare_context_embeddings(
                        context.as_paragraph(),
                        os.path.join(context_embeddings_dir,
                                     f'enriched_{utt_info.utt_id}_{context_idx}')
                    )

    def _obtain_input_for_utterance(self,
                                    utt_info: raw_libri_dir_handler.UtteranceInfo
                                    ) -> Optional[Dict[str, torch.Tensor]]:

        text = self._text_processor.load_text(utt_info.text_path)
        text_features = self._text_processor.tokenize_text(text)

        alignments = self._alignments_path_hand.get_alignment_for(utt_info)
        word_phoneme_int_mapping = alignment_prep.get_word_phoneme_mapping(alignments)

        if len(word_phoneme_int_mapping) != len(text_features.word_phoneme_mapping):
            _logger().debug('Alignment and text processor word counts do not match for utt %s, '
                            'skipping', utt_info)
            return None

        pauses = alignment_prep.get_pauses(word_phoneme_int_mapping)
        text_prep.add_pauses(text_features, pauses)

        phoneme_ids = self._text_processor.obtain_phoneme_ids(
            text_features.get_phoneme_sequence())
        bert_embeddings = self._text_processor.obtain_bert_embeddings(
            text_features.get_bert_token_sequence())

        bert_to_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
            text_features.get_word_to_token_spans()
        )
        word_to_phoneme_indices = alignment_prep.spans_to_indices_of_smaller_seq(
            text_features.get_word_to_phoneme_spans()
        )

        wav = self._audio_processor.load_wav(utt_info.wav_path)
        spec, energy, f0 = self._audio_processor.extract_spec_energy_f0(wav)

        phone_to_spec_indices = alignment_prep.spans_to_indices_of_smaller_seq(
            alignment_prep.get_phone_to_spec_spans(word_phoneme_int_mapping,
                                                   text_features.word_phoneme_mapping,
                                                   spec.shape[1])
        )
        spec_to_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
            alignment_prep.get_word_to_spec_spans(word_phoneme_int_mapping,
                                                  spec.shape[1])
        )

        return {
            'phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
            'bert_embeddings': bert_embeddings,
            'bert_to_word_pool_matrix': torch.tensor(bert_to_word_pool_matrix, dtype=torch.float),
            'word_to_phoneme_indices': torch.tensor(word_to_phoneme_indices, dtype=torch.long),
            'spec': torch.tensor(spec, dtype=torch.float),
            'energy': torch.tensor(energy, dtype=torch.float),
            'f0': torch.tensor(f0, dtype=torch.float),
            'phone_to_spec_indices': torch.tensor(phone_to_spec_indices, dtype=torch.long),
            'spec_to_word_pool_matrix': torch.tensor(spec_to_word_pool_matrix, dtype=torch.float)
        }

    def _normalize_contours_for_speaker(self, spk_id: int):

        speaker_path = os.path.join(self._output_path,
                                    'samples',
                                    str(spk_id))

        self._normalize_contours(speaker_path, 'f0')
        self._normalize_contours(speaker_path, 'energy')

    def _normalize_contours(self,
                            speaker_path: str,
                            contour_file_name: str):

        scaler = StandardScaler()

        for para_dir in os.listdir(speaker_path):
            for utt_dir in os.listdir(os.path.join(speaker_path, para_dir, 'input_data')):

                contour_path = os.path.join(speaker_path,
                                            para_dir,
                                            utt_dir,
                                            f'{contour_file_name}.pt')

                contour = torch.load(contour_path).numpy().reshape(-1, 1)
                scaler.partial_fit(contour)

        for para_dir in os.listdir(speaker_path):
            for utt_dir in os.listdir(os.path.join(speaker_path, para_dir, 'input_data')):

                contour_path = os.path.join(speaker_path,
                                            para_dir,
                                            utt_dir,
                                            f'{contour_file_name}.pt')

            contour = torch.load(contour_path).numpy().reshape(-1, 1)
            normalized_contour = scaler.transform(contour).squeeze()
            torch.save(torch.tensor(normalized_contour, dtype=torch.float), contour_path)

    def _prepare_context_embeddings(self,
                                    context_sentences: List[str],
                                    output_dir: str):

        if os.path.exists(output_dir):
            _logger().debug('Context embeddings for %s already exist, skipping.',
                            context_sentences)
            return

        single_embeddings = self._text_processor.obtain_bert_embeddings_for_sentences(
            context_sentences
        )

        if len(context_sentences) > 1:
            paired_embeddings = self._text_processor.obtain_paired_bert_embeddings(
                context_sentences
            )

        else:
            paired_embeddings = []

        os.makedirs(output_dir)

        torch.save(single_embeddings,
                   os.path.join(output_dir, 'single_embeddings.pt'))

        torch.save(paired_embeddings,
                   os.path.join(output_dir, 'paired_embeddings.pt'))

    def _prepare_spk_embedding(self, speaker_id: int):

        embedding_path = os.path.join(self._output_path,
                                      'spk_embeddings',
                                      f'{speaker_id}.pt')

        if os.path.exists(embedding_path):
            _logger().debug('Speaker embedding for spk %d already exist, skipping preparation.',
                            speaker_id)
            return

        embeddings: List[np.ndarray] = []

        for utterance_info in self._raw_path_handler.iter_utterances_for_spk(speaker_id):
            embedder_input = deepspeaker.preprocess.load_wav_for_deepseaker(
                utterance_info.wav_path)
            embeddings.append(self._spk_embedder(embedder_input)[0])

        final_embedding = np.mean(embeddings, axis=0)

        torch.save(torch.tensor(final_embedding),
                   embedding_path)
