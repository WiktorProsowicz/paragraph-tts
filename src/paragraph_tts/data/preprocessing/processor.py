"""Contains classes for processing/reading LibriTTS-R dataset."""

import json
import logging
import os
from typing import Any
from typing import Annotated
from typing import Iterator
import pickle
import pathlib
import tqdm


import numpy as np
import torch
import pydantic
from pydantic import Field
from comp_trans_tts import deepspeaker
from sklearn.preprocessing import StandardScaler
from torch_dev_utils.text_preprocessing import embeddings
from torch_dev_utils.tts import alignment_prep
from torch_dev_utils.tts import text_prep

from paragraph_tts.data.preprocessing import audio as audio_prep
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.utils.path import alignments_dir_handler
from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.data import librittsr_helpers


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class LibriTTSRProcessor:
    """Runs preprocessing on raw dataset."""

    class Configuration(pydantic.BaseModel):
        """Configuration of the preprocessor."""

        multi_speaker: Annotated[bool, Field(description='Whether to prepare speaker embeddings.')]

        embedders_device: Annotated[str, Field(description='Device to run embedders on.')]

        n_words_boundaries: Annotated[tuple[int, int] | None, Field(
            description=('Minimum and maximum number of words a paragraph should contain'
                         ' to be included. If None, no filtering by word count is performed.')
        )]

        duration_boundaries: Annotated[tuple[float, float] | None, Field(
            description=('Minimum and maximum duration in seconds of an utterance to be included.'
                         ' If None, no filtering by duration is performed.')
        )]

        n_utterances_boundaries: Annotated[tuple[int, int] | None, Field(
            description=('Minimum and maximum number of utterances in a paragraph to be included.'
                         ' If None, no filtering by utterance count is performed.')
        )]

        bert_embedder_model: Annotated[str, Field(description='Model name for BERT embedder.')]

        audio_processor_cfg: Annotated[audio_prep.AudioProcessor.Configuration, Field(
            description='Configuration of the audio processor.'
        )]

    def __init__(self, cfg: Configuration):

        self._spk_embedder = deepspeaker.embedder.DeepSpeakerEmbedder(cfg.embedders_device)
        self._audio_processor = audio_prep.AudioProcessor(cfg.audio_processor_cfg)
        self._text_processor = text_prep.TextProcessor(cfg.bert_embedder_model)
        self._embedder = embeddings.BERTEmbedder(pretrained_model_name=cfg.bert_embedder_model,
                                                 device=cfg.embedders_device,
                                                 batch_size=16)

        self._cfg = cfg

    def process_dataset(self,
                        raw_ds_handler: raw_libri_dir_handler.RawLibriDirHandler,
                        alignments_handler: alignments_dir_handler.AlignmentsDirHandler,
                        output_dir: pathlib.Path,
                        metadata_output_dir: pathlib.Path) -> None:
        """Runs preprocessing on the dataset."""

        output_dir.mkdir(parents=True, exist_ok=True)

        for speaker_id in tqdm.tqdm(raw_ds_handler.iter_speakers(),
                                    desc='Preparing paragraph drafts for speakers',
                                    unit='speaker'):

            for para_info in tqdm.tqdm(raw_ds_handler.iter_paragraphs(speaker_id),
                                       desc=f'Preparing paragraph drafts of speaker {speaker_id}',
                                       unit='paragraph',
                                       leave=False):

                _logger().debug('Preparing paragraph draft %d_%d of speaker %d',
                                para_info.chap_id,
                                para_info.para_id,
                                speaker_id)

                self._prepare_paragraph_draft(para_info,
                                              alignments_handler,
                                              output_dir)

        processed_ds_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(output_dir)

        for utterance in tqdm.tqdm(processed_ds_handler.iter_utterances(),
                                   desc='Preparing data for utterances',
                                   unit='utterance'):

            _logger().debug('Preparing data for utterance %d of paragraph %d of speaker %d',
                            utterance.raw_utterance.utt_id,
                            utterance.raw_utterance.para_id,
                            utterance.raw_utterance.spk_id)

            self._prepare_data_for_utterance(utterance)

    def _prepare_paragraph_draft(self,
                                 paragraph_info: raw_libri_dir_handler.ParagraphInfo,
                                 alignments_handler: alignments_dir_handler.AlignmentsDirHandler,
                                 output_dir: pathlib.Path):
        """Prepares a paragraph for processing.

        The paragraph is validated and all its valid utterances are prepared in their respective
        directories. If the paragraph does not meet the filtering criteria or if none of its
        utterances could be prepared successfully, the paragraph is not included in the processed
        dataset.
        """

        processed_ds_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(output_dir)

        if not self._should_process_paragraph(paragraph_info):
            _logger().debug('Paragraph %s does not meet filtering criteria, skipping.',
                            paragraph_info)
            return

        processed_paragraph = processed_ds_handler.create_new_paragraph(
            paragraph_info,
            list(filter(self._should_process_utterance, paragraph_info.utterances))
        )

        has_any_utterances_to_process = False

        for utterance in processed_paragraph.utterances:

            if not utterance.normalized_text_path.exists():
                if not self._prepare_utterance_draft(utterance, alignments_handler):
                    processed_ds_handler.delete_utterance(utterance)

                else:
                    has_any_utterances_to_process = True

            else:
                has_any_utterances_to_process = True

        if not has_any_utterances_to_process:
            _logger().debug('No utterances processed for %s, skipping.',
                            paragraph_info)
            processed_ds_handler.delete_paragraph(processed_paragraph)
            return

        self._prepare_context_data(processed_paragraph)

    def _prepare_utterance_draft(self,
                                 utt_info: processed_libri_dir_handler.ProcessedUtterance,
                                 alignments_handler: alignments_dir_handler.AlignmentsDirHandler
                                 ) -> bool:
        """Prepares an utterance for processing. Returns True if preparation was successful."""

        text_features = self._text_processor.tokenize_text(utt_info.raw_utterance.normalized_text)

        if not alignments_handler.has_alignment_for(utt_info.raw_utterance):
            _logger().debug('No alignment found for utterance %s, skipping.',
                            utt_info.raw_utterance)
            return False

        alignments = alignments_handler.get_alignment_for(utt_info.raw_utterance)
        word_phoneme_int_mapping = alignment_prep.get_word_phoneme_mapping(alignments,
                                                                           trim_silences=True)

        if len(word_phoneme_int_mapping) != len(text_features.word_phoneme_mapping):
            _logger().debug('Alignment and text processor word counts do not match for utt %s, '
                            'skipping', utt_info.raw_utterance)
            return False

        text_prep.add_pauses(text_features, alignment_prep.get_pauses(word_phoneme_int_mapping))

        with open(utt_info.text_features_path, 'wb') as f:
            pickle.dump(text_features, f)

        with open(utt_info.word_phone_interval_mapping_path, 'wb') as f:
            pickle.dump(word_phoneme_int_mapping, f)

        return True

    def _prepare_data_for_utterance(self,
                                    utt_info: processed_libri_dir_handler.ProcessedUtterance
                                    ) -> None:
        """Prepares all data for a single utterance after its has been created."""

        with open(utt_info.text_features_path, 'rb') as f:
            text_features: text_prep.TextFeatures = pickle.load(f)

        with open(utt_info.word_phone_interval_mapping_path, 'rb') as f:
            word_phoneme_int_mapping: alignment_prep.WordPhonemeMapping = pickle.load(f)

        phoneme_ids = self._text_processor.obtain_phoneme_ids(
            text_features.get_phoneme_sequence())

        bert_embeddings = self._embedder.obtain_bert_embeddings(
            text_features.get_bert_token_sequence()).clone().to(torch.float16)

        bert_to_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
            text_features.get_word_to_token_spans()
        )
        word_to_phoneme_indices = alignment_prep.spans_to_indices_of_smaller_seq(
            text_features.get_word_to_phoneme_spans()
        )
        ling_stats = text_prep.obtain_ling_stats(text_features)
        pos_tags = self._text_processor.obtain_pos_tags(text_features)

        wav = self._audio_processor.load_wav(utt_info.raw_utterance.wav_path)
        spec, energy, f0 = self._audio_processor.extract_spec_energy_f0(wav)

        spec_phone_spans = alignment_prep.get_phone_to_spec_spans(
            word_phoneme_int_mapping,
            text_features.word_phoneme_mapping,
            spec.shape[1])

        phone_to_spec_indices = alignment_prep.spans_to_indices_of_smaller_seq(
            spec_phone_spans
        )
        spec_to_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
            alignment_prep.get_word_to_spec_spans(word_phoneme_int_mapping,
                                                  spec.shape[1])
        )

        with open(utt_info.normalized_text_path, 'w', encoding='utf-8') as f:
            f.write(utt_info.raw_utterance.normalized_text)

        torch.save(spec, utt_info.spec_pth)
        torch.save(phoneme_ids, utt_info.phoneme_ids_path)
        torch.save(bert_embeddings, utt_info.bert_embeddings_path)
        torch.save(f0, utt_info.f0_pth)
        torch.save(energy, utt_info.energy_pth)
        torch.save(spec_phone_spans, utt_info.durations_pth)
        torch.save(ling_stats, utt_info.ling_stats_pth)
        torch.save(pos_tags, utt_info.pos_tags_pth)
        torch.save(bert_to_word_pool_matrix, utt_info.bert_to_word_pool_matrix_pth)
        torch.save(phone_to_spec_indices, utt_info.phone_to_spec_indices_pth)
        torch.save(spec_to_word_pool_matrix, utt_info.spec_to_word_pool_matrix_pth)
        torch.save(word_to_phoneme_indices, utt_info.word_to_phoneme_indices_pth)

    def _prepare_context_data(self,
                              processed_paragraph: processed_libri_dir_handler.ProcessedParagraph
                              ) -> None:

        if processed_paragraph.token_embeddings_path.exists():
            _logger().debug('Context embeddings for %s already exist, skipping.',
                            processed_paragraph.raw_paragraph)
            return

        context_sentences = [utt.raw_utterance.normalized_text
                             for utt in processed_paragraph.utterances]

        single_embeddings = self._embedder.obtain_bert_embeddings_for_sentences(context_sentences)

        if len(context_sentences) > 1:
            paired_embeddings = self._embedder.obtain_paired_bert_embeddings(
                context_sentences
            )

        else:
            paired_embeddings = []

        torch.save([t.clone().to(torch.float16) for t in single_embeddings],
                   processed_paragraph.token_embeddings_path)

        torch.save([t.clone().to(torch.float16) for t in paired_embeddings],
                   processed_paragraph.pse_path)

    def _should_process_paragraph(self,
                                  paragraph_info: raw_libri_dir_handler.ParagraphInfo) -> bool:
        """Tells if the given paragraph should be processed based on the filtering criteria."""

        if self._cfg.n_utterances_boundaries is not None:

            min_utt, max_utt = self._cfg.n_utterances_boundaries

            if not min_utt <= len(paragraph_info.utterances) <= max_utt:
                return False

        if self._cfg.n_words_boundaries is not None:

            min_words, max_words = self._cfg.n_words_boundaries
            total_words = sum(librittsr_helpers.count_words_in_text(utt.normalized_text)
                              for utt in paragraph_info.utterances)

            if not min_words <= total_words <= max_words:
                return False

        return True

    def _should_process_utterance(self,
                                  utterance_info: raw_libri_dir_handler.UtteranceInfo) -> bool:
        """Tells if the given utterance should be processed based on the filtering criteria."""

        if utterance_info.wav_path is None:
            return False

        if self._cfg.duration_boundaries is not None:

            min_dur, max_dur = self._cfg.duration_boundaries
            dur = audio_prep.length_in_sec_of_file(utterance_info.wav_path)

            if not min_dur <= dur <= max_dur:
                return False

        return True

    # def _save_normalization_stats_for_speaker(self, spk_id: int) -> None:

    #     self._save_norm_stats(spk_id, 'f0')
    #     self._save_norm_stats(spk_id, 'energy')

    # def _save_norm_stats(self,
    #                      spk_id: int,
    #                      contour_file_name: str) -> None:

    #     speaker_path = os.path.join(self._output_path,
    #                                 'samples',
    #                                 str(spk_id))

    #     scaler = StandardScaler()

    #     for para_dir in os.listdir(speaker_path):
    #         for utt_dir in os.listdir(os.path.join(speaker_path, para_dir, 'input_data')):

    #             contour_path = os.path.join(speaker_path,
    #                                         para_dir,
    #                                         'input_data',
    #                                         utt_dir,
    #                                         f'{contour_file_name}.pt')

    #             contour = torch.load(contour_path).numpy().reshape(-1, 1)
    #             scaler.partial_fit(contour)

    #     stats_path = os.path.join(self._output_path,
    #                               'speaker_num_stats',
    #                               str(spk_id))

    #     os.makedirs(stats_path, exist_ok=True)

    #     torch.save({
    #         'mean': torch.tensor(scaler.mean_, dtype=torch.float),
    #         'std': torch.tensor(np.sqrt(scaler.var_), dtype=torch.float)
    #     }, os.path.join(stats_path, f'{contour_file_name}_stats.pt'))

    # def _prepare_spk_embedding(self, speaker_id: int) -> None:

    #     embedding_path = os.path.join(self._output_path,
    #                                   'spk_embeddings',
    #                                   f'{speaker_id}.pt')

    #     if os.path.exists(embedding_path):
    #         _logger().debug('Speaker embedding for spk %d already exist, skipping preparation.',
    #                         speaker_id)
    #         return

    #     spk_embeddings: List[np.ndarray] = []

    #     for utterance_info in self._raw_path_handler.iter_utterances_for_spk(speaker_id):
    #         embedder_input = deepspeaker.preprocess.load_wav_for_deepseaker(
    #             utterance_info.wav_path)
    #         spk_embeddings.append(self._spk_embedder(embedder_input)[0])

    #     final_embedding = np.mean(spk_embeddings, axis=0)

    #     torch.save(torch.tensor(final_embedding),
    #                embedding_path)
