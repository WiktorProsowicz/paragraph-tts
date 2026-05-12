"""Contains module that prepares evaluation dataset."""

import pathlib
from typing import Annotated
import dataclasses
import tqdm
import random

import pydantic
from pydantic import Field
import torch

from torch_dev_utils.tts import text_prep
from torch_dev_utils.text_preprocessing import embeddings
from torch_dev_utils.tts import alignment_prep
from paragraph_tts.data.preprocessing import audio as audio_prep
from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.utils.path import alignments_dir_handler
from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.utils.path import eval_ds_handler
from paragraph_tts.data.preprocessing import stl_predictor_ds_processor


@dataclasses.dataclass
class UtteranceData:
    """Data components for a single utterance."""

    text_features: text_prep.TextFeatures
    tensors: dict[str, torch.Tensor]


@dataclasses.dataclass
class ParagraphData:
    """Data components for a single paragraph."""

    utterances_data: list[UtteranceData]
    context_features: dict[int, torch.Tensor]


class TestDsProcessor:
    """Prepares evaluation dataset."""

    class Configuration(pydantic.BaseModel):
        """Configuration for the processor."""

        bert_model_tag: Annotated[str, Field(
            description='Tag of the BERT model to use.')]

        embedder_device: Annotated[str, Field(
            description='Device to use for BERT/speaker embedding.')]

        spec_frames_per_second: Annotated[int, Field(
            description='Num. of spectrogram frames per second to calculate spk rate.')]

    def __init__(self,
                 config: Configuration,
                 raw_ds_handler: raw_libri_dir_handler.RawLibriDirHandler,
                 alignments_handler: alignments_dir_handler.AlignmentsDirHandler
                 ) -> None:
        """Initializes the processor."""

        self._raw_ds_handler = raw_ds_handler
        self._alignments_handler = alignments_handler

        self._text_preprocessor = text_prep.TextProcessor(config.bert_model_tag)
        self._bert_embedder = embeddings.BERTEmbedder(config.bert_model_tag,
                                                      device=config.embedder_device,
                                                      batch_size=16)

        self._cfg = config

    def prepare_dataset(self,
                        output_dir: pathlib.Path,
                        max_partial_paragraphs: int,
                        max_whole_paragraphs: int) -> None:
        """Prepares evaluation dataset."""

        output_dir.mkdir(parents=True, exist_ok=True)
        eval_dataset_handler = eval_ds_handler.EvalDsHandler(output_dir)

        partial_paragraphs: list[raw_libri_dir_handler.ParagraphInfo] = []
        whole_paragraphs: list[raw_libri_dir_handler.ParagraphInfo] = []

        all_paragraphs = list(self._raw_ds_handler.iter_all_paragraphs())
        random.Random(42).shuffle(all_paragraphs)

        for paragraph in all_paragraphs:

            if not any(utt.wav_path is not None for utt in paragraph.utterances):
                continue

            if all(utt.wav_path is not None for utt in paragraph.utterances):
                whole_paragraphs.append(paragraph)

            else:
                partial_paragraphs.append(paragraph)

        for paragraph in tqdm.tqdm(partial_paragraphs[:max_partial_paragraphs],
                                   desc='Preparing partial paragraphs'):
            self._prepare_paragraph_parts(paragraph, eval_dataset_handler)

        for paragraph in tqdm.tqdm(whole_paragraphs[:max_whole_paragraphs],
                                   desc='Preparing whole paragraphs'):
            self._prepare_whole_paragraph(paragraph, eval_dataset_handler)

    def _prepare_whole_paragraph(self,
                                 paragraph: raw_libri_dir_handler.ParagraphInfo,
                                 eval_dataset_handler: eval_ds_handler.EvalDsHandler) -> None:

        para_data = self._prepare_paragraph_data(paragraph, add_inter_utterance_silence=True)
        eval_para = eval_dataset_handler.create_whole_paragraph(paragraph)

        sentence_pos_list = [
            torch.repeat_interleave(utt.tensors['sentence_pos'],
                                    repeats=torch.tensor(len(utt.tensors['input_phoneme_ids'])))
            for utt in para_data.utterances_data
        ]

        spk_rate_list = [
            torch.repeat_interleave(utt.tensors['spk_rate'],
                                    repeats=torch.tensor(len(utt.tensors['input_phoneme_ids'])))
            for utt in para_data.utterances_data
        ]

        word_phone_indices_list: list[torch.Tensor] = []
        cum_word_count = 0

        for utt in para_data.utterances_data:

            word_phone_indices_list.append(utt.tensors['word_to_phoneme_indices'] + cum_word_count)
            cum_word_count += len(utt.text_features.words)

        utt_data = {
            'input_word_emb': torch.cat([utt.tensors['input_word_emb']
                                         for utt in para_data.utterances_data]),
            'input_ling_stats': torch.cat([utt.tensors['input_ling_stats']
                                          for utt in para_data.utterances_data]),
            'input_pos_tags': torch.cat([utt.tensors['input_pos_tags']
                                        for utt in para_data.utterances_data]),
            'input_phoneme_ids': torch.cat([utt.tensors['input_phoneme_ids']
                                           for utt in para_data.utterances_data]),
            'sentence_pos': torch.cat(sentence_pos_list),
            'spk_rate': torch.cat(spk_rate_list),
            'word_to_phoneme_indices': torch.cat(word_phone_indices_list)
        }

        for feature_name, feature_tensor in utt_data.items():
            torch.save(feature_tensor, eval_para.utterances[0].tensors_paths[feature_name])

        self._save_graph_features(eval_para, para_data)

        for feature_name, feature_tensor in para_data.context_features.items():
            torch.save(feature_tensor, eval_para.context_features_paths[feature_name])

    def _prepare_paragraph_parts(self,
                                 paragraph: raw_libri_dir_handler.ParagraphInfo,
                                 eval_dataset_handler: eval_ds_handler.EvalDsHandler) -> None:

        para_data = self._prepare_paragraph_data(paragraph, add_inter_utterance_silence=False)
        eval_para = eval_dataset_handler.create_partial_paragraph(paragraph)

        for eval_utt in eval_para.utterances:

            utt_data = para_data.utterances_data[eval_utt.raw_utterance.utt_id]

            for feature_name, feature_tensor in utt_data.tensors.items():
                torch.save(feature_tensor, eval_utt.tensors_paths[feature_name])

        self._save_graph_features(eval_para, para_data)

        for feature_name, feature_tensor in para_data.context_features.items():
            torch.save(feature_tensor, eval_para.context_features_paths[feature_name])

    def _save_graph_features(self,
                             eval_para: eval_ds_handler.EvalParagraph,
                             para_data: ParagraphData) -> None:

        torch.save([utt.tensors['input_word_emb'] for utt in para_data.utterances_data],
                   eval_para.graph_features_paths['word_embeddings_path'])

        text_features_list = [utt_data.text_features for utt_data in para_data.utterances_data]

        stl_predictor_ds_processor.GraphAssociationsProcessor(
            eval_para.graph_features_paths
        ).prepare_graph_associations_for_paragraph(text_features_list)

    def _prepare_paragraph_data(self,
                                paragraph: raw_libri_dir_handler.ParagraphInfo,
                                add_inter_utterance_silence: bool) -> ParagraphData:
        """Prepares data for a single paragraph."""

        context_sentences = [utt.normalized_text for utt in paragraph.utterances]

        context_tokens = torch.cat(
            self._bert_embedder.obtain_bert_embeddings_for_sentences(context_sentences)
        )

        if len(context_sentences) > 1:
            context_pse_list = self._bert_embedder.obtain_paired_bert_embeddings(context_sentences)
            context_pse = torch.stack(context_pse_list, dim=0)
        else:
            context_pse = torch.zeros((1, context_tokens.shape[-1]), dtype=torch.float32)

        utterances_data: list[UtteranceData] = []

        for utterance in paragraph.utterances:
            text_features = self._prepare_text_features(utterance, add_inter_utterance_silence)

            token_emb = self._bert_embedder.obtain_bert_embeddings(
                text_features.get_bert_token_sequence()).float()
            bert_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
                text_features.get_word_to_token_spans())
            word_emb = torch.matmul(torch.tensor(bert_word_pool_matrix).T, token_emb)
            phoneme_ids = self._text_preprocessor.obtain_phoneme_ids(
                text_features.get_phoneme_sequence())
            input_ling_stats = text_prep.obtain_ling_stats(text_features)
            pos_tags = self._text_preprocessor.obtain_pos_tags(text_features)
            word_phone_indices = alignment_prep.spans_to_indices_of_smaller_seq(
                text_features.get_word_to_phoneme_spans()
            )
            sentence_pos = processed_libri_dir_handler.SentencePosType.from_utt_id(
                utterance.utt_id,
                len(paragraph.utterances)
            )

            if utterance.wav_path is not None:
                sound_dur = audio_prep.length_in_sec_of_file(utterance.wav_path)
                spk_rate = sound_dur * self._cfg.spec_frames_per_second / word_emb.shape[0]
            else:
                spk_rate = -1.0

            utterances_data.append(
                UtteranceData(
                    text_features=text_features,
                    tensors={
                        'input_word_emb': word_emb,
                        'input_phoneme_ids': torch.tensor(phoneme_ids, dtype=torch.long),
                        'input_ling_stats': input_ling_stats,
                        'input_pos_tags': torch.tensor(pos_tags, dtype=torch.long),
                        'word_to_phoneme_indices': torch.tensor(word_phone_indices, dtype=torch.long),
                        'sentence_pos': torch.tensor(sentence_pos.value, dtype=torch.long),
                        'spk_rate': torch.tensor(spk_rate, dtype=torch.float32)
                    }
                )
            )

        return ParagraphData(
            utterances_data=utterances_data,
            context_features={
                'context_token_emb': context_tokens,
                'context_pse': context_pse
            }
        )

    def _prepare_text_features(self,
                               utterance: raw_libri_dir_handler.UtteranceInfo,
                               add_inter_utterance_silence: bool) -> text_prep.TextFeatures:
        """Prepares text features for a single utterance."""

        text_features = self._text_preprocessor.tokenize_text(utterance.normalized_text)

        if self._alignments_handler.has_alignment_for(utterance):
            alignments = self._alignments_handler.get_alignment_for(utterance)

            word_phone_int_mapping = alignment_prep.get_word_phoneme_mapping(alignments,
                                                                             trim_silences=True)
            pauses = alignment_prep.get_pauses(word_phone_int_mapping)

            if add_inter_utterance_silence:
                pauses.append((len(text_features.words) - 1, '<medium_pause>'))

            text_prep.add_pauses(text_features, pauses)

        return text_features
