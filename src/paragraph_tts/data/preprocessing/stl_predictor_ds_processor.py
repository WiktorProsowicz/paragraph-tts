""""Contains module that prepares the dataset for training the STL predictor."""

import pathlib
import dataclasses
import json
import logging
import pickle
import yaml

import tqdm
import torch
import numpy as np

from torch_dev_utils.tts import text_prep
from torch_dev_utils.text_preprocessing import embeddings
from torch_dev_utils.tts import alignment_prep
from comp_trans_tts.model import modules as ctt_modules

from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.utils.path import stl_predictor_ds_handler
from paragraph_tts.data.preprocessing.processor import LibriTTSRProcessor
from paragraph_tts.models.acoustic import acoustic as acoustic_models


def _logger() -> logging.Logger:
    """Returns the logger for the module."""

    return logging.getLogger(__name__)


@dataclasses.dataclass
class AcousticInferenceResources:
    """Holds the resources needed to run inference with the acoustic model."""

    acoustic_model: acoustic_models.AcousticModel
    gst_bin_params: ctt_modules.StlBinarizationParams
    wsv_bin_params: ctt_modules.StlBinarizationParams


def _cat_edge_indices(edge_idx_list: list[np.ndarray]) -> torch.Tensor:
    """Concatenates a list of edge index arrays into a single edge index tensor."""

    if len(edge_idx_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)

    return torch.from_numpy(np.concatenate(edge_idx_list, axis=1)).long()


class STLPredictorDatasetProcessor:
    """Prepares the dataset for training the STL predictor."""

    def __init__(self,
                 acoustic_ds_handler: processed_libri_dir_handler.ProcessedLibriDirHandler,
                 inference_resources: dict[str, AcousticInferenceResources],
                 embedder_device: str):
        """Initializes the dataset processor."""

        self._acoustic_ds_handler = acoustic_ds_handler

        with open(self._acoustic_ds_handler.processor_cfg_pth, 'r', encoding='utf-8') as f:
            acoustic_processor_cfg = LibriTTSRProcessor.Configuration.model_validate(
                yaml.safe_load(f))

        self._inference_resources = inference_resources

        self._text_processor = text_prep.TextProcessor(acoustic_processor_cfg.bert_embedder_model)
        self._embedder = embeddings.BERTEmbedder(acoustic_processor_cfg.bert_embedder_model,
                                                 embedder_device,
                                                 16)

    def process(self, output_dir: pathlib.Path) -> None:
        """Processes the dataset and saves it to the output directory."""

        processed_ds_handler = stl_predictor_ds_handler.STLPredictorDatasetHandler(
            output_dir,
            stl_series=list(self._inference_resources.keys())
        )

        for acoustic_para in tqdm.tqdm(self._acoustic_ds_handler.iter_paragraphs(),
                                       desc='Processing paragraphs',
                                       unit='paragraph'):

            processed_para = processed_ds_handler.create_paragraph(acoustic_para)

            acoustic_utt_dict = {
                utt.raw_utterance.utt_id: utt for utt in acoustic_para.utterances
            }

            utt_pairs = [
                (acoustic_utt_dict.get(utt.raw_utterance.utt_id), utt)
                for utt in processed_para.utterances
            ]

            for acoustic_utterance, utterance in utt_pairs:

                self._prepare_utterance_draft(utterance)

                self._prepare_data_for_utterance(acoustic_utterance, utterance)

        for processed_para in tqdm.tqdm(processed_ds_handler.iter_paragraphs(),
                                        desc='Preparing graph associations',
                                        unit='paragraph'):

            self._prepare_graph_associations_for_paragraph(processed_para)

    def _prepare_utterance_draft(self,
                                 utterance: stl_predictor_ds_handler.ProcessedUtterance) -> None:
        """Prepares draft data for a single utterance."""

        if utterance.text_features_pth.exists():
            _logger().debug('Utt draft already exists for utt %s, skipping',
                            utterance.raw_utterance)
            return

        text_features = self._text_processor.tokenize_text(utterance.raw_utterance.normalized_text)

        with open(utterance.text_features_pth, 'wb') as f:
            pickle.dump(text_features, f)

    def _prepare_data_for_utterance(
        self,
        acoustic_utterance: processed_libri_dir_handler.ProcessedUtterance | None,
        utterance: stl_predictor_ds_handler.ProcessedUtterance
    ) -> None:
        """Prepares all data for a single utterance."""

        with open(utterance.text_features_pth, 'rb') as f:
            text_features: text_prep.TextFeatures = pickle.load(f)

        if not utterance.word_embeddings_path.exists():

            token_embeddings = self._embedder.obtain_bert_embeddings(
                text_features.get_bert_token_sequence()).clone().to(torch.float32)
            token_word_pool_matrix = alignment_prep.spans_to_pool_matrix(
                text_features.get_word_to_token_spans()
            )
            word_embeddings = torch.matmul(torch.tensor(token_word_pool_matrix).T, token_embeddings)

            torch.save(word_embeddings, utterance.word_embeddings_path)

        if acoustic_utterance is None:
            _logger().debug('No acoustic data for utt %s, skipping prosody enc outputs',
                            utterance.raw_utterance)
            return

        for stl_series in self._inference_resources:
            self._prepare_stl_data(acoustic_utterance, utterance, stl_series)

    def _prepare_stl_data(self,
                          acoustic_utterance: processed_libri_dir_handler.ProcessedUtterance,
                          utterance: stl_predictor_ds_handler.ProcessedUtterance,
                          stl_series: str) -> None:
        """Prepares the STL predictor data for a single utterance and inference setup."""

        if utterance.stl_weights[stl_series].gst_path.exists():
            _logger().debug('STL data already exists for utt %s and STL series %s, skipping',
                            utterance.raw_utterance, stl_series)
            return

        inf_resources = self._inference_resources[stl_series]

        prosody_enc_inputs = {
            'input_spec': torch.load(acoustic_utterance.spec_pth),
            'spec_to_word_pool_matrix': torch.load(acoustic_utterance.spec_to_word_pool_matrix_pth),
            'input_phoneme_ids': torch.load(acoustic_utterance.phoneme_ids_pth),
            'input_ling_stats': torch.load(acoustic_utterance.ling_stats_pth),
            'phone_to_spec_indices': torch.load(acoustic_utterance.phone_to_spec_indices_pth)
        }
        prosody_enc_inputs['input_spec_length'] = torch.tensor(
            prosody_enc_inputs['input_spec'].size(1))

        prosody_enc_inputs = {
            key: value.unsqueeze(0).to(inf_resources.acoustic_model.device)
            for key, value in prosody_enc_inputs.items()
        }

        with torch.no_grad():
            prosody_enc_outputs = inf_resources.acoustic_model.obtain_prosody_encoder_outputs(
                prosody_enc_inputs,
                wsv_bin_params=inf_resources.wsv_bin_params,
                gst_bin_params=inf_resources.gst_bin_params
            )

            wsv_weights = prosody_enc_outputs['wsv_weights'].squeeze(0).cpu()
            gst_weights = prosody_enc_outputs['gst_weights'].squeeze(0).cpu()

        torch.save(wsv_weights, utterance.stl_weights[stl_series].wsv_path)
        torch.save(gst_weights, utterance.stl_weights[stl_series].gst_path)

    def _prepare_graph_associations_for_paragraph(
        self,
            paragraph: stl_predictor_ds_handler.ProcessedParagraph
    ) -> None:
        """Prepares the graph associations between nodes for a single paragraph."""

        sentence_lengths: list[int] = []

        for utterance in paragraph.utterances:
            with open(utterance.text_features_pth, 'rb') as f:
                text_features: text_prep.TextFeatures = pickle.load(f)

            sentence_lengths.append(len(text_features.words))

        word_indices = np.split(np.arange(sum(sentence_lengths)),
                                np.cumsum(sentence_lengths)[:-1])

        sentence_indices = np.arange(len(paragraph.utterances))

        self._prepare_local_graph_associations(paragraph, word_indices)
        self._prepare_global_graph_associations(paragraph, sentence_indices)
        self._prepare_hybrid_graph_associations(paragraph, word_indices, sentence_indices)

    def _prepare_local_graph_associations(
        self,
        paragraph: stl_predictor_ds_handler.ProcessedParagraph,
        word_indices: list[np.ndarray],
    ) -> None:
        """Prepares the local graph associations between nodes for a single paragraph."""

        local_prev_edge_idx = []
        local_next_edge_idx = []

        for sent_word_indices in word_indices:

            for word_idx, word_id in enumerate(sent_word_indices):

                previous_indices = sent_word_indices[:word_idx + 1]
                next_indices = sent_word_indices[word_idx + 1:]

                local_prev_edge_idx.append(np.array([previous_indices,
                                                     np.repeat(word_id, len(previous_indices))]))

                local_next_edge_idx.append(np.array([next_indices,
                                                     np.repeat(word_id, len(next_indices))]))

        torch.save(_cat_edge_indices(local_prev_edge_idx), paragraph.local_prev_edge_idx_pth)
        torch.save(_cat_edge_indices(local_next_edge_idx), paragraph.local_next_edge_idx_pth)

    def _prepare_global_graph_associations(
        self,
        paragraph: stl_predictor_ds_handler.ProcessedParagraph,
        sentence_indices: np.ndarray
    ) -> None:
        """Prepares the global graph associations between nodes for a single paragraph."""

        global_prev_sent_edge_idx = []
        global_next_sent_edge_idx = []

        for sent_idx in sentence_indices:

            prev_sent_indices = sentence_indices[:sent_idx + 1]
            next_sent_indices = sentence_indices[sent_idx + 1:]

            global_prev_sent_edge_idx.append(
                np.array([prev_sent_indices,
                          np.repeat(sent_idx, len(prev_sent_indices))])
            )

            global_next_sent_edge_idx.append(
                np.array([next_sent_indices,
                          np.repeat(sent_idx, len(next_sent_indices))])
            )

        torch.save(_cat_edge_indices(global_prev_sent_edge_idx), paragraph.global_prev_edge_idx_pth)
        torch.save(_cat_edge_indices(global_next_sent_edge_idx), paragraph.global_next_edge_idx_pth)

    def _prepare_hybrid_graph_associations(self,
                                           paragraph: stl_predictor_ds_handler.ProcessedParagraph,
                                           word_indices: list[np.ndarray],
                                           sentence_indices: np.ndarray) -> None:
        """Prepares the hybrid graph associations between nodes for a single paragraph."""

        local_global_edge_idx = []

        for sent_idx, sent_word_indices in zip(sentence_indices, word_indices):

            local_global_edge_idx.append(
                np.array([sent_word_indices,
                          np.repeat(sent_idx, len(sent_word_indices))])
            )

        torch.save(_cat_edge_indices(local_global_edge_idx), paragraph.local_global_edge_idx_pth)
