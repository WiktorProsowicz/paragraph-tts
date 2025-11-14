# -*- coding: utf-8 -*-
"""Contains utils for handling paths in processed LibriTTS-R dataset."""
import dataclasses
import enum
import json
import logging
import os
import sys
from typing import Iterator
from typing import List


def _logger():
    return logging.getLogger(__name__)


class SentencePosType(enum.Enum):
    """Represents position of a sentence in a paragraph."""

    FIRST = 0
    LAST = 1
    MIDDLE = 2
    ONLY = 3


def get_sentence_pos_type(n_preceding_sentences: int,
                          n_following_sentences: int) -> SentencePosType:
    """Determines position of a sentence in a paragraph.

    Args:
        n_preceding_sentences: Number of sentences preceding the sentence.
        n_following_sentences: Number of sentences following the sentence.

    Returns:
        Position type of the sentence.
    """

    if n_preceding_sentences == 0 and n_following_sentences == 0:
        return SentencePosType.ONLY

    if n_preceding_sentences == 0:
        return SentencePosType.FIRST

    if n_following_sentences == 0:
        return SentencePosType.LAST

    return SentencePosType.MIDDLE


@dataclasses.dataclass
class UtteranceContextInfo:
    """Contains information about paragraph context sentences."""

    is_original: bool
    utterance_pos: SentencePosType
    token_embeddings_path: str
    pse_path: str


@dataclasses.dataclass
class UtteranceDataInfo:
    """Contains paths to data related to a single utterance."""

    bert_embeddings_pth: str
    phoneme_ids_pth: str

    spec_pth: str
    f0_pth: str
    energy_pth: str
    durations_pth: str

    ling_stats_pth: str
    pos_tags_pth: str

    bert_to_word_pool_matrix_pth: str
    phone_to_spec_indices_pth: str
    spec_to_word_pool_matrix_pth: str
    word_to_phoneme_indices_pth: str


@dataclasses.dataclass
class SpeakerNumericalStats:
    """Contains paths containing speaker-specific numerical data."""
    f0_stats_pth: str
    energy_stats_pth: str


@dataclasses.dataclass
class SampleInfo:
    """Contains information about a sample in processed dataset."""

    spk_id: int
    chap_id: int
    para_id: int
    utt_id: int

    contexts: List[UtteranceContextInfo]
    input_data: UtteranceDataInfo
    spk_embedding_path: str


class ProcessedLibriDirHandler:
    """Manages access to content inside directory with processed LibriTTS-R ds."""

    def __init__(self, ds_path: str):
        """
        Args:
            ds_path: Path to processed ds.
        """

        os.makedirs(ds_path, exist_ok=True)

        self._metadata_path = os.path.join(ds_path, 'metadata.json')
        self._samples_path = os.path.join(ds_path, 'samples')
        self._num_stats_path = os.path.join(ds_path, 'speaker_num_stats')
        self._spk_embeddings_path = os.path.join(ds_path, 'spk_embeddings')

    def get_metadata(self):
        """Returns processed dataset's metadata."""

        with open(self._metadata_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def iter_samples(self) -> Iterator[SampleInfo]:
        """Iterates over all samples in the dataset."""

        for spk_id in os.listdir(self._samples_path):

            for para_signature in os.listdir(os.path.join(self._samples_path, spk_id)):

                chap_id, para_id = tuple(para_signature.split('_'))

                contexts_path = os.path.join(self._samples_path,
                                             spk_id,
                                             para_signature,
                                             'context_embeddings')

                for utt_id in os.listdir(os.path.join(self._samples_path,
                                                      spk_id,
                                                      para_signature,
                                                      'input_data')):

                    contexts = self._obtain_contexts_for_utterance(contexts_path, int(utt_id))

                    input_data_path = os.path.join(self._samples_path,
                                                   spk_id,
                                                   para_signature,
                                                   'input_data',
                                                   utt_id)

                    yield SampleInfo(
                        spk_id=int(spk_id),
                        chap_id=int(chap_id),
                        para_id=int(para_id),
                        utt_id=int(utt_id),
                        contexts=contexts,
                        input_data=self._obtain_utterance_data_info(input_data_path),
                        spk_embedding_path=os.path.join(self._spk_embeddings_path, f'{spk_id}.pt')
                    )

    def get_numerical_stats(self, spk_id: int) -> SpeakerNumericalStats:
        """Returns paths to speaker-specific numerical stats."""

        f0_stats_pth = os.path.join(self._num_stats_path, str(spk_id), 'f0_stats.pt')
        energy_stats_pth = os.path.join(self._num_stats_path, str(spk_id), 'energy_stats.pt')

        for path in [f0_stats_pth, energy_stats_pth]:
            if not os.path.exists(path):
                _logger().critical('Path %s does not exist!', path)
                sys.exit(1)

        return SpeakerNumericalStats(
            f0_stats_pth=f0_stats_pth,
            energy_stats_pth=energy_stats_pth
        )

    def _obtain_contexts_for_utterance(self,
                                       contexts_path: str,
                                       utt_id: int) -> List[UtteranceContextInfo]:

        contexts: List[UtteranceContextInfo] = []

        context_signatures = os.listdir(contexts_path)

        if 'original' in context_signatures:

            context_path = os.path.join(contexts_path, 'original')

            with open(os.path.join(context_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            contexts.append(UtteranceContextInfo(
                is_original=True,
                utterance_pos=get_sentence_pos_type(utt_id, metadata['length'] - utt_id - 1),
                token_embeddings_path=os.path.join(context_path, 'single_embeddings.pt'),
                pse_path=os.path.join(context_path, 'paired_embeddings.pt')
            ))

        for context_signature in context_signatures:

            if context_signature.startswith(f'enriched_{utt_id}_'):

                context_path = os.path.join(contexts_path, context_signature)

                with open(os.path.join(context_path, 'metadata.json'), 'r', encoding='utf-8') as f:
                    metadata = json.load(f)

                contexts.append(UtteranceContextInfo(
                    is_original=False,
                    utterance_pos=get_sentence_pos_type(
                        metadata['n_preceding_sentences'],
                        metadata['n_following_sentences']
                    ),
                    token_embeddings_path=os.path.join(context_path, 'single_embeddings.pt'),
                    pse_path=os.path.join(context_path, 'paired_embeddings.pt')
                ))

        for context in contexts:
            if not os.path.exists(context.token_embeddings_path):
                _logger().critical('Path %s does not exist!', context.token_embeddings_path)
                sys.exit(1)

            if not os.path.exists(context.pse_path):
                _logger().critical('Path %s does not exist!', context.pse_path)
                sys.exit(1)

        return contexts

    def _obtain_utterance_data_info(self, input_data_path: str) -> UtteranceDataInfo:

        paths = {
            'bert_embeddings_pth': os.path.join(input_data_path, 'bert_embeddings.pt'),
            'phoneme_ids_pth': os.path.join(input_data_path, 'phoneme_ids.pt'),

            'spec_pth': os.path.join(input_data_path, 'spec.pt'),
            'f0_pth': os.path.join(input_data_path, 'f0.pt'),
            'energy_pth': os.path.join(input_data_path, 'energy.pt'),

            'ling_stats_pth': os.path.join(input_data_path, 'ling_stats.pt'),
            'pos_tags_pth': os.path.join(input_data_path, 'pos_tags.pt'),

            'bert_to_word_pool_matrix_pth': os.path.join(input_data_path,
                                                         'bert_to_word_pool_matrix.pt'),
            'phone_to_spec_indices_pth': os.path.join(input_data_path,
                                                      'phone_to_spec_indices.pt'),
            'spec_to_word_pool_matrix_pth': os.path.join(input_data_path,
                                                         'spec_to_word_pool_matrix.pt'),
            'word_to_phoneme_indices_pth': os.path.join(input_data_path,
                                                        'word_to_phoneme_indices.pt'),
            'durations_pth': os.path.join(input_data_path, 'explicit_durations.pt')
        }

        for path in paths.values():
            if not os.path.exists(path):
                _logger().critical('Path %s does not exist!', path)
                sys.exit(1)

        return UtteranceDataInfo(**paths)
