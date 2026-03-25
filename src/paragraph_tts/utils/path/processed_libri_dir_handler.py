"""Contains utils for handling paths in processed LibriTTS-R dataset."""
import dataclasses
import enum
import json
import logging
import os
import pathlib
import sys
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List

import pydantic

from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class SentencePosType(enum.Enum):
    """Represents position of a sentence in a paragraph."""

    FIRST = 0
    LAST = 1
    MIDDLE = 2
    ONLY = 3

    @staticmethod
    def from_utt_id(utt_id: int, n_utterances_in_paragraph: int) -> 'SentencePosType':

        if n_utterances_in_paragraph == 1:
            return SentencePosType.ONLY

        if utt_id == 0:
            return SentencePosType.FIRST

        if utt_id == n_utterances_in_paragraph - 1:
            return SentencePosType.LAST

        return SentencePosType.MIDDLE


class ProcessedUtterance(pydantic.BaseModel):
    """Represents an utterance in the processed dataset."""

    raw_utterance: raw_libri_dir_handler.UtteranceInfo

    normalized_text_path: pathlib.Path

    spec_pth: pathlib.Path
    f0_pth: pathlib.Path
    energy_pth: pathlib.Path
    durations_pth: pathlib.Path

    ling_stats_pth: pathlib.Path
    pos_tags_pth: pathlib.Path

    bert_to_word_pool_matrix_pth: pathlib.Path
    phone_to_spec_indices_pth: pathlib.Path
    spec_to_word_pool_matrix_pth: pathlib.Path
    word_to_phoneme_indices_pth: pathlib.Path

    utterance_pos: SentencePosType


class SpeakerInfo(pydantic.BaseModel):
    """Represents a speaker in the processed dataset."""

    spk_id: int
    embedding_path: pathlib.Path
    energy_stats_path: pathlib.Path
    f0_stats_path: pathlib.Path


class ProcessedParagraph(pydantic.BaseModel):
    """Represents a paragraph in the processed dataset."""

    raw_paragraph: raw_libri_dir_handler.ParagraphInfo
    speaker_info: SpeakerInfo
    utterances: list[ProcessedUtterance]

    token_embeddings_path: pathlib.Path
    pse_path: pathlib.Path


class ProcessedLibriDirHandler:
    """Manages access to content inside directory with processed LibriTTS-R ds."""

    def __init__(self, ds_path: pathlib.Path):
        """
        Args:
            ds_path: Path to processed ds.
        """

        os.makedirs(ds_path, exist_ok=True)

        self._root_path = ds_path
        self._speakers_path = ds_path / 'speakers'

    def get_metadata(self) -> Dict[str, Any]:
        """Returns processed dataset's metadata."""

        with open(self._metadata_path, encoding='utf-8') as f:
            return json.load(f)  # type: ignore[no-any-return]

    def create_new_paragraph(self,
                             raw_paragraph: raw_libri_dir_handler.ParagraphInfo) -> ProcessedParagraph:
        """Initializes a new paragraph in the processed dataset based on the raw paragraph."""

        paragraphs_dir = self._speakers_path / str(raw_paragraph.spk_id) / 'paragraphs'

        paragraph_dir = paragraphs_dir / f'{raw_paragraph.chap_id}_{raw_paragraph.para_id}'
        paragraph_dir.mkdir(parents=True, exist_ok=True)

        with open(paragraph_dir / 'raw_paragraph.json', 'w', encoding='utf-8') as f:
            json.dump(raw_paragraph.model_dump(), f, ensure_ascii=False, indent=4)

        utterances_dir = paragraph_dir / 'utterances'
        utterances_dir.mkdir(parents=True, exist_ok=True)

        for utt_info in raw_paragraph.utterances:

            utterance_dir = utterances_dir / str(utt_info.utt_id)
            utterance_dir.mkdir(parents=True, exist_ok=True)

            with open(utterance_dir / f'raw_utterance.json', 'w', encoding='utf-8') as f:
                json.dump(utt_info.model_dump(), f, ensure_ascii=False, indent=4)

        return self._obtain_paragraph(paragraph_dir)

    def iter_utterances(self) -> Iterator[ProcessedUtterance]:
        """Iterates over all utterances in the dataset."""

        for spk_id in self._iter_speakers():

            spk_path = self._speakers_path / str(spk_id)
            paragraphs_path = spk_path / 'paragraphs'

            for paragraph_dir in paragraphs_path.iterdir():

                paragraph = self._obtain_paragraph(paragraph_dir)

                for utterance in paragraph.utterances:
                    yield utterance

    def _obtain_speaker_info(self, spk_id: int) -> SpeakerInfo:
        """Initializes or retrieves speaker info for a given speaker ID."""

        spk_path = self._speakers_path / str(spk_id)
        spk_path.mkdir(parents=True, exist_ok=True)

        paragraphs_path = spk_path / 'paragraphs'
        paragraphs_path.mkdir(parents=True, exist_ok=True)

        return SpeakerInfo(
            spk_id=spk_id,
            embedding_path=spk_path / 'embedding.pt',
            energy_stats_path=spk_path / 'energy_stats.pt',
            f0_stats_path=spk_path / 'f0_stats.pt'
        )

    def _obtain_paragraph(self, paragraph_dir: pathlib.Path) -> ProcessedParagraph:
        """Reads paragraph info from the given paragraph directory."""

        with open(paragraph_dir / 'raw_paragraph.json', encoding='utf-8') as f:
            raw_paragraph = raw_libri_dir_handler.ParagraphInfo.model_validate(json.load(f))

        speaker_info = self._obtain_speaker_info(raw_paragraph.spk_id)

        utterances: list[ProcessedUtterance] = []

        for utt_dir in (paragraph_dir / 'utterances').iterdir():

            with open(utt_dir / 'raw_utterance.json', encoding='utf-8') as f:
                raw_utt_info = raw_libri_dir_handler.UtteranceInfo.model_validate(json.load(f))

            utterances.append(
                ProcessedUtterance(
                    raw_utterance=raw_utt_info,
                    normalized_text_path=utt_dir / 'normalized_text.txt',
                    spec_pth=utt_dir / 'spec.pt',
                    f0_pth=utt_dir / 'f0.pt',
                    energy_pth=utt_dir / 'energy.pt',
                    durations_pth=utt_dir / 'durations.pt',
                    ling_stats_pth=utt_dir / 'ling_stats.pt',
                    pos_tags_pth=utt_dir / 'pos_tags.pt',
                    bert_to_word_pool_matrix_pth=utt_dir / 'bert_to_word_pool_matrix.pt',
                    phone_to_spec_indices_pth=utt_dir / 'phone_to_spec_indices.pt',
                    spec_to_word_pool_matrix_pth=utt_dir / 'spec_to_word_pool_matrix.pt',
                    word_to_phoneme_indices_pth=utt_dir / 'word_to_phoneme_indices.pt',
                    utterance_pos=SentencePosType.from_utt_id(raw_utt_info.utt_id,
                                                              len(raw_paragraph.utterances))
                ))

        return ProcessedParagraph(
            raw_paragraph=raw_paragraph,
            speaker_info=speaker_info,
            utterances=utterances,
            token_embeddings_path=paragraph_dir / 'token_embeddings.pt',
            pse_path=paragraph_dir / 'pse.pt'
        )

    def _iter_speakers(self) -> Iterator[int]:
        """Iterates over all speaker IDs in the dataset."""

        for spk_dir in self._speakers_path.iterdir():

            yield int(spk_dir.name)
