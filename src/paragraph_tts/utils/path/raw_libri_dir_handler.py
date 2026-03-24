"""Contains utilities for handling paths in raw LibriTTS-R dataset."""
import csv
import dataclasses
import logging
from collections import defaultdict
import pathlib
import os
from typing import Iterator


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


@dataclasses.dataclass
class UtteranceInfo:
    """Contains information about an utterance."""

    utt_id: int
    normalized_text: str
    wav_path: pathlib.Path | None = None


@dataclasses.dataclass
class ParagraphInfo:
    """Contains information about a paragraph."""

    spk_id: int
    chap_id: int
    para_id: int
    utterances: list[UtteranceInfo]


class RawLibriDirHandler:
    """Manages access to contents of raw LibriTTS-R dataset."""

    def __init__(self,
                 raw_ds_path: str,
                 choose_splits: list[str] | None = None):
        """
        Args:
            raw_ds_path: Root path to the LibriTTS-R raw ds.
        """

        self._raw_ds_path = raw_ds_path
        self._choose_splits = choose_splits

        self._spk_to_split = {}

        for ds_split in os.listdir(self._raw_ds_path):
            split_path = os.path.join(self._raw_ds_path, ds_split)

            for spk_id in os.listdir(split_path):

                if not os.path.isdir(os.path.join(split_path, spk_id)):
                    continue

                self._spk_to_split[int(spk_id)] = ds_split

    @property
    def num_speakers(self) -> int:
        """Returns number of speakers in the dataset."""
        return len(self._spk_to_split)

    def get_split_for_speaker(self, spk_id: int) -> str:
        """Returns the split (train/dev/test) for a given speaker ID."""
        return self._spk_to_split[spk_id]

    def iter_speakers(self) -> Iterator[int]:
        """Iterates over speaker IDs."""

        for ds_split in os.listdir(self._raw_ds_path):

            if (self._choose_splits is not None and ds_split not in self._choose_splits):
                continue

            split_path = os.path.join(self._raw_ds_path, ds_split)

            for spk_id in os.listdir(split_path):

                if not os.path.isdir(os.path.join(split_path, spk_id)):
                    continue

                yield int(spk_id)

    def iter_all_paragraphs(self) -> Iterator[ParagraphInfo]:
        """Iterates over all paragraphs in the dataset."""

        for spk_id in self.iter_speakers():
            yield from self.iter_paragraphs(spk_id)

    def iter_utterances_for_spk(self, spk_id: int) -> Iterator[UtteranceInfo]:
        """Iterates over all utterances for a given speaker."""

        for para_info in self.iter_paragraphs(spk_id):
            yield from para_info.utterances

    def iter_paragraphs(self, spk_id: int) -> Iterator[ParagraphInfo]:
        """Iterates over all paragraphs for a given speaker.

        Reads paragraph data from .book.tsv files across all chapters.

        Args:
            spk_id: Speaker ID.
        """

        speaker_path = os.path.join(
            self._raw_ds_path,
            self._spk_to_split[spk_id],
            str(spk_id)
        )

        if not os.path.isdir(speaker_path):
            _logger().debug('Speaker path does not exist: %s', speaker_path)
            return

        for chapter_id in os.listdir(speaker_path):

            for para_info in self._get_paragraph_drafts(spk_id, int(chapter_id)):

                for utt_info in para_info.utterances:

                    wav_path = os.path.join(
                        self._raw_ds_path,
                        self._spk_to_split[spk_id],
                        str(spk_id),
                        str(chapter_id),
                        f'{spk_id}_{chapter_id}_{para_info.para_id:06d}_{utt_info.utt_id:06d}.wav'
                    )

                    if os.path.exists(wav_path):
                        utt_info.wav_path = pathlib.Path(wav_path)

                yield para_info

    def _get_paragraph_drafts(self,
                              spk_id: int,
                              chapter_id: int) -> Iterator[ParagraphInfo]:
        """Composes initial paragraph drafts from .book.tsv."""

        book_tsv_path = os.path.join(
            self._raw_ds_path,
            self._spk_to_split[spk_id],
            str(spk_id),
            str(chapter_id),
            f'{spk_id}_{chapter_id}.book.tsv'
        )

        if not os.path.exists(book_tsv_path):
            _logger().debug('Missing book.tsv file: %s', book_tsv_path)
            return

        para_to_utt_ids = defaultdict(set)

        with open(book_tsv_path, encoding='utf-8') as f:

            for row in csv.reader(f, delimiter='\t', quotechar=None):

                if len(row) < 4:
                    _logger().debug('Failed to parse row in %s: %s', book_tsv_path, row)
                    return

                _, _, para_id, utt_id = row[0].split('_')

                para_to_utt_ids[para_id].add((utt_id, row[2]))

        for para_id, utt_ids_and_texts in para_to_utt_ids.items():
            yield ParagraphInfo(
                spk_id=spk_id,
                chap_id=chapter_id,
                para_id=int(para_id),
                utterances=[UtteranceInfo(utt_id=int(utt_id),
                                          normalized_text=text,
                                          wav_path=None)
                            for utt_id, text in sorted(utt_ids_and_texts)]
            )
