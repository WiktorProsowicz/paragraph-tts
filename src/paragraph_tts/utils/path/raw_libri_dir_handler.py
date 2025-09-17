"""Contains utilities for handling paths in raw LibriTTS-R dataset."""

from typing import Iterator, Dict, Set, Optional, List
import dataclasses
import os
import re
import logging


def _logger():
    return logging.getLogger(__name__)


class ProcessedLibriDirHandler:
    """Manages access to content inside directory with processed LibriTTS-R ds."""

    def __init__(self, ds_path: str):
        """
        Args:
            ds_path: Path to processed ds.
        """

        os.makedirs(ds_path, exist_ok=True)

        self._metadata_path = os.path.join(ds_path, 'metadata.json')

    @property
    def metadata_path(self):
        """Returns path to a json file containing dataset's metadata."""
        return self._metadata_path


@dataclasses.dataclass
class UtteranceInfo:
    """Contains information about an utterance."""

    utt_id: int
    text_path: str
    wav_path: str


@dataclasses.dataclass
class ParagraphInfo:
    """Contains information about a paragraph."""

    spk_id: int
    para_id: int
    chap_id: int
    is_complete: bool
    utterances: list[UtteranceInfo]


class RawLibriDirHandler:
    """Manages access to contents of raw LibriTTS-R dataset."""

    def __init__(self, raw_ds_path: str):
        """
        Args:
            raw_ds_path: Root path to the LibriTTS-R raw ds.
        """

        self._raw_ds_path = raw_ds_path

        self._spk_to_split = {}

        for ds_split in os.listdir(self._raw_ds_path):
            split_path = os.path.join(self._raw_ds_path, ds_split)

            for spk_id in os.listdir(split_path):
                self._spk_to_split[int(spk_id)] = ds_split

    @property
    def num_speakers(self) -> int:
        """Returns number of speakers in the dataset."""
        return len(self._spk_to_split)

    def iter_speakers(self) -> Iterator[int]:
        """Iterates over speaker IDs."""

        for ds_split in os.listdir(self._raw_ds_path):
            split_path = os.path.join(self._raw_ds_path, ds_split)

            yield from map(int, os.listdir(split_path))

    def iter_chapters(self, speaker_id: Optional[int] = None) -> Iterator[int]:
        """Iterates over chapter IDs for a given speaker."""

        speaker_ids = [speaker_id] if speaker_id is not None else list(self.iter_speakers())

        for spk_id in speaker_ids:
            speaker_path = os.path.join(
                self._raw_ds_path,
                self._spk_to_split[spk_id],
                str(spk_id)
            )

            yield from os.listdir(speaker_path)

    def iter_all_paragraphs(self) -> Iterator[ParagraphInfo]:
        """Iterates over all paragraphs in the dataset."""

        for spk_id in self.iter_speakers():
            for chap_id in self.iter_chapters(spk_id):
                yield from self.iter_paragraphs(spk_id, chap_id)

    def iter_paragraphs(self, spk_id: int, chapter_id: int) -> Iterator[ParagraphInfo]:
        """Iterates over paragraphs in a chapter.

        Args:
            spk_id: Speaker ID.
            chapter_path: Path to the chapter directory.
        """

        chapter_path = os.path.join(
            self._raw_ds_path,
            self._spk_to_split[spk_id],
            str(spk_id),
            str(chapter_id)
        )

        para_to_utts = self._get_chap_and_utt_ids(chapter_id, spk_id)

        for para_id in sorted(para_to_utts.keys()):

            utterances = []

            for utt_id in sorted(para_to_utts[para_id]):
                base_name = f'{spk_id}_{chapter_id}_{para_id:06d}_{utt_id:06d}'

                utt_info = UtteranceInfo(utt_id=utt_id,
                                         text_path=os.path.join(
                                             chapter_path,
                                             base_name + '.normalized.txt'),
                                         wav_path=os.path.join(
                                             chapter_path,
                                             base_name + '.wav'))

                for required_path in (
                    utt_info.text_path,
                    utt_info.wav_path
                ):
                    if not os.path.exists(required_path):
                        _logger().warning('Required file %s does not exist!', required_path)
                        break

                utterances.append(utt_info)

            yield ParagraphInfo(
                spk_id=spk_id,
                para_id=para_id,
                chap_id=chapter_id,
                utterances=utterances,
                is_complete=self._is_chapter_complete(sorted(para_to_utts[para_id]))
            )

    def iter_utterances_for_spk(self, spk_id: int) -> Iterator[UtteranceInfo]:
        """Iterates over all utterances for a given speaker."""

        for chap_id in self.iter_chapters(spk_id):
            for para_info in self.iter_paragraphs(spk_id, chap_id):
                yield from para_info.utterances

    def _get_chap_and_utt_ids(self,
                              chap_id: int,
                              spk_id) -> Dict[int, Set[int]]:
        """Gets mapping from paragraph IDs to sets of utterance IDs in a chapter."""

        chapter_path = os.path.join(self._raw_ds_path,
                                    self._spk_to_split[spk_id],
                                    str(spk_id),
                                    str(chap_id))

        name_pattern = re.compile(r'\d+_\d+_(\d+)_(\d+)\..+')

        para_to_utts: Dict[int, Set[int]] = {}

        for file_name in os.listdir(chapter_path):
            match = name_pattern.match(file_name)

            if match is None:
                continue

            para_id = int(match.group(1))
            utt_id = int(match.group(2))

            if para_id not in para_to_utts:
                para_to_utts[para_id] = set()

            para_to_utts[para_id].add(utt_id)

        return para_to_utts

    def _is_chapter_complete(self, utt_ids: List[int]) -> bool:
        """Checks if sorted utterances IDs are contiguous and start with 0."""

        if len(utt_ids) == 0 or utt_ids[0] != 0:
            return False

        for prev_id, curr_id in zip(utt_ids, utt_ids[1:]):
            if curr_id != prev_id + 1:
                return False

        return True
