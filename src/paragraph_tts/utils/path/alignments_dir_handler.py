"""Contains utilities for handling paths in alignments directory for LibriTTS-R dataset."""
import os
from typing import Dict

import tgt

from paragraph_tts.utils.path import raw_libri_dir_handler


class AlignmentsDirHandler:
    """Manages access to content inside alignments directory for LibriTTS-R ds."""

    def __init__(self, alignments_path: str):
        """
        Args:
            alignments_path: Path to alignments directory.
        """

        self._root_path = os.path.join(alignments_path, 'alignments')

        self._spk_to_split: Dict[int, str] = {}

        for split in os.listdir(self._root_path):

            for spk_id in os.listdir(os.path.join(self._root_path, split)):
                self._spk_to_split[int(spk_id)] = split

    def get_alignment_for(self,
                          utterance: raw_libri_dir_handler.UtteranceInfo) -> tgt.TextGrid:
        """Returns TextGrid alignment for given utterance."""

        alignment_path = self.get_alignment_path_for(utterance)

        return tgt.io.read_textgrid(alignment_path, include_empty_intervals=True)

    def has_alignment_for(self,
                          utterance: raw_libri_dir_handler.UtteranceInfo) -> bool:
        """Checks if alignment exists for given utterance."""

        return os.path.exists(self.get_alignment_path_for(utterance))

    def get_alignment_path_for(self,
                               utterance: raw_libri_dir_handler.UtteranceInfo) -> str:
        """Returns path to TextGrid alignment for given utterance."""

        file_name = '%d_%d_%06d_%06d.TextGrid' % (utterance.spk_id,  # pylint: disable=consider-using-f-string
                                                  utterance.chap_id,
                                                  utterance.para_id,
                                                  utterance.utt_id)

        alignment_path = os.path.join(self._root_path,
                                      self._spk_to_split[utterance.spk_id],
                                      str(utterance.spk_id),
                                      str(utterance.chap_id),
                                      file_name)

        return alignment_path
