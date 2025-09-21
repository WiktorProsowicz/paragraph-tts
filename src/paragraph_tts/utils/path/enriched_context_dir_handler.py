# -*- coding: utf-8 -*-
"""Contains utilities for handling paths in generated enriched context for LibriTTS-R dataset."""
import dataclasses
import json
import logging
import os
import sys
from typing import List

from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


@dataclasses.dataclass
class ContextForUtterance:
    """Holds enriched context for a single utterance."""

    preceding_sentences: List[str]
    following_sentences: List[str]


class EnrichedContextDirHandler:
    """Manages access to content inside directory with enriched context for LibriTTS-R ds."""

    def __init__(self, contexts_path: str):
        """
        Args:
            contexts_path: Path to enriched contexts directory.
        """

        self._contexts_path = contexts_path

        metadata_path = os.path.join(contexts_path, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

    def contains_contexts_for(self,
                              utterance: raw_libri_dir_handler.UtteranceInfo) -> bool:
        """Checks if contexts for given paragraph exist."""

        utterance_contexts_path = self.path_for_utt_contexts(utterance)

        if not os.path.exists(utterance_contexts_path):
            return False

        return True

    def get_contexts_for(self,
                         utterance: raw_libri_dir_handler.UtteranceInfo
                         ) -> List[ContextForUtterance]:
        """Returns contexts for given utterance."""

        utterance_contexts_path = self.path_for_utt_contexts(utterance)

        if not os.path.exists(utterance_contexts_path):
            _logger().critical('No contexts found for utterance: %s', utterance)
            sys.exit(1)

        with open(utterance_contexts_path, 'r', encoding='utf-8') as f:
            contexts_json = json.load(f)

        return [
            ContextForUtterance(
                preceding_sentences=c['preceding_sentences'],
                following_sentences=c['following_sentences']
            ) for c in contexts_json
        ]

    def path_for_utt_contexts(self, utterance: raw_libri_dir_handler.UtteranceInfo) -> str:
        """Returns path to json file with contexts for given utterance."""

        file_name = '{spk}_{cha}_{para:06d}_{utt:06d}.contexts.json'.format(spk=utterance.spk_id,  # pylint: disable=C0209
                                                                            cha=utterance.chap_id,
                                                                            para=utterance.para_id,
                                                                            utt=utterance.utt_id)

        utterance_contexts_path = os.path.join(self._contexts_path,
                                               str(utterance.spk_id),
                                               str(utterance.chap_id),
                                               str(utterance.para_id),
                                               file_name)

        return utterance_contexts_path
