"""Contains utilities for handling paths in generated enriched context for LibriTTS-R dataset."""

import os
import json
import logging
import sys

from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


class EnrichedContextDirHandler:
    """Manages access to content inside directory with enriched context for LibriTTS-R ds."""

    def __init__(self, contexts_path: str):
        """
        Args:
            contexts_path: Path to enriched contexts directory.
        """

        if not os.path.exists(contexts_path):
            _logger().warning("Enriched contexts directory does not exist: %s", contexts_path)
            sys.exit(1)

        self._contexts_path = contexts_path

        metadata_path = os.path.join(contexts_path, 'metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self._metadata = json.load(f)

    def contains_contexts_for(self,
                              para_info: raw_libri_dir_handler.ParagraphInfo,
                              utterance: raw_libri_dir_handler.UtteranceInfo) -> bool:
        """Checks if contexts for given paragraph exist."""

        utterance_contexts_path = self.path_for_utt_contexts(para_info, utterance)

        if not os.path.exists(utterance_contexts_path):
            return False

        return True

    def path_for_utt_contexts(self, para_info: raw_libri_dir_handler.ParagraphInfo,
                              utterance: raw_libri_dir_handler.UtteranceInfo) -> str:
        """Returns path to json file with contexts for given utterance."""

        file_name = '{spk}_{cha}_{para:06d}_{utt:06d}.contexts.json'.format(spk=para_info.spk_id,  # pylint: disable=C0209
                                                                            cha=para_info.chap_id,
                                                                            para=para_info.para_id,
                                                                            utt=utterance.utt_id)

        utterance_contexts_path = os.path.join(self._contexts_path,
                                               str(para_info.spk_id),
                                               str(para_info.chap_id),
                                               str(para_info.para_id),
                                               file_name)

        return utterance_contexts_path
