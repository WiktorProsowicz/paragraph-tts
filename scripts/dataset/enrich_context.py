# -*- coding: utf-8 -*-
"""Generates enriched context for dataset samples using a language model."""
import itertools
import json
import logging
import os
import random
import sys
from typing import Any, Dict, Optional

import hydra
import omegaconf
import tqdm

from paragraph_tts.data import enrichment
from paragraph_tts.data import librittsr_helpers
from paragraph_tts.data.preprocessing import text as text_prep
from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import enriched_context_dir_handler
from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


def _prepare_utterance_for_enrichment(
        original_paragraph: Optional[raw_libri_dir_handler.OriginalParagraph],
                                      utt_info: raw_libri_dir_handler.UtteranceInfo,
                                      ds_metadata: librittsr_helpers.LibriTTSRMetadata
                                      ) -> enrichment.UtteranceForEnrichment:
    
    if original_paragraph is None:
        preceding_sentences = []
        following_sentences = []

    else:
        preceding_sentences = [
            original_paragraph.sentences[k] for k in sorted(original_paragraph.sentences)
            if k < utt_info.utt_id
        ]

        following_sentences = [
            original_paragraph.sentences[k] for k in sorted(original_paragraph.sentences)
            if k > utt_info.utt_id
        ]

    utt_text = text_prep.TextProcessor.load_text(utt_info.text_path)

    speaker, book = ds_metadata.get_speaker_and_book(utt_info.spk_id,
                                                     utt_info.chap_id)

    return enrichment.UtteranceForEnrichment(
        text=utt_text,
        original_preceding_sentences=preceding_sentences,
        original_following_sentences=following_sentences,
        book=book,
        speaker=speaker
    )


def _enrich_utterance_and_save(enricher: enrichment.ContextEnricher,
                               ds_metadata: librittsr_helpers.LibriTTSRMetadata,
                               original_paragraph: raw_libri_dir_handler.OriginalParagraph,
                               utt_info: raw_libri_dir_handler.UtteranceInfo,
                               num_contexts_to_generate: int,
                               output_path: str):


    utterance_for_enrichment = _prepare_utterance_for_enrichment(
        original_paragraph, utt_info, ds_metadata
    )

    contexts = [enricher.generate_context_for_utt(utterance_for_enrichment)
                for _ in range(num_contexts_to_generate)]

    if any(c is None for c in contexts):
        _logger().debug('Failed to generate some contexts for utterance: %s', utt_info)

    contexts = [c for c in contexts if c is not None]

    if not contexts:
        _logger().debug('Failed to generate any contexts for utterance: %s', utt_info)
        return

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(contexts, f, indent=4, ensure_ascii=False)


def _should_enrich_utterance(utt_info: raw_libri_dir_handler.UtteranceInfo,
                             filters: Dict[str, Any]) -> bool:

    text = text_prep.TextProcessor.load_text(utt_info.text_path)
    text = text_prep.TextProcessor.clean_text(text)

    n_words = len(text.split())

    if n_words > filters['max_words_in_utterance']:
        return False

    if not filters['allow_fragmented_sentences'] and not librittsr_helpers.is_sentence_whole(text):
        return False

    return True


@hydra.main(version_base=None, config_path='cfg', config_name='enrich_context')
def main(script_cfg: omegaconf.DictConfig):
    """Performs context enrichment for dataset samples."""

    logging_utils.setup_logging('enrich_context')

    _logger().info('Script configuration:\n%s',
                   json.dumps(omegaconf.OmegaConf.to_container(script_cfg), indent=4))

    os.makedirs(script_cfg.output_path, exist_ok=True)

    metadata = {
        'model_name': script_cfg.model_name,
        'max_paragraph_len': script_cfg.max_paragraph_len,
        'min_paragraph_len': script_cfg.min_paragraph_len,
        'num_contexts_to_generate': script_cfg.num_contexts_to_generate,
        'filters': omegaconf.OmegaConf.to_container(script_cfg.filters)
    }

    with open(os.path.join(script_cfg.output_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    enricher = enrichment.ContextEnricher(model_name=script_cfg.model_name,
                                          ollama_host=script_cfg.ollama_host,
                                          max_paragraph_len=script_cfg.max_paragraph_len,
                                          min_paragraph_len=script_cfg.min_paragraph_len,
                                          should_retry=script_cfg.should_retry_generate)

    if not enricher.is_model_available(script_cfg.model_name):
        _logger().critical('The following model is unavailable at the Ollama server: %s',
                           script_cfg.model_name)
        sys.exit(1)

    raw_path_handler = raw_libri_dir_handler.RawLibriDirHandler(
        script_cfg.raw_ds_path)

    ds_metadata = librittsr_helpers.LibriTTSRMetadata()

    enriched_dir_handler = enriched_context_dir_handler.EnrichedContextDirHandler(
        script_cfg.output_path)

    def iter_filtered_utterances():
        for para_info in raw_path_handler.iter_all_paragraphs():
            split = raw_path_handler.get_split_for_speaker(para_info.spk_id)

            if split not in script_cfg.filters.choose_splits:
                continue

            original_paragraph = raw_path_handler.get_original_paragraph(para_info)

            if original_paragraph is not None:
                context_len = len(original_paragraph.sentences)

                if context_len > script_cfg.filters.max_original_context_length:
                    _logger().debug('Skipping paragraph %s: context length %d exceeds maximum %d',
                                    para_info, context_len,
                                    script_cfg.filters.max_original_context_length)
                    continue

            for utt_info in para_info.utterances:

                if not _should_enrich_utterance(utt_info,
                                                dict(script_cfg.filters)):
                    _logger().debug('Skipping utterance %s: did not pass filtering', utt_info)
                    continue

                if enriched_dir_handler.contains_contexts_for(utt_info):
                    _logger().debug('Skipping utterance %s: already enriched', utt_info)
                    continue

                yield utt_info, para_info

    utterances_to_enrich_l = list(iter_filtered_utterances())
    random.shuffle(utterances_to_enrich_l)

    utterances_to_enrich = itertools.islice(utterances_to_enrich_l,
                                                script_cfg.max_enriched_utterances)
    
    num_utterances = sum(1 for _ in itertools.islice(utterances_to_enrich_l,
                                                     script_cfg.max_enriched_utterances))

    for utt_info, para_info in tqdm.tqdm(utterances_to_enrich,
                               desc='Enriching context',
                               dynamic_ncols=True,
                               total=num_utterances,
                               miniters=1,
                               unit='utterances',
                               colour='#115b80'):

        _logger().debug('Enriching utterance: %s', utt_info)

        _enrich_utterance_and_save(enricher,
                                   ds_metadata,
                                   raw_path_handler.get_original_paragraph(para_info),
                                   utt_info,
                                   script_cfg.num_contexts_to_generate,
                                   enriched_dir_handler.path_for_utt_contexts(utt_info))


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
