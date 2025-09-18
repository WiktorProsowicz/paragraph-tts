"""Generates enriched context for dataset samples using a language model."""

import json
import os
import asyncio
import logging
import sys
import time

import tqdm
import hydra
import omegaconf

from paragraph_tts.data import enrichment
from paragraph_tts.utils.path import (raw_libri_dir_handler, enriched_context_dir_handler)
from paragraph_tts.utils import logging_utils
from paragraph_tts.data import librittsr_helpers
from paragraph_tts.data.preprocessing import text as text_prep


def _logger():
    return logging.getLogger(__name__)


def _prepare_utterance_for_enrichment(original_paragraph: raw_libri_dir_handler.OriginalParagraph,
                                      utt_info: raw_libri_dir_handler.UtteranceInfo,
                                      ds_metadata: librittsr_helpers.LibriTTSRMetadata
                                      ) -> enrichment.UtteranceForEnrichment:
    preceding_sentences = [
        original_paragraph.sentences[k] for k in sorted(original_paragraph.sentences)
        if k < utt_info.utt_id
    ]

    following_sentences = [
        original_paragraph.sentences[k] for k in sorted(original_paragraph.sentences)
        if k > utt_info.utt_id
    ]

    utt_text = text_prep.TextProcessor.load_text(utt_info.text_path)

    speaker, book = ds_metadata.get_speaker_and_book(original_paragraph.spk_id,
                                                     original_paragraph.chap_id)

    return enrichment.UtteranceForEnrichment(
        text=utt_text,
        original_preceding_sentences=preceding_sentences,
        original_following_sentences=following_sentences,
        book=book,
        speaker=speaker
    )


async def _gather_contexts(promises):
    return await asyncio.gather(*(promises))


def _enrich_paragraph_and_save(ds_metadata: librittsr_helpers.LibriTTSRMetadata,
                               raw_path_handler: raw_libri_dir_handler.RawLibriDirHandler,
                               para_info: raw_libri_dir_handler.ParagraphInfo,
                               script_cfg: omegaconf.DictConfig):

    contexts_dir_handler = enriched_context_dir_handler.EnrichedContextDirHandler(
        script_cfg.output_path
    )

    original_paragraph = raw_path_handler.get_original_paragraph(para_info)

    utterances_to_enrich = [utt_info for utt_info in para_info.utterances
                            if not contexts_dir_handler.contains_contexts_for(para_info, utt_info)]

    for utt_info in utterances_to_enrich:

        utterance_for_enrichment = _prepare_utterance_for_enrichment(
            original_paragraph, utt_info, ds_metadata
        )

        enricher = enrichment.ContextEnricher(model_name=script_cfg.model_name,
                                        ollama_host=script_cfg.ollama_host,
                                        max_paragraph_len=script_cfg.max_paragraph_len,
                                        min_paragraph_len=script_cfg.min_paragraph_len)

        promises = (
            enricher.generate_context_for_utt(utterance_for_enrichment)
            for _ in range(script_cfg.num_contexts_to_generate)
        )

        contexts = asyncio.run(_gather_contexts(promises))

        if any(c is None for c in contexts):
            _logger().warning('Failed to generate contexts for utterance: %s', utt_info)
            continue

        utterance_contexts_path = contexts_dir_handler.path_for_utt_contexts(para_info, utt_info)
        os.makedirs(os.path.dirname(utterance_contexts_path), exist_ok=True)

        with open(utterance_contexts_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, indent=4, ensure_ascii=False)


@hydra.main(version_base=None, config_path="cfg", config_name="enrich_context")
def main(script_cfg: omegaconf.DictConfig):
    """Performs context enrichment for dataset samples."""

    logging_utils.setup_logging('enrich_context')

    os.makedirs(script_cfg.output_path, exist_ok=True)

    metadata = {
        'model_name': script_cfg.model_name,
        'max_paragraph_len': script_cfg.max_paragraph_len,
        'min_paragraph_len': script_cfg.min_paragraph_len,
        'num_contexts_to_generate': script_cfg.num_contexts_to_generate
    }

    with open(os.path.join(script_cfg.output_path, 'metadata.json'), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

    if not enrichment.ContextEnricher.is_model_available(script_cfg.model_name,
                                                         script_cfg.ollama_host):
        _logger().critical('The following model is unavailable at the Ollama server: %s',
                           script_cfg.model_name)
        sys.exit(1)

    raw_path_handler = raw_libri_dir_handler.RawLibriDirHandler(script_cfg.raw_ds_path)

    ds_metadata = librittsr_helpers.LibriTTSRMetadata()

    for para_info in tqdm.tqdm(raw_path_handler.iter_all_paragraphs(),
                               desc='Enriching context',
                               dynamic_ncols=True,
                               miniters=1,
                               unit='paragraphs',
                               colour='#115b80'):

        _logger().debug('Enriching paragraph: %s', para_info)
        _enrich_paragraph_and_save(ds_metadata, raw_path_handler, para_info, script_cfg)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
