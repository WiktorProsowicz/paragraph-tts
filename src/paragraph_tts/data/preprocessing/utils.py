"""Contains utilities for data preprocessing."""

from typing import Dict, Any
from collections import defaultdict

from paragraph_tts.utils.path import processed_libri_dir_handler


def compose_processed_ds_stats(processed_ds_path: str) -> Dict[str, Any]:
    """Collects statistics based on a processed dataset.

    Args:
        processed_ds_path: Path to the processed dataset.
    """

    dir_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(processed_ds_path)

    context_sentences_count = defaultdict(int)

    for sample in dir_handler.iter_samples():
        for context in sample.contexts:
            if 'length' in context.metadata:
                context_sentences_count[context.metadata['length']] += 1

            else:
                length = context.metadata['n_preceding_sentences']
                length += context.metadata['n_following_sentences']
                length += 1

                context_sentences_count[length] += 1

    return {
        'n_utterances': sum(1 for _ in dir_handler.iter_samples()),
        'n_speakers': len({sample.spk_id for sample in dir_handler.iter_samples()}),
        'context_sentences_count': dict(context_sentences_count)
    }