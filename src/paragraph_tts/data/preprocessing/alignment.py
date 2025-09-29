"""Contains utilities for processing text-audio alignments."""

from typing import List, Tuple
from typing import TypeAlias
import collections

import numpy as np
import tgt

from paragraph_tts.data.preprocessing import text as text_prep


def spans_to_indices_of_smaller_seq(spans: np.ndarray) -> np.ndarray:
    """Convert spans to indices.

    Args:
        spans: (N,) array of integer spans. The elements indicate how many elements of the larger
            sequence correspond to the i-th element of the smaller sequence.

    Returns:
        (sum(spans),) array of indices. The elements indicate which element of the smaller sequence
                corresponds to the i-th element of the larger sequence.
    """

    return np.repeat(np.arange(len(spans)), spans)


def spans_to_pool_matrix(spans: np.ndarray) -> np.ndarray:
    """Converts sequence of integer spans to a pooling matrix.

    Args:
        spans: (N,) array of integer spans. The elements indicate how many
        elements of the larger sequence correspond to the i-th element of the
        smaller sequence.

        Returns:
                (sum(spans), N) pooling matrix. The matrix, once multiplied by the larger
        sequence will produce a sequence of the smaller size, with spanned elements averaged.
    """

    total_length = np.sum(spans)
    large_to_small_mapping = spans_to_indices_of_smaller_seq(spans)

    mat_value = (1.0 / spans + 1e-6) * (spans > 0).astype(np.float32)

    pool_matrix = np.zeros((total_length, len(spans)), dtype=np.float32)
    pool_matrix[np.arange(total_length), large_to_small_mapping] = np.repeat(mat_value, spans)

    return pool_matrix


def _trim_silences(intervals: List[tgt.Interval]) -> List[tgt.Interval]:
    """Trims leading and trailing silences from a list of intervals."""

    first_idx = next(i for i, interval in enumerate(intervals) if interval.text != '')
    last_idx = len(intervals) - next(i for i, interval
                                     in enumerate(reversed(intervals))
                                     if interval.text != '')
    return intervals[first_idx: last_idx]


def _stretch_words_to_breaks(word_intervals: List[tgt.Interval]) -> List[tgt.Interval]:
    """Stretches word intervals to fill in silences between them."""

    new_words: List[tgt.Interval] = []

    for word_interval in word_intervals:

        if word_interval.text == '':
            new_words[-1].end_time = word_interval.end_time
            continue

        new_words.append(tgt.Interval(
            start_time=word_interval.start_time,
            end_time=word_interval.end_time,
            text=word_interval.text))

    return new_words


WordPhonemeMapping: TypeAlias = List[Tuple[tgt.Interval, List[tgt.Interval]]]


def _map_words_to_phones(word_intervals: List[tgt.Interval],
                         phone_intervals: List[tgt.Interval]
                         ) -> WordPhonemeMapping:
    """Maps word intervals to corresponding phone intervals."""

    phone_mapping: WordPhonemeMapping = []

    for word_interval in word_intervals:
        chosen_phones = [
            phone_interval for phone_interval in phone_intervals
            if (phone_interval.start_time >= word_interval.start_time and
                phone_interval.end_time <= word_interval.end_time)
        ]

        phone_mapping.append((word_interval, chosen_phones))

    return phone_mapping


def get_pauses(word_phoneme_mapping: WordPhonemeMapping) -> List[Tuple[int, str]]:
    """Returns positions of words ending with a pause together with the pause types."""

    pause_positions = []

    for idx, (_, phone_intervals) in enumerate(word_phoneme_mapping):
        if phone_intervals[-1].text == '':

            pause_type = text_prep.TextProcessor.get_pause_type(
                phone_intervals[-1].end_time - phone_intervals[-1].start_time)
            pause_positions.append((idx, pause_type))

    return pause_positions


def get_word_phoneme_mapping(alignment: tgt.TextGrid) -> WordPhonemeMapping:
    """Extracts word to phoneme mapping from the given alignment."""

    word_intervals = _trim_silences(
        alignment.get_tier_by_name('words').intervals)
    word_intervals = _stretch_words_to_breaks(word_intervals)

    phone_intervals = _trim_silences(
        alignment.get_tier_by_name('phones').intervals)

    return _map_words_to_phones(word_intervals, phone_intervals)


def _max_times_to_spec_frames(max_times: List[float],
                              spec_length: int) -> np.ndarray:
    """Converts list of max times of tokens to spectrogram frame spans."""

    secs_till_now = np.array(max_times)
    frames_till_now = ((secs_till_now / secs_till_now[-1]) * spec_length).astype(np.int32)
    frames_till_now[1:] = frames_till_now[1:] - frames_till_now[:-1]

    return frames_till_now

def get_phone_to_spec_spans(interval_mapping: WordPhonemeMapping,
                            original_mapping: text_prep.TokenMapping,
                            spec_length: int) -> np.ndarray:
    """Computes mapping from phones to spectrogram frames."""

    max_times = []

    for (word_int, phone_ints), (_, phones) in zip(interval_mapping,
                                              original_mapping):

        if len(phones) != len(phone_ints):
            phone_length = (word_int.end_time - word_int.start_time) / len(phones)
            for i in range(len(phones)):
                max_times.append(word_int.start_time + (i + 1) * phone_length)

        else:
            max_times.extend([phone_int.end_time for phone_int in phone_ints])

    return _max_times_to_spec_frames(max_times, spec_length)

def get_word_to_spec_spans(interval_mapping: WordPhonemeMapping,
                           spec_length: int) -> np.ndarray:
    """Computes mapping from words to spectrogram frames."""

    max_times = [word_int.end_time for (word_int, _) in interval_mapping]

    return _max_times_to_spec_frames(max_times, spec_length)