"""Contains audio processing utilities."""

from typing import Tuple

import torch
import librosa
import numpy as np


def load_wav(file_path: str,
             sr: int,
             trip_top_db: int,
             filter_length: int,
             hop_length: int) -> Tuple[np.ndarray, int]:
    """Loads a waveform from a file.

    Args:
        file_path: Path to the audio file.
        sr: Sampling rate to use.
        trim_top_db: Threshold (in decibels) below reference to consider as silence.
        filter_length: Filter length used while trimming top db.
        hop_length: Hop length used while trimming top db.

    Returns:
        Waveform as a 1D tensor and duration.
    """

    wav_raw, _ = librosa.load(file_path, sr=sr)
    wav, index = librosa.effects.trim(wav_raw,
                                    top_db=trip_top_db,
                                    frame_length=filter_length,
                                    hop_length=hop_length)
    duration = (index[1] - index[0]) / hop_length
    return wav.astype(np.float32), int(duration)


class
