# -*- coding: utf-8 -*-
"""Contains audio processing utilities."""
from typing import Tuple

import comp_trans_tts  # type: ignore
import librosa
import numpy as np


class AudioProcessor:
    """Processes audio data."""

    def __init__(self,
                 sr: int,
                 hop_length: int,
                 win_length: int,
                 n_mels: int,
                 fmin: int,
                 fmax: int,
                 trim_top_db: int):
        """Initializes the audio processor.

        Args:
            sr: Sampling rate.
            hop_length: Hop length for STFT.
            win_length: Window length for STFT.
            n_mels: Number of Mel bands.
            fmin: Minimum frequency for Mel filter bank.
            fmax: Maximum frequency for Mel filter bank.
            trim_top_db: Loudness level used to detect leading and trailing silence.
        """

        self._sr = sr
        self._hop_length = hop_length
        self._pitch_extract_cfg = {'preprocessing': {
            'audio': {'sampling_rate': self._sr},
            'stft': {'hop_length': self._hop_length}
        }}
        self._trim_top_db = trim_top_db
        self._win_length = win_length

        self._stft_transform = comp_trans_tts.audio.stft.TacotronSTFT(
            filter_length=win_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mel_channels=n_mels,
            sampling_rate=sr,
            mel_fmin=fmin,
            mel_fmax=fmax)

    def extract_spec_energy_f0(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts a waveform to its spectrogram and extracts energy and F0 contour.

        Args:
            wav: Input waveform.
        """
        duration = int(len(wav) / self._hop_length)
        spec, energy = comp_trans_tts.audio.tools.get_mel_from_wav(
            wav, self._stft_transform)

        spec = spec[:, :duration]
        energy = energy[:duration]

        f0, _ = comp_trans_tts.utils.pitch_tools.get_pitch(wav, spec.T, self._pitch_extract_cfg)

        return spec, energy, f0

    def load_wav(self,
                 file_path: str) -> np.ndarray:
        """Loads a waveform from a file.

        Args:
            file_path: Path to the audio file.
            sr: Sampling rate to use.
        """

        _, wav, _ = comp_trans_tts.preprocessor.preprocessor.Preprocessor.load_audio(
            file_path, self._sr, self._trim_top_db, self._hop_length, self._win_length
        )

        return wav

    def load_wav_raw(self, file_path: str) -> np.ndarray:
        """Loads a waveform from a file without any trimming or padding.

        Args:
            file_path: Path to the audio file.
        """

        wav, _ = librosa.load(file_path, sr=self._sr)

        return wav

    def length_in_sec_of_file(self, file_path: str) -> float:
        """Calculates length of the given wav file in seconds without reading its payload."""

        return librosa.get_duration(path=file_path, sr=self._sr)
