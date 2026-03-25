"""Contains audio processing utilities."""
from typing import Tuple
from typing import Annotated

import comp_trans_tts
import librosa
import numpy as np
import pydantic
from pydantic import Field


def length_in_sec_of_file(file_path: str) -> float:
    """Calculates length of the given wav file in seconds without reading its payload."""

    return librosa.get_duration(path=file_path, sr=librosa.get_samplerate(file_path))


def load_wav_raw(file_path: str, sr: int | None = None) -> np.ndarray:
    """Loads a waveform from a file without any trimming or padding.

    Args:
        file_path: Path to the audio file.
        sr: Sampling rate. If None, the sampling rate is inferred from the file.
    """

    wav, _ = librosa.load(file_path, sr=sr)

    return wav


class AudioProcessor:
    """Processes audio data."""

    class Configuration(pydantic.BaseModel):
        """Configuration of the audio processor."""

        sr: Annotated[int, Field(description='Sampling rate.')]
        hop_length: Annotated[int, Field(description='Hop length for STFT.')]
        win_length: Annotated[int, Field(description='Window length for STFT.')]
        n_mels: Annotated[int, Field(description='Number of Mel bands.')]
        fmin: Annotated[int, Field(description='Minimum frequency for Mel filter bank.')]
        fmax: Annotated[int, Field(description='Maximum frequency for Mel filter bank.')]
        trim_top_db: Annotated[int, Field(
            description='Loudness level used to detect leading and trailing silence.')]

    def __init__(self, cfg: Configuration):
        """Initializes the audio processor."""

        self._pitch_extract_cfg = {'preprocessing': {
            'audio': {'sampling_rate': cfg.sr},
            'stft': {'hop_length': cfg.hop_length}
        }}
        self._trim_top_db = cfg.trim_top_db
        self._win_length = cfg.win_length

        self._stft_transform = comp_trans_tts.audio.stft.TacotronSTFT(
            filter_length=cfg.win_length,
            hop_length=cfg.hop_length,
            win_length=cfg.win_length,
            n_mel_channels=cfg.n_mels,
            sampling_rate=cfg.sr,
            mel_fmin=cfg.fmin,
            mel_fmax=cfg.fmax)

        self._cfg = cfg

    def extract_spec_energy_f0(self, wav: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Converts a waveform to its spectrogram and extracts energy and F0 contour.

        Args:
            wav: Input waveform.
        """
        duration = int(len(wav) / self._cfg.hop_length)
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
            file_path, self._cfg.sr, self._cfg.trim_top_db, self._cfg.hop_length,
            self._cfg.win_length
        )

        return np.asanyarray(wav)
