"""Contains utilities for inference with trained models."""

from typing import List, Callable, Optional

import numpy as np
import torch

from paragraph_tts.utils import neural as neural_utils

MAX_ALLOWED_DURATION = 10000  # To avoid OOM errors during inference


def sanitize_predicted_durations(durations: torch.Tensor,
                                 sequences_lens: torch.Tensor) -> torch.Tensor:
    """Sanitizes predicted durations by rounding and clamping to non-negative values.

    The durations can be either batched (B, T) or unbatched (T,).
    """

    durations *= neural_utils.binary_mask_from_lengths(sequences_lens)

    durations_quant = torch.clamp(torch.round(durations), min=0).long()
    duration_mask = torch.cumsum(durations_quant, dim=-1) <= MAX_ALLOWED_DURATION

    return durations_quant * duration_mask.long()


def split_spectrogram_by_silences(spec: torch.Tensor,
                                  energy_threshold_percentile: float = 20.0,
                                  min_silence_length: int = 11,
                                  min_chunk_length: int = 80
                                  ) -> List[torch.Tensor]:
    """Splits mel-spectrogram into chunks separated by silences."""

    frame_energy = np.mean(spec.cpu().numpy(), axis=0)

    thresh = np.percentile(frame_energy, energy_threshold_percentile)
    is_frame_silent = frame_energy < thresh

    is_frame_silent = np.concatenate(([0], is_frame_silent, [0]))
    diff = np.diff(is_frame_silent.astype(np.int8))
    silence_starts = np.where(diff == 1)[0]
    silence_ends = np.where(diff == -1)[0]

    sil_lens = silence_ends - silence_starts

    sil_points = (silence_starts + silence_ends) // 2
    sil_points = sil_points[sil_lens >= min_silence_length]

    chunk_lens = np.diff(np.concatenate(([0], sil_points, [spec.shape[1]])))

    chunk_lens_l = [chunk_lens[0]] if chunk_lens[0] >= min_chunk_length else [0]
    first_chunk_to_check = 1 if chunk_lens[0] >= min_chunk_length else 0

    for l in chunk_lens[first_chunk_to_check:]:
        if l < min_chunk_length:
            chunk_lens_l[-1] += l

        else:
            chunk_lens_l.append(l)

    return torch.split(spec, chunk_lens_l, dim=1)


def transform_mel_to_wav(mel: torch.Tensor,
                         vocoder: Callable[[torch.Tensor], torch.Tensor],
                         split_spec_by_silences: bool = True) -> Optional[torch.Tensor]:
    """Transforms mel-spectrogram to waveform using a vocoder model."""

    if mel.shape[1] <= 1:
        return None

    if split_spec_by_silences:
        mel_chunks = split_spectrogram_by_silences(mel)

    else:
        mel_chunks = [mel]

    wav_chunks = []

    for chunk in mel_chunks:
        chunk = chunk.unsqueeze(0).to(mel)

        with torch.no_grad():
            wav_chunk = vocoder(chunk).squeeze(0)

        wav_chunks.append(wav_chunk.cpu())

    return torch.cat(wav_chunks, dim=-1)
