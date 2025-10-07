"""Contains utilities for inference with trained models."""

from typing import List, Callable

import numpy as np
import torch


def split_spectrogram_by_silences(spec: torch.Tensor,
                                  energy_threshold_percentile: float = 20.0,
                                  min_silence_length: int = 11
                                  ) -> List[torch.Tensor]:
    """Splits mel-spectrogram into chunks separated by silences."""

    frame_energy = np.mean(spec.cpu().numpy(), axis=0)

    thresh = np.percentile(frame_energy, energy_threshold_percentile)
    is_frame_silent = frame_energy < thresh

    is_frame_silent = np.concatenate(([0], is_frame_silent, [0]))
    diff = np.diff(is_frame_silent.astype(np.int8))
    silence_starts = np.where(diff == 1)[0]
    silence_ends = np.where(diff == -1)[0]

    silences = [((start + end) // 2, end - start)
                for start, end in zip(silence_starts, silence_ends)]
    silences = [sil for sil in silences if sil[1] >= min_silence_length]

    split_points = [0] + [silence[0] for silence in silences] + [spec.shape[1]]

    slices = []

    for slice_start, slice_end in zip(split_points[:-1], split_points[1:]):
        slices.append(spec[:, slice_start:slice_end])

    return slices


def transform_mel_to_wav(mel: torch.Tensor,
                         vocoder: Callable[[torch.Tensor], torch.Tensor],
                         split_spec_by_silences: bool = True) -> torch.Tensor:
    """Transforms mel-spectrogram to waveform using a vocoder model."""

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