"""Utilities for performing EDA or processed ds analysis."""
from typing import Any
import pathlib

import pandas as pd
import seaborn as sns
import numpy as np
import soundfile

from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.data.preprocessing import audio as audio_prep


def save_example_paragraph(paragraph: raw_libri_dir_handler.ParagraphInfo,
                           output_dir: pathlib.Path) -> None:
    """Saves an example paragraph to the specified directory."""

    output_dir.mkdir(parents=True, exist_ok=True)

    with output_dir.joinpath('sentences.txt').open('w', encoding='utf-8') as f:
        for utt in paragraph.utterances:
            f.write(f'{utt.utt_id:3d}\t{utt.normalized_text}\n')

    if all(utt.wav_path is not None for utt in paragraph.utterances):

        full_wav = np.concatenate([
            audio_prep.load_wav_raw(utt.wav_path) for utt in paragraph.utterances
        ], axis=0)

        soundfile.write(output_dir.joinpath('audio.wav'), full_wav, 22050)


def is_outlier(series: pd.Series) -> pd.Series:
    """Determines what sentences are outliers using the IQR method."""

    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    return (series < lower_bound) | (series > upper_bound)


DARK_COLOR_STD = '#24403e'
LIGHT_COLOR_STD = '#67e0da'
MIDDLE_COLOR_STD = '#4A8F8E'


def get_gradient_palette(n: int) -> Any:
    """Returns a gradient colormap used for EDA plots."""

    return sns.blend_palette(
        colors=[DARK_COLOR_STD, LIGHT_COLOR_STD],
        n_colors=n,
        as_cmap=False
    )


def get_gradient_cmap() -> Any:
    """Returns a gradient colormap used for EDA plots."""

    return sns.blend_palette(
        colors=[DARK_COLOR_STD, LIGHT_COLOR_STD],
        as_cmap=True
    )


def get_gradient_palette_reversed(n: int) -> Any:
    """Returns a reversed gradient palette (light to dark) for violin plots."""

    return sns.blend_palette(
        colors=[LIGHT_COLOR_STD, DARK_COLOR_STD],
        n_colors=n,
        as_cmap=False
    )


def get_gradient_cmap_reversed() -> Any:
    """Returns a reversed gradient colormap (light to dark) for violin plots."""

    return sns.blend_palette(
        colors=[LIGHT_COLOR_STD, DARK_COLOR_STD],
        as_cmap=True
    )
