"""Utilities for performing EDA or processed ds analysis."""
from typing import Any

import pandas as pd
import seaborn as sns


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
COOL_COLOR_STD = '#67c2c7'
WARM_COLOR_STD = '#d96891'
NEUTRAL_COLOR_STD = '#bbbbbb'


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


def get_coolwarm_cmap() -> Any:
    """Returns the coolwarm colormap used for EDA plots."""

    return sns.blend_palette(
        colors=[COOL_COLOR_STD, NEUTRAL_COLOR_STD, WARM_COLOR_STD],
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
