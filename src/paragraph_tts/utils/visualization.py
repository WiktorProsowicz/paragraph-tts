"""Contains utilities for visualization during training/inference."""
import pathlib
import logging

import torch
from speechbrain.inference.vocoders import HIFIGAN
import soundfile
from matplotlib import pyplot as plt
from torch_dev_utils.tts import visualization as tdu_viz
from paragraph_tts.utils import inference as inference_utils


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def visualize_acoustic_model_outputs(sample: dict[str, torch.Tensor],
                                     model_output: dict[str, torch.Tensor],
                                     hifi_gan: HIFIGAN,
                                     save_target_wav: bool,
                                     output_dir: pathlib.Path) -> None:
    """Visualizes model outputs (spectrograms, pitch/energy, output wav)."""

    _logger().debug('Visualizing and saving outputs to %s.', output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec_length = int(sample['input_spec_length'].item())
    predicted_spec_length = int(model_output['durations_rounded'].sum().item())
    prosody_features_length = int(sample['prosody_features_length'].item())
    phonemes_length = int(sample['input_phonemes_length'].item())
    words_length = int(sample['input_word_emb_length'].item())

    if prosody_features_length == spec_length:
        predicted_prosody_features_length = predicted_spec_length

    else:
        predicted_prosody_features_length = phonemes_length

    tdu_viz.plot_and_save_spectrograms(
        model_output['pred_mel_spec'][:, :spec_length].numpy(),
        sample['input_spec'][:, :spec_length].numpy(),
        sr=22050,
        hop_length=256,
        output_path=output_dir.joinpath('spectrograms_target_length.svg')
    )

    tdu_viz.plot_and_save_spectrograms(
        model_output['pred_mel_spec'][:, :predicted_spec_length].numpy(),
        sample['input_spec'][:, :spec_length].numpy(),
        sr=22050,
        hop_length=256,
        output_path=output_dir.joinpath('spectrograms_predicted_length.svg')
    )

    if 'wsv_weights' in model_output:

        tdu_viz.plot_and_save_matrix(
            model_output['wsv_weights'][:words_length],
            'WSV Weights',
            'Token Index',
            'Word Index',
            output_dir.joinpath('wsv_weights.svg')
        )

    if 'gst_weights' in model_output:

        tdu_viz.plot_and_save_contour(
            model_output['gst_weights'],
            'GST Weights',
            output_dir.joinpath('gst_weights.svg')
        )

    if 'wsv_emb' in model_output:

        tdu_viz.plot_and_save_matrix(
            model_output['wsv_emb'][:words_length],
            'WSV Embeddings',
            'Word Index',
            'Embedding Dimension',
            output_dir.joinpath('wsv_embeddings.svg')
        )

    if 'gst_emb' in model_output:

        tdu_viz.plot_and_save_contour(
            model_output['gst_emb'],
            'GST Embeddings',
            output_dir.joinpath('gst_embedding.svg')
        )

    if 'target_pitch' in model_output:

        tdu_viz.plot_and_save_contours(
            model_output['predicted_pitch'][:prosody_features_length],
            model_output['target_pitch'][:prosody_features_length],
            'Pitch',
            output_dir.joinpath('pitch_contours.svg')
        )

    else:

        tdu_viz.plot_and_save_contour(
            sample['input_f0'][:predicted_prosody_features_length],
            'Pitch',
            output_dir.joinpath('pitch_contour.svg')
        )

    if 'target_energy' in model_output:

        tdu_viz.plot_and_save_contours(
            model_output['predicted_energy'][:prosody_features_length],
            model_output['target_energy'][:prosody_features_length],
            'Energy',
            output_dir.joinpath('energy_contour.svg')
        )

    else:

        tdu_viz.plot_and_save_contour(
            sample['input_energy'][:predicted_prosody_features_length],
            'Energy',
            output_dir.joinpath('energy_contour.svg')
        )

    tdu_viz.plot_and_save_contours(
        model_output['predicted_durations'][:phonemes_length],
        sample['explicit_durations'][:phonemes_length],
        'Duration',
        output_dir.joinpath('duration_contour.svg')
    )

    wav = inference_utils.transform_mel_to_wav(
        model_output['pred_mel_spec'][:, :predicted_spec_length],
        hifi_gan.decode_batch,
        split_spec_by_silences=True
    )

    if wav is not None:
        soundfile.write(output_dir.joinpath('predicted.wav'),
                        wav.squeeze(0).numpy(), 22050)

    if not save_target_wav:
        return

    wav = inference_utils.transform_mel_to_wav(
        sample['input_spec'][:, :spec_length],
        hifi_gan.decode_batch,
        split_spec_by_silences=True
    )

    if wav is not None:
        soundfile.write(output_dir.joinpath('target.wav'),
                        wav.squeeze(0).numpy(), 22050)


def plot_and_save_gst_prediction(pred_gst_weights: torch.Tensor,
                                 target_gst_weights: torch.Tensor,
                                 output_path: pathlib.Path) -> None:
    """Plots and saves the predicted vs target GST weights."""

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.plot(pred_gst_weights, label='Predicted GST Weights')
    ax.plot(target_gst_weights, label='Target GST Weights')

    mae = float(torch.mean(torch.abs(pred_gst_weights - target_gst_weights)).item())

    ax.set_title(f'Predicted vs Target GST Weights. MAE={mae:.4f}')
    ax.set_xlabel('GST Node Index')
    ax.set_ylabel('Weight Value')
    ax.legend()

    plt.close(fig)
    fig.savefig(output_path)


def plot_and_save_wsv_prediction(pred_wsv_weights: torch.Tensor,
                                 target_wsv_weights: torch.Tensor,
                                 output_path: pathlib.Path) -> None:
    """Plots and saves the predicted vs target WSV weights."""

    n_words = int(pred_wsv_weights.shape[0])

    fig, axes = plt.subplots(nrows=n_words,
                             figsize=(8, 3 * n_words),
                             sharex=True)

    for i in range(n_words):

        axes[i].plot(pred_wsv_weights[i].numpy(), label='Predicted WSV Weights')
        axes[i].plot(target_wsv_weights[i].numpy(), label='Target WSV Weights')

        mae = float(torch.mean(torch.abs(pred_wsv_weights[i] - target_wsv_weights[i])))

        axes[i].set_title(f'Word {i} WSV Weights. MAE: {mae:.4f}')
        axes[i].set_ylabel('WSV Weight')

    axes[-1].set_xlabel('Token index')
    axes[0].legend()

    fig.tight_layout()

    plt.close(fig)
    fig.savefig(output_path)
