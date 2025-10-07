"""Contains utilities for visualization during training/inference."""

import torch
import matplotlib.pyplot as plt


def plot_spectrograms(pred_spec: torch.Tensor,
                      target_spec: torch.Tensor):
    """Plots predicted and target mel-spectrograms side by side."""

    fig, axs = plt.subplots(2, 1, figsize=(10, 4))

    max_length = max(pred_spec.shape[1], target_spec.shape[1])

    pred_spec = torch.nn.functional.pad(pred_spec,
                                        (0, max_length - pred_spec.shape[1]),
                                        value=0.0)
    target_spec = torch.nn.functional.pad(target_spec,
                                          (0, max_length - target_spec.shape[1]),
                                          value=0.0)

    axs[0].imshow(pred_spec.cpu().numpy(), origin='lower')
    axs[0].set_title('Predicted Mel-Spectrogram')

    axs[1].imshow(target_spec.cpu().numpy(), origin='lower')
    axs[1].set_title('Target Mel-Spectrogram')

    mae = torch.mean(torch.abs(pred_spec - target_spec)).item()

    fig.suptitle(f'Mel-Spectrograms (MAE: {mae:.4f})')
    fig.tight_layout()

    return fig

def plot_spec_text_alignment(alignment: torch.Tensor):
    """Plots alignment matrix between text and spectrogram frames."""

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.imshow(alignment.cpu().numpy().T, origin='lower', interpolation='none')
    ax.set_xlabel('Spectrogram Frame Index')
    ax.set_ylabel('Text Token Index')
    ax.set_title('Alignment Matrix')

    fig.tight_layout()

    return fig

def plot_contours(pred_contour: torch.Tensor,
                  target_contour: torch.Tensor,
                  contour_name: str):
    """Plots predicted and target contours (pitch/energy/duration) over time."""

    fig, ax = plt.subplots(figsize=(10, 4))

    max_length = max(pred_contour.shape[0], target_contour.shape[0])

    pred_contour = torch.nn.functional.pad(pred_contour,
                                          (0, max_length - pred_contour.shape[0]),
                                          value=0.0)
    target_contour = torch.nn.functional.pad(target_contour,
                                            (0, max_length - target_contour.shape[0]),
                                            value=0.0)
    
    ax.plot(pred_contour.cpu().numpy(), label='Predicted', color='blue')
    ax.plot(target_contour.cpu().numpy(), label='Target', color='orange')
    ax.set_title(f'{contour_name} Contours')
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'{contour_name} Value')
    ax.legend()

    fig.tight_layout()

    return fig