"""Contains utilities used in neural networks."""
import torch


def calc_decayed_loss_weight(initial_weight: float, decay_rate: float, epoch: int) -> float:
    """Calculates decayed loss weight based on initial weight, decay rate and current epoch."""
    return initial_weight * (decay_rate ** epoch)


def binary_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    """Creates binary mask from a batch of lengths."""

    if lengths.dim() == 0:
        mask = torch.zeros((int(lengths.item()),), dtype=torch.bool, device=lengths.device)
        mask[:int(lengths.item())] = 1

    else:
        batch_size = lengths.size(0)

        max_len = int(torch.max(lengths).item())
        mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=lengths.device)

        for i in range(batch_size):
            mask[i, :lengths[i]] = 1

    return mask
