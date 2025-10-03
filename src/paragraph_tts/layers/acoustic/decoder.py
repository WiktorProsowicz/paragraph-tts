"""Contains definition of acoustic model's decoder module."""


import torch
from comp_trans_tts.model.transformers import conformer

from paragraph_tts.layers.shared import permute_former_att
from paragraph_tts.utils import neural as neural_utils


class Decoder(torch.nn.Module):
    """Decodes encoded representations into mel-spectrogram frames."""

    def __init__(self,
                 hidden_size: int,
                 n_att_blocks: int,
                 n_mel_channels: int,
                 att_feature_map_dim: int,
                 num_att_heads: int,
                 blocks_dropout_rate: float,
                 postnet_dropout_rate: float):

        super().__init__()

        self._blocks = torch.nn.ModuleList(
            [
                conformer.ConformerBlock(
                    attention_module=permute_former_att.PermuteFormerMHA(
                        d_model=hidden_size,
                        num_heads=num_att_heads,
                        feature_map_dim=att_feature_map_dim
                    ),
                    encoder_dim=hidden_size,
                    feed_forward_dropout_p=blocks_dropout_rate,
                    attention_dropout_p=blocks_dropout_rate,
                    conv_dropout_p=blocks_dropout_rate
                )
                for _ in range(n_att_blocks)
            ]
        )

        self._post_net = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=postnet_dropout_rate),
            torch.nn.Linear(hidden_size, n_mel_channels)
        )

    def forward(self,
                encoder_outputs: torch.Tensor,
                input_lengths: torch.Tensor) -> torch.Tensor:
        """Generates mel-spectrogram frames from encoded linguistic representations.
        
        Args:
            encoder_outputs: Tensor of shape [B, L, H] containing encoded linguistic reprs.
            input_lengths: Tensor of shape [B] containing lengths of the input sequences.
        """

        sequence_mask = neural_utils.binary_mask_from_lengths(input_lengths)

        outputs = encoder_outputs

        for block in self._blocks:
            outputs = block(outputs,
                            query_mask=sequence_mask,
                            input_mask=sequence_mask)

        return self._post_net(outputs)
