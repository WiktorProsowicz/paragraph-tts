"""Contains definition of acoustic model's decoder module."""
import torch
from comp_trans_tts.model.transformers import conformer
import pydantic

from paragraph_tts.layers.shared import permute_former_att
from paragraph_tts.utils import neural as neural_utils


class Decoder(torch.nn.Module):
    """Decodes encoded representations into mel-spectrogram frames."""

    class Configuration(pydantic.BaseModel):
        """Input configuration for acoustic decoder."""

        hidden_size: int
        n_att_blocks: int
        conformer_kernel_size: int
        n_mel_channels: int
        att_feature_map_dim: int
        num_att_heads: int
        blocks_dropout_rate: float
        postnet_dropout_rate: float

    def __init__(self, cfg: Configuration):

        super().__init__()

        self._blocks = torch.nn.ModuleList(
            [
                conformer.ConformerBlock(
                    attention_module=permute_former_att.PermuteFormerMHA(
                        d_model=cfg.hidden_size,
                        num_heads=cfg.num_att_heads,
                        feature_map_dim=cfg.att_feature_map_dim
                    ),
                    encoder_dim=cfg.hidden_size,
                    feed_forward_dropout_p=cfg.blocks_dropout_rate,
                    attention_dropout_p=cfg.blocks_dropout_rate,
                    conv_dropout_p=cfg.blocks_dropout_rate,
                    conv_kernel_size=cfg.conformer_kernel_size
                )
                for _ in range(cfg.n_att_blocks)
            ]
        )

        self._post_net = torch.nn.Sequential(
            torch.nn.Linear(cfg.hidden_size, cfg.hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=cfg.postnet_dropout_rate),
            torch.nn.Linear(cfg.hidden_size, cfg.n_mel_channels)
        )

    def forward(self,
                encoder_output: torch.Tensor,
                input_length: torch.Tensor) -> torch.Tensor:
        """Generates mel-spectrogram frames from encoded linguistic representations.

        Args:
            encoder_outputs: Tensor of shape [B, L, H] containing encoded linguistic reprs.
            input_lengths: Tensor of shape [B] containing lengths of the input sequences.
        """

        sequence_mask = neural_utils.binary_mask_from_lengths(input_length)

        outputs = encoder_output

        for block in self._blocks:
            outputs = block(outputs,
                            input_mask=sequence_mask)

        return self._post_net(outputs).transpose(1, 2)  # type: ignore[no-any-return]
