"""Contains definition of acoustic model's encoder module."""


import torch
from comp_trans_tts.model.transformers import conformer

from paragraph_tts.layers.shared import permute_former_att
from paragraph_tts.utils import neural as neural_utils


class Encoder(torch.nn.Module):
    """Encodes input phonemes, BERT embeddings and linguistic stats."""

    def __init__(self,
                 hidden_size: int,
                 phoneme_vocab_size: int,
                 pos_tags_vocab_size: int,
                 ling_stats_dim: int,
                 n_att_blocks: int,
                 att_feature_map_dim: int,
                 num_att_heads: int,
                 prenet_dropout_rate: float,
                 blocks_dropout_rate: float):

        super().__init__()

        self._phoneme_emb = torch.nn.Embedding(
            num_embeddings=phoneme_vocab_size,
            embedding_dim=hidden_size
        )

        self._pos_tags_emb = torch.nn.Embedding(
            num_embeddings=pos_tags_vocab_size,
            embedding_dim=pos_tags_vocab_size // 2
        )

        self._ling_stats_fc = torch.nn.Linear(
            in_features=ling_stats_dim + (pos_tags_vocab_size // 2),
            out_features=hidden_size
        )

        SENTENCE_POS_EMB_DIM = 8

        self._sentence_pos_emb = torch.nn.Embedding(
            num_embeddings=4,
            embedding_dim=SENTENCE_POS_EMB_DIM
        )

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2 + SENTENCE_POS_EMB_DIM + 1, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=prenet_dropout_rate),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=prenet_dropout_rate)
        )

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

    def forward(self,
                phoneme_ids: torch.Tensor,
                pos_tag_ids: torch.Tensor,
                ling_stats: torch.Tensor,
                sentence_pos_ids: torch.Tensor,
                spk_rate: torch.Tensor,
                input_length: torch.Tensor) -> torch.Tensor:
        """Encodes input sequences.

        Args:
            phoneme_ids: Tensor of shape [B, L] with input phoneme IDs.
            pos_tag_ids: Tensor of shape [B, L] with input POS tag IDs.
            ling_stats: Tensor of shape [B, L, D] with input linguistic features.
            sentence_pos_ids: Tensor of shape [B] with sentence position IDs. (e.g. "1- middle")
            spk_rate: Tensor of shape [B] with speaker's speech rate.
            input_length: Tensor of shape [B] with lengths of input sequences.
        """

        sequence_mask = neural_utils.binary_mask_from_lengths(input_length)

        phoneme_emb = self._phoneme_emb(phoneme_ids)
        pos_tags_emb = self._pos_tags_emb(pos_tag_ids)
        sent_pos_emb = self._sentence_pos_emb(sentence_pos_ids)

        sent_pos_emb = sent_pos_emb.unsqueeze(1).expand(-1, phoneme_emb.size(1), -1)
        spk_rate = spk_rate.unsqueeze(-1).unsqueeze(-1).expand(-1, phoneme_emb.size(1), 1)

        encoded_ling = self._ling_stats_fc(torch.cat([ling_stats, pos_tags_emb], dim=-1))
        prenet_input = torch.cat([phoneme_emb, encoded_ling, sent_pos_emb, spk_rate], dim=-1)

        outputs = self._prenet(prenet_input)

        for block in self._blocks:
            outputs = block(
                inputs=outputs,
                input_mask=sequence_mask
            )

        return outputs
