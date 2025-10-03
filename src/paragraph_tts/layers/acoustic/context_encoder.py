"""Contains definition of context encoder producing context embeddings for acoustic model."""

import torch

from paragraph_tts.layers.shared import cbhg
from paragraph_tts.layers.shared import permute_former_att
from paragraph_tts.utils import neural as neural_utils


class _ContextProcessingBlock(torch.nn.Module):
    """Processes sequence of context sentence embeddings and attends them into phoneme reprs."""

    def __init__(self,
                 input_emb_dim: int,
                 hidden_size: int,
                 n_blocks: int,
                 dropout_rate: float,
                 num_att_heads: int,
                 att_feature_map_dim: int):

        super().__init__()

        self._prenet = torch.nn.Sequential(
            torch.nn.Linear(input_emb_dim, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate)
        )

        self._blocks = torch.nn.ModuleList(
            [
                cbhg.CBHG(
                    in_dim=hidden_size,
                    K=16,
                    hidden_sizes=[hidden_size, hidden_size]
                )
                for _ in range(n_blocks)
            ]
        )

        self._global_states_enc = torch.nn.Sequential(
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=dropout_rate)
        )

        self._att = permute_former_att.PermuteFormerMHA(
            d_model=hidden_size,
            num_heads=num_att_heads,
            feature_map_dim=att_feature_map_dim
        )

    def forward(self,
                inputs: torch.Tensor,
                input_lengths: torch.Tensor,
                phoneme_representations: torch.Tensor,
                phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """Processes context sentences and produces context embeddings.

        Args:
            inputs: Tensor of shape [B, L, D] containing input context sentence embeddings.
            input_lengths: Tensor of shape [B] containing lengths of the input sequences.
            phoneme_representations: Tensor of shape [B, L_p, H] containing phoneme-level reprs.
            phoneme_lengths: Tensor of shape [B] containing lengths of the phoneme sequences.
        """

        batch_size = inputs.size(0)

        input_mask = neural_utils.binary_mask_from_lengths(input_lengths)
        phoneme_mask = neural_utils.binary_mask_from_lengths(phoneme_lengths)

        outputs = self._prenet(inputs)

        for block in self._blocks:
            outputs = block(outputs, input_lengths)

        global_states_last = outputs[torch.arange(batch_size), input_lengths - 1]
        global_states_first = outputs[:, 0]

        global_states = torch.cat([global_states_first, global_states_last], dim=-1)
        global_states = self._global_states_enc(global_states)

        chosen_context = self._att(
            query=phoneme_representations,
            key=outputs,
            value=outputs,
            key_mask=input_mask,
            query_mask=phoneme_mask
        )

        return chosen_context + phoneme_representations


class ContextEncoder(torch.nn.Module):
    """Encodes context sentences into phoneme-level context embeddings."""

    def __init__(self,
                 input_emb_dim: int,
                 hidden_size: int,
                 n_blocks: int,
                 dropout_rate: float,
                 num_att_heads: int,
                 att_feature_map_dim: int):

        super().__init__()

        self._token_embs_enc = _ContextProcessingBlock(
            input_emb_dim=input_emb_dim,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            dropout_rate=dropout_rate,
            num_att_heads=num_att_heads,
            att_feature_map_dim=att_feature_map_dim
        )

        self._pse_enc = _ContextProcessingBlock(
            input_emb_dim=input_emb_dim,
            hidden_size=hidden_size,
            n_blocks=n_blocks,
            dropout_rate=dropout_rate,
            num_att_heads=num_att_heads,
            att_feature_map_dim=att_feature_map_dim
        )

    def forward(self,
                token_embs: torch.Tensor,
                token_embs_lengths: torch.Tensor,
                pse_embs: torch.Tensor,
                pse_embs_lengths: torch.Tensor,
                phoneme_representations: torch.Tensor,
                phoneme_lengths: torch.Tensor) -> torch.Tensor:
        """Encodes context sentences and produces context embeddings.

        Args:
            token_embs: Tensor of shape [B, L_t, D] containing input token-level context embeddings.
            token_embs_lengths: Tensor of shape [B] containing lengths of the token-level
                context sequences.
            pse_embs: Tensor of shape [B, L_pse, D] containing input Paired Sentence Embeddings.
            pse_embs_lengths: Tensor of shape [B] containing lengths of the PSE.
            phoneme_representations: Tensor of shape [B, L_p, H] containing phoneme-level reprs.
            phoneme_lengths: Tensor of shape [B] containing lengths of the phoneme sequences.
        """

        token_context = self._token_embs_enc(
            inputs=token_embs,
            input_lengths=token_embs_lengths,
            phoneme_representations=phoneme_representations,
            phoneme_lengths=phoneme_lengths
        )

        pse_context = self._pse_enc(
            inputs=pse_embs,
            input_lengths=pse_embs_lengths,
            phoneme_representations=phoneme_representations,
            phoneme_lengths=phoneme_lengths
        )

        return token_context + pse_context