"""Contains definition of acoustic model's encoder module."""



import torch


class Encoder(torch.nn.Module):
    """Encodes input phonemes, BERT embeddings and linguistic stats."""

    def __init__(self,
                 phoneme_vocab_size: int,
                 pos_tags_vocab_size: int,
                 ling_stats_dim: int,
                 hidden_size: int,
                 dropout_rate: float):
        
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
            in_features=ling_stats_dim + pos_tags_vocab_size // 2,
            out_features=hidden_size
        )


