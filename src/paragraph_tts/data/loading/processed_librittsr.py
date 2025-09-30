"""Contains processed LibriTTS-R dataset loader."""

import torch
from typing import List, Dict
import random
import numpy as np
from typing import Optional

import lightning.pytorch as pl

from paragraph_tts.utils.path import processed_libri_dir_handler


class _DataSet(torch.utils.data.Dataset):
    """Loads serialized data from disk."""

    def __init__(self, samples: List[processed_libri_dir_handler.SampleInfo]):

        self._samples = samples

    def __len__(self):
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self._samples[idx]

        context = random.choice(sample.contexts)

        phoneme_ids = torch.load(sample.input_data.phoneme_ids_pth)
        spec = torch.load(sample.input_data.spec_pth)

        speaking_rate = torch.tensor(spec.shape[1] / phoneme_ids.shape[0], dtype=torch.float)

        context_token_emb = torch.load(context.token_embeddings_path)
        context_token_lengths = [embs.shape[0] for embs in context_token_emb]
        context_token_pse = torch.load(context.pse_path)

        return {
            'spk_emb': torch.load(sample.spk_embedding_path),
            'context_token_emb': torch.cat(context_token_emb, dim=0),
            'context_token_lengths': torch.tensor(context_token_lengths, dtype=torch.long),
            'context_pse': torch.stack(context_token_pse, dim=0),
            'input_token_emb': torch.load(sample.input_data.bert_embeddings_pth),
            'input_phoneme_ids': phoneme_ids,
            'input_spec': spec,
            'input_f0': torch.load(sample.input_data.f0_pth),
            'input_energy': torch.load(sample.input_data.energy_pth),
            'input_ling_stats': torch.load(sample.input_data.ling_stats_pth),
            'input_pos_tags': torch.load(sample.input_data.pos_tags_pth),
            'bert_to_word_pool_matrix': torch.load(sample.input_data.bert_to_word_pool_matrix_pth),
            'phone_to_spec_indices': torch.load(sample.input_data.phone_to_spec_indices_pth),
            'spec_to_word_pool_matrix': torch.load(sample.input_data.spec_to_word_pool_matrix_pth),
            'word_to_phoneme_indices': torch.load(sample.input_data.word_to_phoneme_indices_pth),
            'sentence_pos': torch.tensor(context.utterance_pos.value, dtype=torch.long),
            'spk_rate': speaking_rate
        }

    @staticmethod
    def collate_fn(batch_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Composes a padded batch from list of samples."""

        batch = {}

        batch['spk_emb'] = torch.stack([b['spk_emb'] for b in batch_samples], dim=0)

        batch['input_spec'] = torch.nn.utils.rnn.pad_sequence(
            [b['input_spec'].T for b in batch_samples], batch_first=True, padding_value=0.0
        ).transpose(1, 2)

        for key in ['bert_to_word_pool_matrix', 'spec_to_word_pool_matrix']:

            pad_dim_0 = max(b[key].shape[0] for b in batch_samples)
            pad_dim_1 = max(b[key].shape[1] for b in batch_samples)

            padded_matrices = []

            for b in batch_samples:
                padded = torch.nn.functional.pad(b[key],
                                                 (0, pad_dim_1 - b[key].shape[1],
                                                  0, pad_dim_0 - b[key].shape[0]))
                padded_matrices.append(padded)

            batch[key] = torch.stack(padded_matrices, dim=0)

        for key in ['context_token_emb', 'context_pse', 'input_token_emb', 'input_f0',
                    'input_energy', 'input_ling_stats']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0.0)

        for key in ['context_token_lengths', 'input_phoneme_ids', 'input_pos_tags',
                    'phone_to_spec_indices', 'word_to_phoneme_indices']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0)

        return batch


class ProcessedLibriTTSR(pl.LightningDataModule):
    """Loads processed LibriTTS-R dataset."""

    def __init__(self,
                 ds_path: str,
                 batch_size: int,
                 num_workers: int,
                 num_test_samples: int,
                 train_val_split: float):
        """
        Args:
            ds_path: Path to processed dataset.
            batch_size: Batch size.
            num_workers: Number of workers for data loading.
        """

        super().__init__()

        self._ds_path_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(ds_path)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_test_samples = num_test_samples
        self._train_val_split = train_val_split

        self._train_set: Optional[torch.utils.data.Dataset] = None
        self._val_set: Optional[torch.utils.data.Dataset] = None
        self._test_set: Optional[torch.utils.data.Dataset] = None

    def setup(self, _: str):

        rng = np.random.RandomState(2137)  # pylint: disable=no-member

        all_samples = list(self._ds_path_handler.iter_samples())
        test_samples = rng.choice(all_samples, size=100, replace=False)  # type: ignore
        train_val_samples = [s for s in all_samples if s not in test_samples]

        self._test_set = _DataSet(list(test_samples))

        generator = torch.Generator().manual_seed(2137)
        percentages = [self._train_val_split, 1 - self._train_val_split]
        self._train_set, self._val_set = torch.utils.data.random_split(_DataSet(train_val_samples),
                                                                       percentages,
                                                                       generator=generator)

    def train_dataloader(self):
        assert self._train_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._train_set,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=_DataSet.collate_fn)

    def val_dataloader(self):
        assert self._val_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._val_set,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=_DataSet.collate_fn)

    def test_dataloader(self):
        assert self._test_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._test_set,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=_DataSet.collate_fn)
