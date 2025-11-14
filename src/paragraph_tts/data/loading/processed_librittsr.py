# -*- coding: utf-8 -*-
"""Contains processed LibriTTS-R dataset loader."""
import logging
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import lightning.pytorch as pl
import torch

from paragraph_tts.utils.path import processed_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


class _DataSet(torch.utils.data.Dataset):
    """Loads serialized data from disk."""

    def __init__(self,
                 processed_dir_handler: processed_libri_dir_handler.ProcessedLibriDirHandler,
                 samples: List[processed_libri_dir_handler.SampleInfo],
                 n_pitch_bins: int,
                 pitch_bounds: Tuple[float, float],
                 n_energy_bins: int,
                 energy_bounds: Tuple[float, float]):

        self._samples = samples
        self._processed_dir_handler = processed_dir_handler
        self._n_pitch_bins = n_pitch_bins
        self._min_pitch, self._max_pitch = pitch_bounds
        self._n_energy_bins = n_energy_bins
        self._min_energy, self._max_energy = energy_bounds

        self._f0_possible_values = torch.linspace(
            self._min_pitch, self._max_pitch, self._n_pitch_bins)
        self._energy_possible_values = torch.linspace(
            self._min_energy, self._max_energy, self._n_energy_bins)

    def __len__(self):
        return len(self._samples)

    def _scale(self,
               values: torch.Tensor,
               mean: torch.Tensor,
               std: torch.Tensor) -> torch.Tensor:

        return (values - mean) / std

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        sample = self._samples[idx]

        context = random.choice(sample.contexts)

        phoneme_ids = torch.load(sample.input_data.phoneme_ids_pth)
        phoneme_lengths = torch.tensor(phoneme_ids.shape[0], dtype=torch.long)
        spec = torch.load(sample.input_data.spec_pth)
        spec_length = torch.tensor(spec.shape[1], dtype=torch.long)
        input_token_emb = torch.load(sample.input_data.bert_embeddings_pth).to(torch.float)
        bert_to_word_pool_matrix = torch.load(sample.input_data.bert_to_word_pool_matrix_pth)

        input_token_emb = torch.matmul(bert_to_word_pool_matrix.T, input_token_emb)

        speaking_rate = torch.tensor(spec.shape[1] / phoneme_ids.shape[0], dtype=torch.float)

        context_token_emb_list = torch.load(context.token_embeddings_path)
        context_token_emb = torch.cat(context_token_emb_list, dim=0).to(torch.float)
        context_tokens_length = torch.tensor(context_token_emb.shape[0], dtype=torch.long)

        context_token_pse_list = torch.load(context.pse_path)
        context_token_pse = torch.stack(context_token_pse_list, dim=0).to(torch.float)
        context_pse_length = torch.tensor(context_token_pse.shape[0], dtype=torch.long)

        num_stats = self._processed_dir_handler.get_numerical_stats(sample.spk_id)

        f0_stats = torch.load(num_stats.f0_stats_pth)
        mean_f0, std_f0 = f0_stats['mean'], f0_stats['std']
        energy_stats = torch.load(num_stats.energy_stats_pth)
        mean_energy, std_energy = energy_stats['mean'], energy_stats['std']

        f0 = self._scale(
            torch.load(sample.input_data.f0_pth),
            mean_f0, std_f0
        )

        energy = self._scale(
            torch.load(sample.input_data.energy_pth),
            mean_energy, std_energy
        )

        word_to_phoneme_indices = torch.load(sample.input_data.word_to_phoneme_indices_pth)

        pos_tags = torch.load(sample.input_data.pos_tags_pth)
        pos_tags = pos_tags[word_to_phoneme_indices]

        return {
            'spk_emb': torch.load(sample.spk_embedding_path),
            'context_token_emb': context_token_emb,
            'context_tokens_length': context_tokens_length,
            'context_pse': context_token_pse,
            'context_pse_length': context_pse_length,
            'input_token_emb': input_token_emb,
            'input_phoneme_ids': phoneme_ids,
            'input_phonemes_length': phoneme_lengths,
            'input_spec': spec,
            'input_spec_length': spec_length,
            'input_f0': f0,
            'input_energy': energy,
            'input_ling_stats': torch.load(sample.input_data.ling_stats_pth),
            'input_pos_tags': pos_tags,
            'phone_to_spec_indices': torch.load(sample.input_data.phone_to_spec_indices_pth),
            'spec_to_word_pool_matrix': torch.load(sample.input_data.spec_to_word_pool_matrix_pth),
            'word_to_phoneme_indices': word_to_phoneme_indices,
            'sentence_pos': torch.tensor(context.utterance_pos.value, dtype=torch.long),
            'spk_rate': speaking_rate,
            'explicit_durations': torch.load(sample.input_data.durations_pth),
        }

    def collate_fn(self, batch_samples: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """Composes a padded batch from list of samples."""

        batch = {}

        for key in ['spk_emb', 'spk_rate', 'input_phonemes_length', 'input_spec_length',
                    'context_pse_length', 'context_tokens_length', 'sentence_pos']:

            batch[key] = torch.stack([b[key] for b in batch_samples], dim=0)

        batch['input_spec'] = torch.nn.utils.rnn.pad_sequence(
            [b['input_spec'].T for b in batch_samples], batch_first=True, padding_value=0.0
        ).transpose(1, 2)

        for key in ['spec_to_word_pool_matrix']:

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
                    'input_energy', 'input_ling_stats', 'explicit_durations']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0.0)

        for key in ['input_phoneme_ids', 'input_pos_tags',
                    'phone_to_spec_indices', 'word_to_phoneme_indices']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0)

        batch['pitch_possible_values'] = self._f0_possible_values
        batch['energy_possible_values'] = self._energy_possible_values

        return batch


class ProcessedLibriTTSR(pl.LightningDataModule):
    """Loads processed LibriTTS-R dataset."""

    def __init__(self,
                 ds_path: str,
                 batch_size: int,
                 num_workers: int,
                 num_test_samples: int,
                 train_val_split: float,
                 n_pitch_bins: int,
                 pitch_bounds: Tuple[float, float],
                 n_energy_bins: int,
                 energy_bounds: Tuple[float, float]):
        """
        Args:
            ds_path: Path to processed dataset.
            batch_size: Batch size.
            num_workers: Number of workers for data loading.
            num_test_samples: Number of samples in test set.
            train_val_split: Percentage of training samples in train+val split.
            n_pitch_bins: Number of bins for pitch quantization.
            pitch_bounds: Min and max values for pitch quantization.
            n_energy_bins: Number of bins for energy quantization.
            energy_bounds: Min and max values for energy quantization.
        """

        super().__init__()

        self._ds_path_handler = processed_libri_dir_handler.ProcessedLibriDirHandler(ds_path)

        self._batch_size = batch_size
        self._num_workers = num_workers
        self._num_test_samples = num_test_samples
        self._train_val_split = train_val_split

        self._train_set: Optional[_DataSet] = None
        self._val_set: Optional[_DataSet] = None
        self._test_set: Optional[_DataSet] = None

        self._n_pitch_bins = n_pitch_bins
        self._pitch_bounds = pitch_bounds
        self._n_energy_bins = n_energy_bins
        self._energy_bounds = energy_bounds

    def setup(self, stage: str):

        _logger().debug('Setting up dataset...')

        all_samples = list(self._ds_path_handler.iter_samples())
        random.shuffle(all_samples)

        test_samples = all_samples[:self._num_test_samples]
        train_val_samples = all_samples[self._num_test_samples:]

        _logger().debug('Creating test set with %d samples.', len(test_samples))

        self._test_set = _DataSet(self._ds_path_handler,
                                  test_samples,
                                  self._n_pitch_bins, self._pitch_bounds,
                                  self._n_energy_bins, self._energy_bounds)

        n_train_samples = int(len(train_val_samples) * self._train_val_split)

        train_samples = train_val_samples[:n_train_samples]
        val_samples = train_val_samples[n_train_samples:]

        _logger().debug('Creating train set with %d samples.', len(train_samples))
        _logger().debug('Creating validation set with %d samples.', len(val_samples))

        self._train_set = _DataSet(self._ds_path_handler,
                                   train_samples,
                                   self._n_pitch_bins, self._pitch_bounds,
                                   self._n_energy_bins, self._energy_bounds)

        self._val_set = _DataSet(self._ds_path_handler,
                                 val_samples,
                                 self._n_pitch_bins, self._pitch_bounds,
                                 self._n_energy_bins, self._energy_bounds)

    def train_dataloader(self):
        assert self._train_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._train_set,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=self._train_set.collate_fn)

    def val_dataloader(self):
        assert self._val_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._val_set,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=self._val_set.collate_fn)

    def test_dataloader(self):
        assert self._test_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._test_set,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=self._test_set.collate_fn)
