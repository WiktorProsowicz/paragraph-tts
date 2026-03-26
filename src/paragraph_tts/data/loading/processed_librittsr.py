"""Contains processed LibriTTS-R dataset loader."""
import logging
import random
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Annotated

import lightning.pytorch as pl
import torch
import pydantic
from pydantic import Field

from paragraph_tts.utils.path import processed_libri_dir_handler


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ProcessedLibriTTSRDataset(torch.utils.data.Dataset[dict[str, torch.Tensor]]):
    """Loads serialized data from disk."""

    class Configuration(pydantic.BaseModel):
        """Configuration for ProcessedLibriTTSRDataset."""

        load_prosody_features: Annotated[bool, Field(
            description='Whether to load pitch and energy features.')]

        pitch_quantization_params: Annotated[tuple[int, float, float] | None, Field(
            description='Number of bins, min and max values for pitch quantization.')]

        energy_quantization_params: Annotated[tuple[int, float, float] | None, Field(
            description='Number of bins, min and max values for energy quantization.')]

        scale_prosody_features: Annotated[bool, Field(
            description='Whether to scale pitch and energy features using speaker-wise statistics.')
        ]

        use_phoneme_level_prosody_features: Annotated[bool, Field(
            description='Whether to average frame-level pitch and energy features to phoneme-level.'
        )]

        load_context_embeddings: Annotated[bool, Field(
            description='Whether to load context token embeddings and PSE features.')]

    def __init__(self,
                 cfg: Configuration,
                 utterances: list[processed_libri_dir_handler.ProcessedUtterance]):

        self._utterances = utterances

        if cfg.pitch_quantization_params is not None:
            n_pitch_bins, min_pitch, max_pitch = cfg.pitch_quantization_params
            self._f0_possible_values = torch.linspace(min_pitch, max_pitch, n_pitch_bins)

        else:
            self._f0_possible_values = None

        if cfg.energy_quantization_params is not None:
            n_energy_bins, min_energy, max_energy = cfg.energy_quantization_params
            self._energy_possible_values = torch.linspace(min_energy, max_energy, n_energy_bins)
        else:
            self._energy_possible_values = None

        self._cfg = cfg

    def __len__(self) -> int:
        return len(self._utterances)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

        utterance = self._utterances[idx]

        phoneme_ids = torch.load(utterance.phoneme_ids_pth)
        phoneme_lengths = torch.tensor(phoneme_ids.shape[0], dtype=torch.long)
        word_to_phoneme_indices = torch.load(utterance.word_to_phoneme_indices_pth)
        pos_tags = torch.load(utterance.pos_tags_pth)
        pos_tags = pos_tags[word_to_phoneme_indices]

        spec = torch.load(utterance.spec_pth)
        spec_length = torch.tensor(spec.shape[1], dtype=torch.long)

        input_token_emb = torch.load(utterance.bert_embeddings_pth).to(torch.float)
        bert_to_word_pool_matrix = torch.load(utterance.bert_to_word_pool_matrix_pth)
        input_word_embeddings = torch.matmul(bert_to_word_pool_matrix.T, input_token_emb)

        data: dict[str, torch.Tensor] = {}

        if self._cfg.load_prosody_features:
            data.update(self._load_prosody_features(utterance))

        if self._cfg.load_context_embeddings:
            data.update(self._load_context_features(utterance))

        return {
            'spk_emb': torch.load(utterance.paragraph.speaker_info.embedding_path),
            'input_word_emb': input_word_embeddings,
            'input_phoneme_ids': phoneme_ids,
            'input_phonemes_length': phoneme_lengths,
            'input_spec': spec,
            'input_spec_length': spec_length,
            'input_ling_stats': torch.load(utterance.ling_stats_pth),
            'input_pos_tags': pos_tags,
            'phone_to_spec_indices': torch.load(utterance.phone_to_spec_indices_pth),
            'spec_to_word_pool_matrix': torch.load(utterance.spec_to_word_pool_matrix_pth),
            'word_to_phoneme_indices': word_to_phoneme_indices,
            'sentence_pos': torch.tensor(utterance.utterance_pos.value, dtype=torch.long),
            'spk_rate': torch.tensor(spec.shape[1] / phoneme_ids.shape[0], dtype=torch.float),
            'explicit_durations': torch.load(utterance.durations_pth),
            **data
        }

    def _load_prosody_features(self,
                               sample: processed_libri_dir_handler.ProcessedUtterance
                               ) -> dict[str, torch.Tensor]:

        f0 = torch.load(sample.f0_pth)
        energy = torch.load(sample.energy_pth)

        if self._cfg.scale_prosody_features:

            f0_stats = torch.load(sample.paragraph.speaker_info.f0_stats_path)
            mean_f0, std_f0 = f0_stats['mean'], f0_stats['std']

            energy_stats = torch.load(sample.paragraph.speaker_info.energy_stats_path)
            mean_energy, std_energy = energy_stats['mean'], energy_stats['std']

            f0 = (f0 - mean_f0) / std_f0
            energy = (energy - mean_energy) / std_energy

        if self._cfg.use_phoneme_level_prosody_features:

            spec_to_phone_pool_matrix = torch.load(sample.spec_to_phone_pool_matrix_pth.T)

            f0 = torch.matmul(spec_to_phone_pool_matrix, f0)
            energy = torch.matmul(spec_to_phone_pool_matrix, energy)

        return {
            'input_f0': f0,
            'input_energy': energy,
        }

    def _load_context_features(self,
                               sample: processed_libri_dir_handler.ProcessedUtterance
                               ) -> dict[str, torch.Tensor]:

        context_token_emb_list = torch.load(sample.paragraph.token_embeddings_path)

        if context_token_emb_list:
            context_token_emb = torch.cat(context_token_emb_list, dim=0).to(torch.float)
            context_tokens_length = torch.tensor(context_token_emb.shape[0], dtype=torch.long)
        else:
            context_token_emb = torch.empty(0, dtype=torch.float)
            context_tokens_length = torch.tensor(0, dtype=torch.long)

        context_token_pse_list = torch.load(sample.paragraph.pse_path)

        if context_token_pse_list:
            context_token_pse = torch.stack(context_token_pse_list, dim=0).to(torch.float)
            context_pse_length = torch.tensor(context_token_pse.shape[0], dtype=torch.long)
        else:
            context_token_pse = torch.empty(0,  dtype=torch.float)
            context_pse_length = torch.tensor(0, dtype=torch.long)

        return {
            'context_token_emb': context_token_emb,
            'context_tokens_length': context_tokens_length,
            'context_pse': context_token_pse,
            'context_pse_length': context_pse_length,
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

        for key in ['context_token_emb', 'context_pse', 'input_word_emb', 'input_f0',
                    'input_energy', 'input_ling_stats', 'explicit_durations']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0.0)

        for key in ['input_phoneme_ids', 'input_pos_tags',
                    'phone_to_spec_indices', 'word_to_phoneme_indices']:

            batch[key] = torch.nn.utils.rnn.pad_sequence(
                [b[key] for b in batch_samples], batch_first=True, padding_value=0)

        if self._f0_possible_values is not None:
            batch['pitch_possible_values'] = self._f0_possible_values

        if self._energy_possible_values is not None:
            batch['energy_possible_values'] = self._energy_possible_values

        return batch


class ProcessedLibriTTSRDataModule(pl.LightningDataModule):
    """Loads processed LibriTTS-R dataset."""

    def __init__(self,
                 ds_cfg: ProcessedLibriTTSRDataset.Configuration,
                 processed_ds_handler: processed_libri_dir_handler.ProcessedLibriDirHandler,
                 batch_size: int,
                 num_workers: int,
                 train_val_split: float,

                 ):

        super().__init__()

        self._processed_ds_handler = processed_ds_handler
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_val_split = train_val_split
        self._ds_cfg = ds_cfg

        self._train_set: ProcessedLibriTTSRDataset | None = None
        self._val_set: ProcessedLibriTTSRDataset | None = None

    def setup(self, stage: str) -> None:

        _logger().debug('Setting up dataset...')

        all_utterances = list(self._processed_ds_handler.iter_utterances())
        random.shuffle(all_utterances)

        n_train_samples = int(len(all_utterances) * self._train_val_split)

        train_samples = all_utterances[:n_train_samples]
        val_samples = all_utterances[n_train_samples:]

        _logger().debug('Creating train set with %d samples.', len(train_samples))
        _logger().debug('Creating validation set with %d samples.', len(val_samples))

        self._train_set = ProcessedLibriTTSRDataset(self._ds_cfg, train_samples)
        self._val_set = ProcessedLibriTTSRDataset(self._ds_cfg, val_samples)

    def train_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, torch.Tensor]]:
        assert self._train_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._train_set,
                                           batch_size=self._batch_size,
                                           shuffle=True,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=self._train_set.collate_fn)

    def val_dataloader(self) -> torch.utils.data.DataLoader[Dict[str, torch.Tensor]]:
        assert self._val_set is not None, 'Make sure to call setup() before using this method!'

        return torch.utils.data.DataLoader(self._val_set,
                                           batch_size=self._batch_size,
                                           shuffle=False,
                                           num_workers=self._num_workers,
                                           pin_memory=True,
                                           collate_fn=self._val_set.collate_fn)
