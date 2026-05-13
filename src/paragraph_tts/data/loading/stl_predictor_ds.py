"""Contains utilities for loading the serialized STL predictor dataset."""

from typing import Annotated

import pydantic
from pydantic import Field
from lightning import pytorch as pl
import torch
import torch_geometric
from torch_geometric.data import HeteroData as GraphData
import torch_geometric.loader
import numpy as np

from paragraph_tts.utils.path import stl_predictor_ds_handler


class STLPredictorDataset(torch_geometric.data.Dataset):  # pylint: disable=abstract-method
    """Loads the serialized STL predictor dataset samples as graphs."""

    class Configuration(pydantic.BaseModel):
        """Configuration for loading the STL predictor dataset."""

        include_stl_series: Annotated[str, Field(
            description="The name of the STL series to include in the dataset.")]

    def __init__(self,
                 config: Configuration,
                 samples: list[stl_predictor_ds_handler.ProcessedParagraph]):

        super().__init__(root=None, transform=None, pre_transform=None, pre_filter=None)

        self._samples = samples
        self._config = config

    def get_sample_metadata(self, idx: int) -> stl_predictor_ds_handler.ProcessedParagraph:
        """Returns the metadata of the sample with the given index."""
        return self._samples[idx]

    def len(self) -> int:
        """Returns the number of samples in the dataset."""
        return len(self._samples)

    def get(self, idx: int) -> GraphData:
        """Returns the sample with the given index as a graph."""

        sample = self._samples[idx]
        graph = GraphData()

        word_embeddings_list = [
            torch.load(utt.word_embeddings_path) for utt in sample.utterances
        ]

        graph['word_emb'].x = torch.cat(word_embeddings_list, dim=0)
        graph['global_emb'].x = torch.stack([torch.mean(word_embeddings, dim=0)
                                             for word_embeddings in word_embeddings_list])

        graph['word_emb',
              'follows',
              'word_emb'].edge_index = torch.load(sample.local_next_edge_idx_pth)

        graph['word_emb',
              'precedes',
              'word_emb'].edge_index = torch.load(sample.local_prev_edge_idx_pth)

        graph['global_emb',
              'follows',
              'global_emb'].edge_index = torch.load(sample.global_next_edge_idx_pth)

        graph['global_emb',
              'precedes',
              'global_emb'].edge_index = torch.load(sample.global_prev_edge_idx_pth)

        graph['word_emb',
              'same_utt',
              'global_emb'].edge_index = torch.load(sample.local_global_edge_idx_pth)

        wsv_weights = []
        gst_weights = []
        has_gst_mask = []
        has_wsv_mask = []

        for utt, word_embeddings in zip(sample.utterances, word_embeddings_list):

            if utt.stl_weights is None:
                has_wsv_mask.append(torch.zeros((word_embeddings.size(0),), dtype=torch.bool))
                has_gst_mask.append(torch.tensor(0, dtype=torch.bool))

            else:
                has_wsv_mask.append(torch.ones((word_embeddings.size(0),), dtype=torch.bool))
                has_gst_mask.append(torch.tensor(1, dtype=torch.bool))

                weights = utt.stl_weights[self._config.include_stl_series]

                wsv_weights.append(torch.load(weights.wsv_path))
                gst_weights.append(torch.load(weights.gst_path))

        graph.wsv_weights = torch.cat(wsv_weights, dim=0)
        graph.has_wsv_mask = torch.cat(has_wsv_mask, dim=0)
        graph.gst_weights = torch.stack(gst_weights)
        graph.has_gst_mask = torch.stack(has_gst_mask)
        graph.sentence_lengths = torch.tensor(
            [word_embeddings.size(0) for word_embeddings in word_embeddings_list], dtype=torch.long)

        return graph


class STLPredictorDataModule(pl.LightningDataModule):
    """Lightning DataModule for the STL predictor dataset."""

    def __init__(self, processed_ds_handler: stl_predictor_ds_handler.STLPredictorDatasetHandler,
                 config: STLPredictorDataset.Configuration,
                 batch_size: int,
                 num_workers: int,
                 train_val_split: float,
                 seed: int):

        super().__init__()

        self._processed_ds_handler = processed_ds_handler
        self._config = config
        self._batch_size = batch_size
        self._num_workers = num_workers
        self._train_val_split = train_val_split
        self._seed = seed

        self._train_ds: STLPredictorDataset | None = None
        self._val_ds: STLPredictorDataset | None = None

    @property
    def val_ds(self) -> STLPredictorDataset:
        """Returns the validation dataset."""
        assert self._val_ds is not None, "Call setup() first."
        return self._val_ds

    def setup(self, stage: str | None = None) -> None:

        all_samples = list(self._processed_ds_handler.iter_paragraphs())

        np.random.RandomState(self._seed).shuffle(all_samples)  # pylint: disable=no-member

        train_size = int(self._train_val_split * len(all_samples))
        train_samples = all_samples[:train_size]
        val_samples = all_samples[train_size:]

        self._train_ds = STLPredictorDataset(config=self._config, samples=train_samples)
        self._val_ds = STLPredictorDataset(config=self._config, samples=val_samples)

    def train_dataloader(self) -> torch_geometric.loader.DataLoader:
        """Returns the training dataloader."""
        assert self._train_ds is not None, "Call setup() first."

        return torch_geometric.loader.DataLoader(self._train_ds,
                                                 batch_size=self._batch_size,
                                                 shuffle=True,
                                                 num_workers=self._num_workers,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 prefetch_factor=4)

    def val_dataloader(self) -> torch_geometric.loader.DataLoader:
        """Returns the validation dataloader."""
        assert self._val_ds is not None, "Call setup() first."

        return torch_geometric.loader.DataLoader(self._val_ds,
                                                 batch_size=self._batch_size,
                                                 shuffle=False,
                                                 num_workers=self._num_workers,
                                                 pin_memory=True,
                                                 persistent_workers=True,
                                                 prefetch_factor=4)
