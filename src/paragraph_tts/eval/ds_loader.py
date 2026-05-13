"""Contains utilities for loading datasets for evaluation."""

import dataclasses

import torch
from torch_geometric.data import HeteroData
import numpy as np

from paragraph_tts.data.preprocessing import audio as audio_prep
from paragraph_tts.utils.path import eval_ds_handler
from paragraph_tts.utils.path import raw_libri_dir_handler


@dataclasses.dataclass
class EvalInputData:
    """Data for a single evaluation input."""

    context_tensors: dict[str, torch.Tensor]
    graph_data: HeteroData
    input_acoustic_data: dict[str, torch.Tensor]
    raw_paragraph: raw_libri_dir_handler.ParagraphInfo
    gt_wav: np.ndarray


def construct_graph_for_paragraph(paragraph: eval_ds_handler.EvalParagraph) -> HeteroData:
    """Constructs a graph for a given evaluation paragraph."""

    graph = HeteroData()
    features_paths = paragraph.graph_features_paths

    word_embeddings_list = torch.load(features_paths['word_embeddings_pth'])

    graph['word_emb'].x = torch.cat(word_embeddings_list, dim=0)
    graph['global_emb'].x = torch.stack([torch.mean(word_embeddings, dim=0)
                                         for word_embeddings in word_embeddings_list])

    graph['word_emb',
          'follows',
          'word_emb'].edge_index = torch.load(features_paths['local_next_edge_idx_pth'])

    graph['word_emb',
          'precedes',
          'word_emb'].edge_index = torch.load(features_paths['local_prev_edge_idx_pth'])

    graph['global_emb',
          'follows',
          'global_emb'].edge_index = torch.load(features_paths['global_next_edge_idx_pth'])

    graph['global_emb',
          'precedes',
          'global_emb'].edge_index = torch.load(features_paths['global_prev_edge_idx_pth'])

    graph['word_emb',
          'same_utt',
          'global_emb'].edge_index = torch.load(features_paths['local_global_edge_idx_pth'])

    graph.sentence_lengths = torch.tensor([len(word_embeddings)
                                           for word_embeddings in word_embeddings_list])

    return graph


class WholeParagraphsDS(torch.utils.data.Dataset):
    """Dataset for loading model data for whole paragraphs."""

    def __init__(self, paragraphs: list[eval_ds_handler.EvalParagraph]) -> None:

        self._paragraphs = paragraphs

    def __len__(self) -> int:
        return len(self._paragraphs)

    def __getitem__(self, idx: int) -> EvalInputData:

        paragraph = self._paragraphs[idx]

        return EvalInputData(
            context_tensors={name: torch.load(path)
                             for name, path in paragraph.context_features_paths.items()},
            graph_data=construct_graph_for_paragraph(paragraph),
            input_acoustic_data={name: torch.load(path)
                                 for name, path in paragraph.utterances[0].tensors_paths.items()},
            raw_paragraph=paragraph.raw_paragraph,
            gt_wav=audio_prep.load_wav_raw(paragraph.utterances[0].gt_wav_path)
        )


class PartialParagraphsDS(torch.utils.data.Dataset):
    """Dataset for loading model data for partial paragraphs."""

    def __init__(self, utterances: list[eval_ds_handler.EvalUtterance]) -> None:

        self._utterances = utterances

    def __len__(self) -> int:
        return len(self._utterances)

    def __getitem__(self, idx: int) -> EvalInputData:

        utterance = self._utterances[idx]
        paragraph = utterance.eval_paragraph

        acoustic_data = {name: torch.load(path) for name, path in utterance.tensors_paths.items()}

        acoustic_data['sentence_pos'] = torch.repeat_interleave(
            acoustic_data['sentence_pos'],
            torch.tensor(acoustic_data['input_phoneme_ids'].size(0))
        )

        acoustic_data['spk_rate'] = torch.repeat_interleave(
            acoustic_data['spk_rate'],
            torch.tensor(acoustic_data['input_phoneme_ids'].size(0))
        )

        return EvalInputData(
            context_tensors={name: torch.load(path)
                             for name, path in paragraph.context_features_paths.items()},
            graph_data=construct_graph_for_paragraph(paragraph),
            input_acoustic_data=acoustic_data,
            raw_paragraph=paragraph.raw_paragraph,
            gt_wav=audio_prep.load_wav_raw(utterance.gt_wav_path)
        )
