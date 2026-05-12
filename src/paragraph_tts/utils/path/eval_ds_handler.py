"""Contains utilities for handling access to evaluation dataset."""

import pathlib
import json
from typing import Iterator

import pydantic

from paragraph_tts.utils.path import raw_libri_dir_handler


class EvalUtterance(pydantic.BaseModel):
    """Information about an evaluation utterance."""

    eval_paragraph: 'EvalParagraph'
    raw_utterance: raw_libri_dir_handler.UtteranceInfo | None
    tensors_paths: dict[str, pathlib.Path]


class EvalParagraph(pydantic.BaseModel):
    """Information about an evaluation paragraph."""

    raw_paragraph: raw_libri_dir_handler.ParagraphInfo
    context_features_paths: dict[str, pathlib.Path]
    graph_features_paths: dict[str, pathlib.Path]
    utterances: list[EvalUtterance]


class EvalDsHandler:
    """Handles access to evaluation dataset."""

    def __init__(self,
                 eval_ds_path: pathlib.Path) -> None:

        self._eval_ds_path = eval_ds_path

        self._speakers_path = eval_ds_path / 'speakers'
        self._partial_paragraphs_path = eval_ds_path / 'partial_paragraphs'
        self._whole_paragraphs_path = eval_ds_path / 'whole_paragraphs'

    def iter_partial_paragraphs(self) -> Iterator[EvalParagraph]:
        """Iterates over all partial paragraphs in the evaluation dataset."""

        for spk_path in self._partial_paragraphs_path.iterdir():
            for para_path in spk_path.iterdir():
                yield self._load_paragraph(para_path)

    def iter_whole_paragraphs(self) -> Iterator[EvalParagraph]:
        """Iterates over all whole paragraphs in the evaluation dataset."""

        for spk_path in self._whole_paragraphs_path.iterdir():
            for para_path in spk_path.iterdir():
                yield self._load_paragraph(para_path)

    def create_whole_paragraph(self,
                               paragraph_info: raw_libri_dir_handler.ParagraphInfo
                               ) -> EvalParagraph:
        """Creates an EvalParagraph instance for a given paragraph."""

        paragraph_path = (self._whole_paragraphs_path
                          .joinpath(str(paragraph_info.spk_id))
                          .joinpath(f'{paragraph_info.chap_id}_{paragraph_info.para_id}'))
        paragraph_path.mkdir(parents=True, exist_ok=True)

        with paragraph_path.joinpath('raw_paragraph.json').open('w') as f:
            json.dump(paragraph_info.model_dump(), f)

        paragraph_path.joinpath('utt').joinpath('0').mkdir(parents=True, exist_ok=True)

        return self._load_paragraph(paragraph_path)

    def create_partial_paragraph(self,
                                 paragraph_info: raw_libri_dir_handler.ParagraphInfo
                                 ) -> EvalParagraph:
        """Creates an EvalParagraph instance for a given paragraph with partial data."""

        paragraph_path = (self._partial_paragraphs_path
                          .joinpath(str(paragraph_info.spk_id))
                          .joinpath(f'{paragraph_info.chap_id}_{paragraph_info.para_id}'))
        paragraph_path.mkdir(parents=True, exist_ok=True)

        with paragraph_path.joinpath('raw_paragraph.json').open('w') as f:
            json.dump(paragraph_info.model_dump(), f)

        for utt in paragraph_info.utterances:
            if utt.wav_path is not None:
                utt_path = paragraph_path.joinpath('utt').joinpath(str(utt.utt_id))
                utt_path.mkdir(parents=True, exist_ok=True)

                with utt_path.joinpath('raw_utterance.json').open('w') as f:
                    json.dump(utt.model_dump(), f)

        return self._load_paragraph(paragraph_path)

    def _load_paragraph(self, paragraph_path: pathlib.Path) -> EvalParagraph:
        """Loads an EvalParagraph instance from the given path."""

        with paragraph_path.joinpath('raw_paragraph.json').open('r') as f:
            raw_paragraph = raw_libri_dir_handler.ParagraphInfo.model_validate(json.load(f))

        eval_paragraph = EvalParagraph(
            raw_paragraph=raw_paragraph,
            context_features_paths={
                'context_token_emb': paragraph_path / 'context_token_emb.pt',
                'context_pse': paragraph_path / 'context_pse.pt'
            },
            graph_features_paths={
                'word_embeddings_path': paragraph_path / 'graph_word_emb.pt',
                'local_next_edge_idx_pth': paragraph_path / 'local_next_edge.pt',
                'local_prev_edge_idx_pth': paragraph_path / 'local_prev_edge.pt',
                'global_next_edge_idx_pth': paragraph_path / 'global_next_edge.pt',
                'global_prev_edge_idx_pth': paragraph_path / 'global_prev_edge.pt',
                'local_global_edge_idx_pth': paragraph_path / 'local_global_edge.pt'
            },
            utterances=[]
        )

        for utt_path in paragraph_path.joinpath('utt').iterdir():

            if utt_path.joinpath('raw_utterance.json').exists():
                with utt_path.joinpath('raw_utterance.json').open('r') as f:
                    raw_utt = raw_libri_dir_handler.UtteranceInfo.model_validate(json.load(f))
            else:
                raw_utt = None

            eval_paragraph.utterances.append(
                EvalUtterance(
                    raw_utterance=raw_utt,
                    tensors_paths={
                        'input_word_emb': utt_path.joinpath('input_word_emb.pt'),
                        'input_phoneme_ids': utt_path.joinpath('input_phoneme_ids.pt'),
                        'input_ling_stats': utt_path.joinpath('input_ling_stats.pt'),
                        'input_pos_tags': utt_path.joinpath('input_pos_tags.pt'),
                        'word_to_phoneme_indices': utt_path.joinpath('word_phone_indices.pt'),
                        'sentence_pos': utt_path.joinpath('sentence_pos.pt'),
                        'spk_rate': utt_path.joinpath('spk_rate.pt')
                    })
            )

        eval_paragraph.utterances.sort(
            key=lambda x: x.raw_utterance.utt_id if x.raw_utterance is not None else -1)

        return eval_paragraph
