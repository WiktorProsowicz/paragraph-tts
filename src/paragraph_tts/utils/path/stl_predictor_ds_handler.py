"""Contains the STL predictor dataset handler."""

import pathlib
from typing import Iterator

import pydantic

from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.utils.path import raw_libri_dir_handler


class SerializedSTLWeights(pydantic.BaseModel):
    """Represents the serialized STL weights for an utterance."""

    wsv_path: pathlib.Path
    gst_path: pathlib.Path


class ProcessedUtterance(pydantic.BaseModel):
    """Represents an utterance in the STL predictor dataset."""

    raw_utterance: raw_libri_dir_handler.UtteranceInfo

    text_features_pth: pathlib.Path
    word_embeddings_path: pathlib.Path

    stl_weights: dict[str, SerializedSTLWeights] | None


class ProcessedParagraph(pydantic.BaseModel):
    """Represents a paragraph in the STL predictor dataset."""

    raw_paragraph: raw_libri_dir_handler.ParagraphInfo
    utterances: list[ProcessedUtterance]

    local_prev_edge_idx_pth: pathlib.Path
    local_next_edge_idx_pth: pathlib.Path

    global_prev_edge_idx_pth: pathlib.Path
    global_next_edge_idx_pth: pathlib.Path

    local_global_edge_idx_pth: pathlib.Path


class STLPredictorDatasetHandler:
    """Manages the access to the STL predictor dataset directory structure."""

    def __init__(self,
                 ds_path: pathlib.Path,
                 stl_series: list[str] | None = None):
        """Initializes the dataset handler.

        Args:
            ds_path: The path to the dataset directory.
            stl_series: The list of STL weights groups to include in the dataset. If None, the
                dataset will be able only to read the already serialized groups..
        """

        self._ds_path = ds_path
        self._paragraphs_path = ds_path / 'paragraphs'
        self._stl_series = stl_series

    def create_paragraph(self,
                         processed_paragraph: processed_libri_dir_handler.ProcessedParagraph
                         ) -> ProcessedParagraph:
        """Creates a new paragraph in the dataset based on the processed acoustic paragraph."""

        para_path = (self._paragraphs_path
                     .joinpath(str(processed_paragraph.speaker_info.spk_id))
                     .joinpath('{}_{}'.format(processed_paragraph.raw_paragraph.chap_id,  # pylint: disable=all
                                              processed_paragraph.raw_paragraph.para_id)))
        para_path.mkdir(parents=True, exist_ok=True)

        with open(para_path / 'raw_paragraph.json', 'w', encoding='utf-8') as f:
            f.write(processed_paragraph.raw_paragraph.model_dump_json(indent=4, ensure_ascii=False))

        utterances_dir = para_path / 'utterances'

        for utterance in processed_paragraph.raw_paragraph.utterances:

            utterance_path = utterances_dir / str(utterance.utt_id)
            utterance_path.mkdir(parents=True, exist_ok=True)

            with open(utterance_path / 'raw_utterance.json', 'w', encoding='utf-8') as f:
                f.write(utterance.model_dump_json(indent=4, ensure_ascii=False))

            if any(utt.raw_utterance.utt_id == utterance.utt_id
                   for utt in processed_paragraph.utterances):

                for stl_series in self._stl_series or []:

                    series_path = utterance_path / 'stl_series' / stl_series
                    series_path.mkdir(parents=True, exist_ok=True)

        return self._load_paragraph(para_path)

    def iter_paragraphs(self) -> Iterator[ProcessedParagraph]:
        """Iterates over the paragraphs in the dataset."""

        for speaker_dir in self._paragraphs_path.iterdir():

            for para_dir in speaker_dir.iterdir():

                if not para_dir.is_dir():
                    continue

                yield self._load_paragraph(para_dir)

    def _load_paragraph(self, paragraph_path: pathlib.Path) -> ProcessedParagraph:
        """Loads a paragraph from the dataset based on the paragraph path."""

        with open(paragraph_path / 'raw_paragraph.json', 'r', encoding='utf-8') as f:
            raw_paragraph = raw_libri_dir_handler.ParagraphInfo.model_validate_json(f.read())

        utterances_dir = paragraph_path / 'utterances'

        utterances = []

        for utterance_path in utterances_dir.iterdir():

            with open(utterance_path / 'raw_utterance.json', 'r', encoding='utf-8') as f:
                raw_utterance = raw_libri_dir_handler.UtteranceInfo.model_validate_json(f.read())

            stl_weights = None

            if utterance_path.joinpath('stl_series').exists():

                stl_weights = {}

                for stl_series_dir in utterance_path.joinpath('stl_series').iterdir():

                    stl_weights[stl_series_dir.name] = SerializedSTLWeights(
                        wsv_path=stl_series_dir / 'wsv.pt',
                        gst_path=stl_series_dir / 'gst.pt'
                    )

            utterances.append(ProcessedUtterance(
                raw_utterance=raw_utterance,
                text_features_pth=utterance_path / 'text_features.pkl',
                word_embeddings_path=utterance_path / 'word_embeddings.pt',
                stl_weights=stl_weights
            ))

        utterances.sort(key=lambda u: u.raw_utterance.utt_id)

        return ProcessedParagraph(
            raw_paragraph=raw_paragraph,
            utterances=utterances,
            local_prev_edge_idx_pth=paragraph_path / 'local_prev_edge_idx.pt',
            local_next_edge_idx_pth=paragraph_path / 'local_next_edge_idx.pt',
            global_prev_edge_idx_pth=paragraph_path / 'global_prev_edge_idx.pt',
            global_next_edge_idx_pth=paragraph_path / 'global_next_edge_idx.pt',
            local_global_edge_idx_pth=paragraph_path / 'local_global_edge_idx.pt'
        )
