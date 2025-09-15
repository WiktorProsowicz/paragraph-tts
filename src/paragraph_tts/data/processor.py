"""Contains classes for processing/reading LibriTTS-R dataset."""

import logging
import tqdm  # type: ignore
import os

import torch
import numpy as np
from comp_trans_tts import (deepspeaker)  # type: ignore

from paragraph_tts import (utils, data)



class LibriTTSRPreprocessor:
    """Runs preprocessing on raw dataset."""

    def __init__(self,
                 raw_path_handler: utils.path.RawLibriDirHandler,
                 output_path: str,
                 multi_speaker: bool):
        """
        Args:
            raw_path_handler: Handler for accessing raw dataset files.
        """

        self._raw_path_handler = raw_path_handler
        self._output_path = output_path
        self._multi_speaker = multi_speaker
        self._embedder = deepspeaker.embedder.DeepSpeakerEmbedder()
        self._audio_processor = data.preprocessing.audio.AudioProcessor(
            sr=22050,
            hop_length=256,
            win_length=1025,
            n_mels=80,
            fmin=0,
            fmax=8000,
            trim_top_db=23
        )

    def run(self):
        """Runs preprocessing."""

        if self._multi_speaker:
            logging.info('Preparing speaker embeddings...')
            self._prepare_spk_embeddings()

    def _prepare_spk_embeddings(self):

        embeddings_path = os.path.join(self._output_path, 'spk_embeddings')

        if os.path.exists(embeddings_path):
            logging.info(
                'Speaker embeddings already exist, skipping preparation.')
            return

        os.makedirs(embeddings_path, exist_ok=True)

        for spk_id in tqdm.tqdm(self._raw_path_handler.iter_speakers(),
                                desc='Speakers',
                                total=self._raw_path_handler.num_speakers):

            embeddings_for_spk = []

            for utterance_info in self._raw_path_handler.iter_utterances_for_spk(spk_id):
                embedder_input = deepspeaker.preprocess.load_wav_for_deepseaker(
                    utterance_info.wav_path)
                embedding = self._embedder(embedder_input)[0]
                embeddings_for_spk.append(embedding)

            final_embedding = np.mean(embeddings_for_spk, axis=0)

            torch.save(torch.tensor(final_embedding),
                       os.path.join(embeddings_path, f'{spk_id}.pt'))
