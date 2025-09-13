"""Contains classes for processing/reading LibriTTS-R dataset."""

import logging
import tqdm
import os

import torch.utils.data as torch_data
from comp_trans_tts import (deepspeaker, audio as ctt_audio) 
import numpy as np

from paragraph_tts import utils


class LibriTTSR(torch_data.Dataset):
    pass


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
        self._stft = ctt_audio.stft.TacotronSTFT(
            filter_length=1024,
            hop_length=256,
            win_length=1024,
            n_mel_channels=80,
            sampling_rate=22050,
            mel_fmin=0,
            mel_fmax=8000
        )

    def run(self):
        """Runs preprocessing."""

        if self._multi_speaker:
            logging.info('Preparing speaker embeddings...')
            self._prepare_spk_embeddings()

        

    def _prepare_spk_embeddings(self):

        embeddings_path = os.path.join(self._output_path, 'spk_embeddings')

        if os.path.exists(embeddings_path):
            logging.info('Speaker embeddings already exist, skipping preparation.')
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

            np.save(os.path.join(embeddings_path, f'{spk_id}.npy'),
                    final_embedding)
