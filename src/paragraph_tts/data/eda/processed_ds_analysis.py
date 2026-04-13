"""Utilities for performing analysis of the processed dataset."""

import logging
import pathlib
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.manifold import TSNE
from torch_dev_utils.tts import visualization as viz_utils


from paragraph_tts.data import librittsr_helpers
from paragraph_tts.data.eda import utils as eda_utils
from paragraph_tts.utils.path import processed_libri_dir_handler
from paragraph_tts.data.loading import processed_librittsr as ds_loading
from paragraph_tts.data.preprocessing import processor as ds_processor


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


class ProcessedDSAnalyzer:
    """Summarizes the processed dataset and composes statistics about it."""

    def __init__(self,
                 ds_handler: processed_libri_dir_handler.ProcessedLibriDirHandler,
                 processor_cfg: ds_processor.LibriTTSRProcessor.Configuration) -> None:

        self._ds_handler = ds_handler
        self._dataset = ds_loading.ProcessedLibriTTSRDataset(
            cfg=ds_loading.ProcessedLibriTTSRDataset.Configuration(
                load_prosody_features=True,
                pitch_quantization_params=None,
                energy_quantization_params=None,
                scale_prosody_features=False,
                use_phoneme_level_prosody_features=False,
                load_context_embeddings=True
            ),
            utterances=list(self._ds_handler.iter_utterances())
        )
        self._processor_cfg = processor_cfg

    def save_stats(self, output_dir: pathlib.Path) -> None:
        """Saves the collected statistics about the processed dataset to the output path."""

        output_dir.mkdir(parents=True, exist_ok=True)

        samples_df = self._construct_samples_df()

        self._save_t_sne_scatter_for_spk_embeddings(output_dir)

        with output_dir.joinpath('samples_stats.json').open('w', encoding='utf-8') as f:
            samples_df.describe(
                percentiles=[0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
            ).to_json(f, indent=4)

        _logger().info('Saving spectrograms and contours for certain samples...')

        example_utterances = {
            'smallest_input_phonemes_length': samples_df.nsmallest(5, 'input_phonemes_length'),
            'largest_input_phonemes_length': samples_df.nlargest(5, 'input_phonemes_length'),
            'smallest_input_spec_length': samples_df.nsmallest(5, 'input_spec_length'),
            'largest_input_spec_length': samples_df.nlargest(5, 'input_spec_length'),
            'smallest_mean_f0': samples_df.nsmallest(5, 'mean_f0'),
            'largest_mean_f0': samples_df.nlargest(5, 'mean_f0'),
            'smallest_mean_energy': samples_df.nsmallest(5, 'mean_energy'),
            'largest_mean_energy': samples_df.nlargest(5, 'mean_energy'),
            'smallest_speaking_rate': samples_df.nsmallest(5, 'spk_rate'),
            'largest_speaking_rate': samples_df.nlargest(5, 'spk_rate')
        }

        for tag, examples_df in example_utterances.items():
            for rank, (_, row) in enumerate(examples_df.iterrows()):

                example_dir = output_dir.joinpath('examples').joinpath(f'{tag}/{rank}')
                example_dir.mkdir(parents=True, exist_ok=True)

                sample = self._dataset[int(row['sample_idx'])]
                metadata = self._dataset.get_sample_metadata(int(row['sample_idx']))

                viz_utils.plot_and_save_spectrogram(
                    sample['input_spec'].cpu().numpy(),
                    sr=self._processor_cfg.audio_processor_cfg.sr,
                    hop_length=self._processor_cfg.audio_processor_cfg.hop_length,
                    title=('Input spectrogram - '
                           f'Mean F0: {row["mean_f0"]:.2f} Hz - '
                           f'Mean energy: {row["mean_energy"]:.2f} - '
                           f'Speaking rate: {row["spk_rate"]:.2f} spec frames / word'),
                    output_path=example_dir.joinpath('input_spectrogram.png'))

                viz_utils.plot_and_save_contour(
                    sample['input_f0'].cpu().numpy(),
                    contour_name='Input F0',
                    output_path=example_dir.joinpath('input_f0_contour.png'))

                viz_utils.plot_and_save_contour(
                    sample['input_energy'].cpu().numpy(),
                    contour_name='Input energy',
                    output_path=example_dir.joinpath('input_energy_contour.png'))

                viz_utils.plot_and_save_contour(
                    sample['explicit_durations'].cpu().numpy(),
                    contour_name='Explicit durations',
                    output_path=example_dir.joinpath('explicit_durations_contour.png'))

    def _save_t_sne_scatter_for_spk_embeddings(self, output_dir: pathlib.Path) -> None:
        """Saves a T-SNE scatter plot of speaker embeddings separated by speaker gender."""

        _logger().info('Saving speaker-embedding T-SNE scatter plots...')

        tsne_df = self._obtain_t_sne_spk_embeddings_scatter()

        plot_dir = output_dir.joinpath('embeddings_tsne')
        plot_dir.mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.scatterplot(data=tsne_df,
                        x='tsne_x',
                        y='tsne_y',
                        hue='spk_gender',
                        hue_order=['M', 'F'],
                        palette={
                            'M': eda_utils.DARK_COLOR_STD,
                            'F': eda_utils.LIGHT_COLOR_STD,
                        },
                        s=70,
                        alpha=0.85,
                        edgecolor='none',
                        ax=ax)

        ax.set_title('T-SNE of Speaker Embeddings')
        ax.set_xlabel('T-SNE component 1')
        ax.set_ylabel('T-SNE component 2')
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.legend(title='Speaker gender')

        fig.tight_layout()
        fig.savefig(plot_dir.joinpath('speaker_embeddings_by_gender.png'), dpi=200)
        plt.close(fig)

    def _obtain_t_sne_spk_embeddings_scatter(self) -> pd.DataFrame:
        """Computes a T-SNE projection for speaker embeddings and returns scatter coordinates."""

        metadata = librittsr_helpers.LibriTTSRMetadata()

        embeddings_for_speakers = []

        for spk in self._ds_handler.iter_speakers():

            embedding = torch.load(spk.embedding_path).numpy().reshape(-1)

            first_utt = next(self._ds_handler.iter_utterances(speaker_id=spk.spk_id))
            spk_info, _ = metadata.get_speaker_and_book(spk.spk_id, first_utt.raw_utterance.chap_id)

            embeddings_for_speakers.append({
                'spk_id': spk.spk_id,
                'spk_gender': spk_info.gender,
                'spk_emb': embedding
            })

        emb_matrix = np.stack([row['spk_emb'] for row in embeddings_for_speakers], axis=0)

        perplexity = min(30.0, float(len(embeddings_for_speakers) - 1))
        tsne = TSNE(n_components=2,
                    random_state=42,
                    init='random',
                    learning_rate='auto',
                    perplexity=perplexity)
        emb_2d = tsne.fit_transform(emb_matrix)

        return pd.DataFrame([
            {
                'spk_id': row['spk_id'],
                'spk_gender': row['spk_gender'],
                'tsne_x': float(emb_2d[idx, 0]),
                'tsne_y': float(emb_2d[idx, 1]),
            }
            for idx, row in enumerate(embeddings_for_speakers)
        ])

    def _construct_samples_df(self) -> pd.DataFrame:
        """Constructs a DataFrame with information about the samples in the processed dataset."""

        def iter_rows():  # type: ignore

            for sample_idx in range(len(self._dataset)):

                sample = self._dataset[sample_idx]

                yield {
                    'sample_idx': sample_idx,
                    'n_words_embeddings': sample['input_word_emb'].shape[0],
                    'input_phonemes_length': int(sample['input_phonemes_length'].item()),
                    'input_spec_length': int(sample['input_spec_length'].item()),
                    'sentence_pos': int(sample['sentence_pos'].item()),
                    'spk_rate': float(sample['spk_rate'].item()),
                    'mean_f0': float(sample['input_f0'].mean().item()),
                    'min_f0': float(sample['input_f0'].min().item()),
                    'max_f0': float(sample['input_f0'].max().item()),
                    'mean_energy': float(sample['input_energy'].mean().item()),
                    'min_energy': float(sample['input_energy'].min().item()),
                    'max_energy': float(sample['input_energy'].max().item()),
                    'context_tokens_length': int(sample['context_tokens_length'].item()),
                    'context_pse_length': int(sample['context_pse_length'].item())
                }

        return pd.DataFrame(tqdm.tqdm(iter_rows(),
                                      total=len(self._dataset),
                                      desc='Constructing samples DataFrame'))
