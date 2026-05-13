"""Contains a module that runs inference on evaluation dataset and saves the results."""

import pathlib

import torch
from speechbrain.inference.vocoders import HIFIGAN
import soundfile

from torch_dev_utils.tts import inference as inference_utils
from torch_dev_utils.tts import visualization as tdu_viz

from paragraph_tts.models.stl_predictor import stl_predictor
from paragraph_tts.models.acoustic import acoustic
from paragraph_tts.utils.path import eval_ds_handler
from paragraph_tts.eval import test_ds_processor
from paragraph_tts.eval import ds_loader


class InferenceEngine:
    """Runs inference on evaluation dataset and saves the results."""

    def __init__(self,
                 acoustic_model: acoustic.AcousticModel,
                 stl_predictor_model: stl_predictor.STLPredictor | None,
                 inference_label: str,
                 inference_device: str) -> None:

        self._acoustic_model = acoustic_model.to(inference_device)

        self._stl_predictor_model = stl_predictor_model
        if self._stl_predictor_model is not None:
            self._stl_predictor_model = stl_predictor_model.to(inference_device)

        self._inference_label = inference_label
        self._inference_device = inference_device

        vocoder: HIFIGAN = HIFIGAN.from_hparams(
            source='speechbrain/tts-hifigan-libritts-22050Hz',
            run_opts={'device': str(inference_device)}
        )
        vocoder.eval()
        for param in vocoder.parameters():
            param.requires_grad = False

        self._vocoder = vocoder

    def run_inference(self,
                      eval_ds_path: pathlib.Path,
                      root_output_dir: pathlib.Path) -> None:
        """Runs inference on evaluation dataset and saves the results."""

        ds_handler = eval_ds_handler.EvalDsHandler(eval_ds_path)

        for input_data in ds_loader.WholeParagraphsDS(list(ds_handler.iter_whole_paragraphs())):

            model_inputs = self._obtain_acoustic_model_inputs(input_data,
                                                              is_whole_paragraph=True)

            para_output_dir = (
                root_output_dir
                .joinpath('whole_paragraphs')
                .joinpath(str(input_data.raw_paragraph.spk_id))
                .joinpath(f'{input_data.raw_paragraph.chap_id}_{input_data.raw_paragraph.para_id}')
            )

            with para_output_dir.joinpath('raw_paragraph.json').open('w') as f:
                f.write(input_data.raw_paragraph.model_dump_json(indent=4))

            with para_output_dir.joinpath('gt_wav.wav').open('wb') as f:
                soundfile.write(f, input_data.gt_wav, 22050)

            self._run_inference_and_save_results(model_inputs,
                                                 para_output_dir / self._inference_label)

    def _obtain_acoustic_model_inputs(self,
                                      input_data: ds_loader.EvalInputData,
                                      is_whole_paragraph: bool) -> dict[str, torch.Tensor]:
        """Composes input tensors for the acoustic model from the input data."""

        inputs: dict[str, torch.Tensor] = {}

        if self._stl_predictor_model is not None:

            with torch.no_grad():
                stl_predictor_outputs = self._stl_predictor_model(
                    input_data.graph_data.to(self._inference_device)
                )

            inputs['gst_weights'] = stl_predictor_outputs['final_gst_weights']
            inputs['wsv_weights'] = stl_predictor_outputs['final_wsv_weights']

            if is_whole_paragraph:
                inputs['gst_to_phone_indices'] = input_data.graph_data.sentence_lengths
            else:
                inputs['gst_to_phone_indices'] = torch.tensor(
                    input_data.input_acoustic_data['input_phoneme_ids'].shape[0])

        inputs.update(input_data.context_tensors)
        inputs.update(input_data.input_acoustic_data)

        return {name: tensor.unsqueeze(0).to(self._inference_device)
                for name, tensor in inputs.items()}

    def _run_inference_and_save_results(self,
                                        model_inputs: dict[str, torch.Tensor],
                                        output_dir: pathlib.Path) -> None:
        """Runs inference for a single sample and saves the results."""

        with torch.no_grad():
            model_outputs = self._acoustic_model(**model_inputs)

            wav = inference_utils.transform_mel_to_wav(
                model_outputs['pred_mel_spec'].squeeze(0).cpu(),
                self._vocoder.decode_batch,
                split_spec_by_silences=True
            )

        soundfile.write(output_dir.joinpath('predicted_sound.wav'), wav.squeeze(0).numpy(), 22050)

        tdu_viz.plot_and_save_contour(
            model_outputs['predicted_energy'].squeeze(0).cpu(),
            'Energy',
            output_dir.joinpath('predicted_energy_contour.svg')
        )

        tdu_viz.plot_and_save_contour(
            model_outputs['predicted_durations'].squeeze(0).cpu(),
            'Duration',
            output_dir.joinpath('predicted_duration_contour.svg')
        )

        tdu_viz.plot_and_save_contour(
            model_outputs['predicted_pitch'].squeeze(0).cpu(),
            'Pitch',
            output_dir.joinpath('predicted_pitch_contour.svg')
        )
