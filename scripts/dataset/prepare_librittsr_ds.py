"""Runs preprocessing on raw LibriTTS-R dataset and saves the preprocessed files."""

import hydra
import omegaconf
from paragraph_tts import (utils, data)


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_librittsr_ds')
def main(script_cfg: omegaconf.DictConfig):
    """Runs LibriTTS-R preprocessing."""

    raw_ds_handler = utils.path.RawLibriDirHandler(script_cfg.raw_ds_path)
    preprocessor = data.librittsr.LibriTTSRPreprocessor(raw_ds_handler,
                                                        script_cfg.processed_ds_output_path,
                                                        script_cfg.multi_speaker)

    preprocessor.run()

if __name__ == '__main__':
    main()  # pylint: disable=E1120
