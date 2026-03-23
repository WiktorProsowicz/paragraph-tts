"""Prepares audio/transcript pairs for MFA alignment."""
import json
import logging
import os

import hydra
import omegaconf
import tqdm
from torch_dev_utils.tts import text_prep

from paragraph_tts.utils import logging_utils
from paragraph_tts.utils.path import raw_libri_dir_handler


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_data_for_mfa')
def main(cfg: omegaconf.DictConfig):
    """Prepares audio/transcript pairs for MFA alignment."""

    logging_utils.setup_logging('prepare_data_for_mfa')

    _logger().info('Config:\n%s', json.dumps(dict(cfg), indent=4))

    os.makedirs(cfg.output_dir, exist_ok=True)

    raw_ds_handler = raw_libri_dir_handler.RawLibriDirHandler(
        cfg.raw_dataset_path)

    def iter_all_utterances():

        for para_info in raw_ds_handler.iter_all_paragraphs():
            yield from para_info.utterances

    for utt_info in tqdm.tqdm(iter_all_utterances()):

        dst_dir = os.path.join(cfg.output_dir,
                               raw_ds_handler.get_split_for_speaker(
                                   utt_info.spk_id),
                               str(utt_info.spk_id),
                               str(utt_info.chap_id))

        file_basename = '%d_%d_%06d_%06d' % (utt_info.spk_id,  # pylint: disable=consider-using-f-string
                                             utt_info.chap_id,
                                             utt_info.para_id,
                                             utt_info.utt_id)

        output_txt_path = os.path.join(dst_dir, file_basename + '.txt')
        output_wav_path = os.path.join(dst_dir, file_basename + '.wav')

        if any(os.path.exists(p) for p in (output_txt_path, output_wav_path)):
            _logger().debug('Skipping already existing utterance: %s', file_basename)

        text = text_prep.TextProcessor.load_text(utt_info.text_path)
        text = text_prep.TextProcessor.clean_text(text)

        text = text.lower().strip()
        text = ''.join(
            filter(lambda x: x in 'abcdefghijklmnopqrstuvwxyz ', text))

        os.makedirs(dst_dir, exist_ok=True)

        with open(output_txt_path, 'w', encoding='utf-8') as f:
            f.write(text)

        if cfg.create_relative_symlinks:
            os.symlink(os.path.relpath(os.path.abspath(utt_info.wav_path),
                                       os.path.abspath(dst_dir)),
                       output_wav_path)
        else:
            os.symlink(os.path.abspath(utt_info.wav_path),
                       output_wav_path)


if __name__ == '__main__':
    main()  # pylint: disable=no-value-for-parameter
