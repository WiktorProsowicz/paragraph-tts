"""Prepares audio/transcript pairs for MFA alignment."""

import hydra
import omegaconf
import os
import tqdm

from paragraph_tts.utils.path import raw_libri_dir_handler
from paragraph_tts.data.preprocessing import text as text_prep


@hydra.main(version_base=None, config_path='cfg', config_name='prepare_data_for_mfa')
def main(cfg: omegaconf.DictConfig):
    """Prepares audio/transcript pairs for MFA alignment."""

    os.makedirs(cfg.output_dir, exist_ok=True)

    raw_ds_handler = raw_libri_dir_handler.RawLibriDirHandler(cfg.raw_dataset_path)

    def iter_all_utterances():

        for para_info in raw_ds_handler.iter_all_paragraphs():
            yield from para_info.utterances

    for utt_info in tqdm.tqdm(iter_all_utterances()):

        text = text_prep.TextProcessor.load_text(utt_info.text_path)
        text = text_prep.TextProcessor.clean_text(text)

        text = text.lower().strip()
        text = ''.join(filter(lambda x: x in 'abcdefghijklmnopqrstuvwxyz ', text))

        file_basename = '%d_%d_%06d_%06d' % (utt_info.spk_id,  # pylint: disable=consider-using-f-string
                                             utt_info.chap_id,
                                             utt_info.para_id,
                                             utt_info.utt_id)

        dst_dir = os.path.join(cfg.output_dir,
                               str(utt_info.spk_id),
                               str(utt_info.chap_id))

        os.makedirs(dst_dir, exist_ok=True)

        with open(os.path.join(dst_dir, file_basename + '.txt'), 'w', encoding='utf-8') as f:
            f.write(text)

        os.symlink(os.path.abspath(utt_info.wav_path),
                   os.path.join(dst_dir, file_basename + '.wav'))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
