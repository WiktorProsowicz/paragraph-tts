"""Downloads prepared alignments for LibriTTS-R dataset."""

import json
import os
import subprocess
import logging

import hydra
import omegaconf
import gdown

from paragraph_tts.utils import logging_utils


ALIGNMENTS_URL = 'https://drive.google.com/uc?id=1KxTWMBajV0-wdACMalLkh_CzNTA3d5Vy'


def _logger():
    return logging.getLogger(__name__)


@hydra.main(version_base=None, config_path='cfg', config_name='download_alignments')
def main(script_cfg: omegaconf.DictConfig):
    """Downloads alignments."""

    logging_utils.setup_logging('download_alignments')

    _logger().info('Script configuration:\n%s', json.dumps(dict(script_cfg), indent=4))

    if os.path.exists(script_cfg.output_path):
        _logger().info(
            'Output path already exists, skipping alignments download: %s',
            script_cfg.output_path)
        return

    os.makedirs(script_cfg.output_path, exist_ok=True)

    arch_path = os.path.join('/tmp/librittsr_alignments.tar.bz2')
    gdown.download(ALIGNMENTS_URL, arch_path, quiet=False)


    try:
        subprocess.run(['bzip2', '-d', arch_path], check=True)

        tar_path = arch_path[:-4]

        subprocess.run(['tar', '-xf', tar_path, '-C', script_cfg.output_path], check=True)

        subprocess.run(['rm', tar_path], check=True)

    except subprocess.CalledProcessError as proc_err:
        logging.critical('Failed to download the phoneme alignments: %s', proc_err)


if __name__ == '__main__':
    main()  # pylint: disable=E1120
