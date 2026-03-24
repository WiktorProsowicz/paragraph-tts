"""Contains definition of custom logging tools based on logging lib."""
import datetime
import logging.config
import os
import pathlib
from typing import Any
from typing import Dict

UTILITIES_HOME = pathlib.Path(__file__).absolute().parent.as_posix()


def setup_logging(script_signature: str,
                  output_dir: str = 'log') -> None:
    """Sets up project-wide logging configuration.

    This function should be called at the
    beginning of the scripts run from the console.

    Args:
        script_signature: Name of the script from which the function is called. Will be used to
            determine the log file name.
        output_dir: Directory where the log files will be stored.
    """

    logging_config = _get_logging_config(script_signature, output_dir)

    os.makedirs(os.path.join(output_dir, script_signature), exist_ok=True)

    logging.config.dictConfig(logging_config)


def _get_logging_config(script_signature: str,
                        output_dir: str) -> Dict[str, Any]:
    """Creates a global logging configuration.

    Two handlers are created:
        - a console handler that outputs primary info logs to the console;
        - a file handler that outputs all logs to a file.

    The loggers are designed to disable all logs from 3rd party libraries, whose level is below
        WARNING.

    Returns:
        Compiled configuration ready to be loaded as a configuration
        dictionary to the logging module.
    """

    log_file_path = os.path.join(output_dir,
                                 script_signature,
                                 f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')

    return {
        'version': 1,
        'loggers': {
            'root': {
                'level': 'WARNING',
                'handlers': ['console_hand', 'file_hand'],
            },
            'paragraph_tts': {
                'level': 'DEBUG',
                'handlers': ['console_hand', 'file_hand'],
                'propagate': False,
            }
        },
        'handlers': {
            'console_hand': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'color_formatter',
                'stream': 'ext://sys.stdout',
            },
            'file_hand': {
                'class': 'logging.FileHandler',
                'level': 'NOTSET',
                'formatter': 'default_formatter',
                'filename': log_file_path,
                'encoding': 'utf-8',
            },
        },
        'formatters': {
            'color_formatter': {
                '()': _ColorFormatter,
                'format': '[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
            'default_formatter': {
                'format': '[%(asctime)s][%(levelname)s][%(name)s] %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },

    }


class _ColorFormatter(logging.Formatter):
    """Adds color to the log messages.

    This is the default formatter used by the sound-processing modules.
    """

    _COLORS = {
        'DEBUG': '\033[32m',
        'INFO': '\033[36m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[31;1m',
    }

    _END_COLOR = '\033[0m'

    def format(self, record: logging.LogRecord) -> str:
        """Overrides the default method of `Formatter` class."""

        pre_formatted = super().format(record)

        return f'{self._COLORS[record.levelname]}{pre_formatted}{self._END_COLOR}'


if __name__ == '__main__':

    setup_logging('logging_utils')

    logger = logging.getLogger('paragraph_tts.utils.logging_utils')

    logger.debug('This is a debug message.')
    logger.info('This is an info message.')
    logger.warning('This is a warning message.')
    logger.error('This is an error message.')
    logger.critical('This is a critical message.')
