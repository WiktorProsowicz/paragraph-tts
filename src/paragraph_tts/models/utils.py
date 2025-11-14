# -*- coding: utf-8 -*-
"""Contains utilities used by trainable models."""
import logging
import sys
from typing import Any
from typing import Dict

import torch


def _logger() -> logging.Logger:
    return logging.getLogger(__name__)


def optimizer_from_cfg(optimizer_cfg: Dict[str, Any],
                       parameters) -> torch.optim.Optimizer:
    """Creates optimizer from configuration dictionary."""

    opt_name, opt_params = optimizer_cfg['name'], optimizer_cfg['params']

    if opt_name == 'adam':
        return torch.optim.Adam(parameters, **opt_params)

    _logger().critical('Unsupported optimizer type: %s', opt_name)
    sys.exit(1)
