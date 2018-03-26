#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Initialization file for ``nussl``, the Northwestern University Source Separation Library.
"""

import core
from .core.constants import *
from .core.audio_signal import AudioSignal
from .core import utils, stft_utils, data_sets
from .evaluation import *
from .separation import *
from .modellers import *

__all__ = ['core', 'utils', 'stft_utils', 'modellers', 'separation', 'evaluation']


__version__ = '0.1.6'

version = __version__  # aliasing version
short_version = '.'.join(version.split('.')[:-1])

__title__ = 'nussl'
__description__ = 'A flexible sound source separation library.'
__uri__ = 'https://github.com/interactiveaudiolab/nussl'

__author__ = 'E. Manilow, P. Seetharaman, F. Pishdadian'
__email__ = 'ethanmanilow@u.northwestern.edu'

__license__ = 'MIT'
__copyright__ = 'Copyright (c) 2018 Interactive Audio Lab'
