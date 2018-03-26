#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Imports for transforms
"""

from .invertible_transform_base import InvertibleSpectralTransformationBase, \
    TransformationBaseException
from .stft import STFT, STFTException
from .mel_spectrogram import MelSpectrogram
from .cqt import CQT


all_representations = [STFT, MelSpectrogram, CQT]
all_transformations_dict = {r.__name__.lower(): r for r in all_representations}


__all__ = ['InvertibleSpectralTransformationBase', 'TransformationBaseException',
           'STFT', 'STFTException',
           'MelSpectrogram',
           'CQT']
