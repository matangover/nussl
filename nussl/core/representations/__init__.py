#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Imports for representations
"""

from .invertible_representation_base import InvertibleRepresentationBase, \
    RepresentationBaseException
from .stft import STFT, STFTException
from .mel_spectrogram import MelSpectrogram
from .cqt import CQT


all_representations = [STFT, MelSpectrogram, CQT]
all_representations_dict = {r.__name__.lower(): r for r in all_representations}


__all__ = ['InvertibleRepresentationBase', 'RepresentationBaseException',
           'STFT', 'STFTException',
           'MelSpectrogram',
           'CQT']
