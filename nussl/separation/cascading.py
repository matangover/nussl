#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Combining different foreground/background algorithms
"""

from __future__ import division

import warnings

import numpy as np

import mask_separation_base
import masks
from ..core import stft_utils
from ..core import utils
from ..core import constants

from ft2d import FT2D
from repet import Repet
from repet_sim import RepetSim


class Cascading(mask_separation_base.MaskSeparationBase):
    """

    """

    def __init__(self, input_audio_signal, algorithm1, algorithm2, series_or_parallel='series',
                 mask_type=constants.SOFT_MASK, bg_weight=0.5, fg_weight=0.5):
        super(Cascading, self).__init__(input_audio_signal, mask_type=mask_type)

        if not isinstance(algorithm1, mask_separation_base.MaskSeparationBase) or \
            not isinstance(algorithm2, mask_separation_base.MaskSeparationBase):
            raise ValueError('')

        self.algorithm1 = algorithm1
        self.algorithm2 = algorithm2
        self.series_or_parallel = series_or_parallel
        self.bg_weight = bg_weight
        self.fg_weight = fg_weight

    @staticmethod
    def weiner_filter_masks(mask1, mask2):
        den = mask1 + mask2
        return masks.SoftMask(mask1 / den), masks.SoftMask(mask2 / den)

    def run(self):

        if self.series_or_parallel == 'parallel':

            self.algorithm1.audio_signal = self.algorithm2.audio_signal = self.audio_signal

            r_bk_mask, r_fg_mask = self.algorithm1.run()
            mel_bk_mask, mel_bk_mask = self.algorithm2.run()

            # Combine the masks
            bk_mask = r_bk_mask.mask * self.bg_weight + ((1 - self.bg_weight) * mel_bk_mask.mask)
            mel_mask = r_fg_mask.mask * self.fg_weight + ((1 - self.fg_weight) *  mel_bk_mask.mask)

            # Make into masks and return
            self.background_mask, self.foreground_mask = self.weiner_filter_masks(bk_mask, mel_mask)
            return self.background_mask, self.foreground_mask

        elif self.series_or_parallel == 'series':

            self.algorithm1.audio_signal = self.audio_signal
            r_bk_mask, r_fg_mask = self.algorithm1.run()
            sig1_bg, sig1_fg = self.algorithm1.make_audio_signals()

            self.algorithm2.audio_signal = sig1_fg



