#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

"""
import warnings

import numpy as np
import librosa

import invertible_transform_base
from ...core import constants
from ...core import utils

class CQT(invertible_transform_base.InvertibleSpectralTransformationBase):

    def __init__(self, audio_data=None, transformation_data=None, num_bins=84,
                 freq_min=None, sample_rate=constants.DEFAULT_SAMPLE_RATE,
                 hop_length=512, bins_per_octave=12, tuning=0.0,
                 filter_scale=1, norm=1, sparsity=0.01,
                 window_type=constants.WINDOW_HANN, scale=True,
                 pad_mode='reflect', amin=1e-06):
        super(CQT, self).__init__(audio_data, transformation_data)

        self.num_bins = num_bins
        self.freq_min = freq_min
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.bins_per_octave = bins_per_octave
        self.tuning = tuning
        self.filter_scale = filter_scale
        self.norm = norm
        self.sparsity = sparsity
        self.window_type = window_type
        self.scale = scale
        self.pad_mode = pad_mode
        self.amin = amin

    @property
    def cqt_data(self):
        """

        Returns:

        """
        return self.transformation_data

    def transform(self):

        if self.audio_data is None or self.audio_data.size == 0:
            raise CQTException('No time domain signal (self.audio_data) to compute CQT from!')

        if self.transformation_data is not None:
            warnings.warn('Overwriting self.stft_data data')

        cqts = []
        for i in range(self.audio_data.shape[constants.CHAN_INDEX]):
            signal = utils._get_axis(self.audio_data, constants.CHAN_INDEX, i)
            cqts.append(librosa.core.cqt(y=signal, sr=self.sample_rate, hop_length=self.hop_length,
                                         fmin=self.freq_min, n_bins=self.num_bins,
                                         bins_per_octave=self.bins_per_octave, tuning=self.tuning,
                                         filter_scale=self.filter_scale, norm=self.norm,
                                         sparsity=self.sparsity, window=self.window_type,
                                         scale=self.scale, pad_mode=self.pad_mode))

        self.transformation_data = np.array(cqts).transpose((1, 2, 0))
        return self.transformation_data

    def inverse_transform(self):
        if self.transformation_data is None or self.transformation_data.size == 0:
            raise CQTException('No stft_data to compute iSTFT from!')

        signal = []

        for i in range(self.transformation_data[constants.TF_CHAN_INDEX]):
            chan = utils._get_axis(self.transformation_data,
                                   constants.TF_CHAN_INDEX, i)

            signal.append(librosa.core.icqt(C=chan, sr=self.sample_rate, hop_length=self.hop_length,
                                            fmin=self.freq_min,
                                            bins_per_octave=self.bins_per_octave,
                                            tuning=self.tuning, filter_scale=self.filter_scale,
                                            norm=self.norm,sparsity=self.sparsity,
                                            window=self.window_type, scale=self.scale,
                                            amin=self.amin))

        self.audio_data = signal

        return signal


class CQTException(Exception):
    """
    Exception class for Constant-Q Transform
    """
    pass
