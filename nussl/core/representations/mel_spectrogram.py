#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Spectrogram with Mel-spaced frequency bins
"""

import numpy as np
import librosa

import stft
from ...core import constants
from ...core import utils

class MelSpectrogram(stft.STFT):
    """
    See Also:
        :class:`STFT`, :class:`InvertibleRepresentationBase`.
    """

    def __init__(self, audio_data=None, representation_data=None,
                 sample_rate=constants.DEFAULT_SAMPLE_RATE, power=2.0, **kwargs):
        super(MelSpectrogram, self).__init__(audio_data, representation_data, sample_rate, **kwargs)

        self.power = power

    @property
    def mel_spectrogram_data(self):
        """

        Returns:

        """
        return self.representation_data


    def forward(self):
        """

        Returns:

        """
        stft_ = super(MelSpectrogram, self).forward()

        mel_stft = []

        for i in range(stft_.shape[constants.STFT_CHAN_INDEX]):
            stft_chan = utils._get_axis(stft_, constants.STFT_CHAN_INDEX, i)
            mel_stft.append(librosa.feature.melspectrogram(S=stft_chan.T, sr=self.sample_rate,
                                                           power=self.power, n_fft=self.n_fft_bins,
                                                           hop_length=self.hop_length))

        self.representation_data = np.array(mel_stft).transpose((1, 2, 0))

        return mel_stft
