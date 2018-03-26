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
        :class:`STFT`, :class:`InvertibleSpectralTransformationBase`.
    """

    def __init__(self, audio_data=None, transformation_data=None, num_mels=128,
                 freq_min=0.0, freq_max=None, htk=False, norm=1,
                 sample_rate=constants.DEFAULT_SAMPLE_RATE, power=2.0, **kwargs):
        super(MelSpectrogram, self).__init__(audio_data, transformation_data, sample_rate, **kwargs)

        self.power = power
        self.mel_filter_bank = librosa.filters.mel(self.n_fft_bins, self.sample_rate,
                                                   n_mels=num_mels, fmin=freq_min, fmax=freq_max,
                                                   htk=htk, norm=norm)
        self.inverse_mel_filter_bank = np.linalg.pinv(self.mel_filter_bank)

    @property
    def mel_spectrogram_data(self):
        """

        Returns:

        """
        return self.transformation_data


    def transform(self):
        """

        Returns:

        """
        stft_ = super(MelSpectrogram, self).transform()

        mel_stft = []

        for i in range(stft_.shape[constants.TF_CHAN_INDEX]):
            stft_chan = utils._get_axis(stft_, constants.TF_CHAN_INDEX, i)
            mel_stft.append(librosa.feature.melspectrogram(S=stft_chan.T, sr=self.sample_rate,
                                                           power=self.power, n_fft=self.n_fft_bins,
                                                           hop_length=self.hop_length))

        self.transformation_data = np.array(mel_stft).transpose((1, 2, 0))

        return mel_stft

    def inverse_transform(self, truncate_to_length=None):
        if self.transformation_data is None or self.transformation_data.size == 0:
            raise MelSpectrogramException('No data to compute inverse_transform MelSpectrogram from!')

        istft = super(MelSpectrogram, self).inverse_transform(truncate_to_length)


class MelSpectrogramException(Exception):
    """

    """
    pass