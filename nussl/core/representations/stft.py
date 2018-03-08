#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

"""
import warnings

import numpy as np
import scipy.fftpack as scifft

from ...core import constants
import invertible_representation_base


class STFT(invertible_representation_base.InvertibleRepresentationBase):
    """

    """

    NAME = __name__.lower()

    def __init__(self, sample_rate, audio_data=None, representation_data=None,
                 window_length=None, hop_length=None, window_type=constants.WINDOW_DEFAULT,
                 n_fft_bins=None, reflection=True, pad=False,
                 original_length=None, dtype=None):
        """

        Args:
            window_length:
            hop_length:
            window_type:
            audio_data:
            representation_data:
            n_fft_bins:
            reflection:
            pad:
            original_length:
            dtype:
        """

        super(STFT, self).__init__(audio_data=audio_data, representation_data=representation_data)

        default_win_len = int(2 ** (np.ceil(np.log2(constants.DEFAULT_WIN_LEN_PARAM * sample_rate))))
        self.window_length = default_win_len if window_length is None else int(window_length)
        self.hop_length = self.window_length // 2 if hop_length is None else int(hop_length)

        self.n_fft_bins = n_fft_bins if n_fft_bins is not None else window_length
        self.reflection = reflection
        self.pad = pad

        self.window_type = window_type
        self.window = self.make_window(window_type, window_length)

        self.original_length = original_length

    @property
    def stft_data(self):
        """
        STFT data, alias to self.representation_data
        Returns:

        """
        return self.representation_data

    def forward(self):
        """

        Returns:

        """
        if self._audio_data is None:
            raise invertible_representation_base.RepresentationBaseException('Cannot do forward STFT without audio data!')

        if self._spectral_data is not None:
            warnings.warn('Overwriting spectral data')

        self.original_length = len(self._audio_data)
        signal, num_blocks = self._add_zero_padding(self._audio_data, self.window_length, self.hop_length)

        # only want just over half of each fft
        stft_bins = self.n_fft_bins // 2 + 1 if self.reflection else self.n_fft_bins

        stfts = []
        for i in range(self._audio_data[constants.CHAN_INDEX]):
            stfts.append(self._forward_process_channel(signal, num_blocks, stft_bins))

        self._spectral_data = np.array(stfts).transpose((1, 2, 0))
        return self._spectral_data

    def _forward_process_channel(self, signal, num_blocks, stft_bins):

        # this is where we do the stft calculation
        stft = np.zeros((num_blocks, stft_bins), dtype=complex)
        for hop in range(num_blocks):
            start = hop * self.hop_length
            end = start + self.window_length
            unwindowed_signal = signal[start:end]
            windowed_signal = np.multiply(unwindowed_signal, self.window)
            fft = scifft.fft(windowed_signal, n=self.n_fft_bins)
            stft[hop,] = fft[:stft_bins]

        # reshape the 2d array, so it's (n_fft, n_hops).
        stft = stft.T
        stft = self._remove_padding(stft, self.original_length, self.window_length, self.hop_length) \
            if self.pad else stft

        return stft

    def inverse(self):
        """

        Returns:

        """

    def _inverse_process_channel(self, stft):
        """

        Returns:

        """
        n_hops = stft.shape[1]
        overlap = self.window_length - self.hop_length
        signal_length = (n_hops * self.hop_length) + overlap
        signal = np.zeros(signal_length)

        norm_window = np.zeros(signal_length)

        # Add reflection back
        stft = self._add_reflection(stft) if self.reflection else stft

        for n in range(n_hops):
            start = n * self.hop_length
            end = start + self.window_length
            inv_sig_temp = np.real(scifft.ifft(stft[:, n]))
            signal[start:end] += inv_sig_temp[:self.window_length]
            norm_window[start:end] = norm_window[start:end] + self.window

        norm_window[norm_window == 0.0] = constants.EPSILON  # Prevent dividing by zero
        signal_norm = signal / norm_window

        # remove zero-padding
        if self.pad:
            if overlap >= self.hop_length:
                ovp_hop_ratio = int(np.ceil(overlap / self.hop_length))
                start = ovp_hop_ratio * self.hop_length
                end = signal_length - overlap

                signal_norm = signal_norm[start:end]

            else:
                signal_norm = signal_norm[self.hop_length:]

        return signal_norm

    @staticmethod
    def _add_reflection(matrix):
        reflection = matrix[-2:0:-1, :]
        reflection = reflection.conj()
        return np.vstack((matrix, reflection))
