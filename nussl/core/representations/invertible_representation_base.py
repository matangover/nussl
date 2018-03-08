#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for invertible representations in nussl
"""
import numpy as np
import scipy.signal

from ...core import constants
from ...core import utils


class InvertibleRepresentationBase(object):
    """
    Base class for spectral representations
    """

    NAME = __name__

    def __init__(self, audio_data=None, representation_data=None):
        """

        Args:
            audio_data:

        Returns:

        """

        if audio_data is not None and representation_data is not None:
            raise RepresentationBaseException('Cannot initialize with audio_data and representation_data!')

        self._audio_data = audio_data
        self._representation_data = representation_data

    def forward(self):
        """

        Returns:

        """

    def inverse(self):
        """

        Returns:

        """

    def plot(self):
        """

        Returns:

        """

    @property
    def audio_data(self):
        """ (:obj:`np.ndarray`): Real-valued, uncompressed, time-domain representation of the audio.
            2D numpy array with shape `(n_channels, n_samples)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        """

        return self._audio_data

    @audio_data.setter
    def audio_data(self, value):
        self._audio_data = utils._verify_audio_data(value)

    @property
    def representation_data(self):
        """ (:obj:`np.ndarray`): Complex-valued, time-frequency representation of the audio.
            2D numpy array with shape `(n_frequency_bins, n_time_bins)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats if not already.
        """

        return self._representation_data

    @representation_data.setter
    def representation_data(self, value):
        self._representation_data = utils._verify_representation_data(value)

    @staticmethod
    def make_window(window_type, length, symmetric=False):
        """Returns an `np.array` populated with samples of a normalized window of type `window_type`

        Args:
            window_type (basestring): Type of window to create, string can be
            length (int): length of window
            symmetric (bool): False (default) generates a periodic window (for use in spectral analysis).
                True generates a symmetric window (for use in filter design).
                Does nothing for rectangular window

        Returns:
            window (np.array): np array with a window of type window_type
        """

        # Generate samples of a normalized window
        if window_type == constants.WINDOW_RECTANGULAR:
            return np.ones(length)
        elif window_type == constants.WINDOW_HANN:
            return scipy.signal.hann(length, symmetric)
        elif window_type == constants.WINDOW_BLACKMAN:
            return scipy.signal.blackman(length, symmetric)
        elif window_type == constants.WINDOW_HAMMING:
            return scipy.signal.hamming(length, symmetric)
        elif window_type == constants.WINDOW_TRIANGULAR:
            return scipy.signal.triang(length, symmetric)
        else:
            return None

    @staticmethod
    def _add_zero_padding(signal, window_length, hop_length):
        """

        Args:
            signal:
            window_length:
            hop_length:
        Returns:
        """
        original_signal_length = len(signal)
        overlap = window_length - hop_length
        num_blocks = np.ceil(len(signal) / hop_length)

        if overlap >= hop_length:  # Hop is less than 50% of window length
            overlap_hop_ratio = np.ceil(overlap / hop_length)

            before = int(overlap_hop_ratio * hop_length)
            after = int((num_blocks * hop_length + overlap) - original_signal_length)

            signal = np.pad(signal, (before, after), 'constant', constant_values=(0, 0))
            extra = overlap

        else:
            after = int((num_blocks * hop_length + overlap) - original_signal_length)
            signal = np.pad(signal, (hop_length, after), 'constant', constant_values=(0, 0))
            extra = window_length

        num_blocks = int(np.ceil((len(signal) - extra) / hop_length))
        num_blocks += 1 if overlap == 0 else 0  # if no overlap, then we need to get another hop at the end

        return signal, num_blocks

    @staticmethod
    def _remove_padding(stft, original_signal_length, window_length, hop_length):
        """

        Args:
            stft:
            original_signal_length:
            window_length:
            hop_length:

        Returns:

        """
        overlap = window_length - hop_length
        first = int(np.ceil(overlap / hop_length))
        num_col = int(np.ceil((original_signal_length - window_length) / hop_length))
        stft_cut = stft[:, first:first + num_col]
        return stft_cut


class RepresentationBaseException(Exception):
    pass