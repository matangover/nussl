#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for all representations in nussl
"""

import numpy as np
import scipy.signal

from ...core import constants
from ...core import utils
from ...separation import masks


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
            raise RepresentationBaseException('Cannot initialize with audio_data '
                                              'and representation_data!')

        self._audio_data = audio_data
        self._representation_data = representation_data

    def forward(self):
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
            Usually, this is expected to be floats. Some functions will convert to floats
            if not already.
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
            Usually, this is expected to be floats. Some functions will convert to floats
            if not already.
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
            symmetric (bool): False (default) generates a periodic window (for use in
            spectral analysis).
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

        # if no overlap, then we need to get another hop at the end
        num_blocks += 1 if overlap == 0 else 0

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

    def make_zeros_mask(self, mask_type=constants.BINARY_MASK):
        """
        Creates a binary or soft mask filled with ZEROS that is the exact shape as
         :attr:`representation_data`.

        Returns:
            (:class:`MaskBase`): with the shape of :attr:`representation_data`.
        """

        if mask_type == constants.BINARY_MASK:
            return masks.BinaryMask.zeros(self.representation_data.shape)
        else:
            return masks.SoftMask.zeros(self.representation_data.shape)

    def make_ones_mask(self, mask_type=constants.BINARY_MASK):
        """
        Creates a binary or soft mask filled with ONES that is the exact shape as
         :attr:`representation_data`.

        Returns:
            (:class:`MaskBase`): with the shape of :attr:`representation_data`.
        """

        if mask_type == constants.BINARY_MASK:
            return masks.BinaryMask.ones(self.representation_data.shape)
        else:
            return masks.SoftMask.ones(self.representation_data.shape)

    def apply_mask(self, mask, overwrite=False):
        """
        Applies a mask to :attr:`representation_data`.
        Args:
            mask:
            overwrite:

        Returns:

        """

        if self.representation_data is None or self.representation_data.size == 0:
            raise RepresentationBaseException('Cannot apply mask when '
                                              'self.representation_data is empty!')

        if not isinstance(mask, masks.MaskBase):
            raise RepresentationBaseException('mask is {} but is expected to be a '
                                              'MaskBase-derived object!'.format(type(mask)))

        if mask.shape != self.representation_data.shape:
            raise RepresentationBaseException('Input mask and self.representation_data are not the '
                                              'same shape! mask: {}, self.representation_data: {}'.
                                              format(mask.shape, self.representation_data.shape))

        masked_representation = self.representation_data * mask.mask

        if overwrite:
            self.representation_data = masked_representation

        return masked_representation


class RepresentationBaseException(Exception):
    pass