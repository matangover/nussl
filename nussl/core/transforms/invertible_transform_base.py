#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Base class for invertible spectral transforms in nussl
"""
from __future__ import division

import numpy as np
import scipy.signal

from .. import utils
from .. import constants
import nussl.separation.masks


class InvertibleSpectralTransformationBase(object):
    """
    Base class for spectral transforms
    """

    NAME = __name__

    def __init__(self, audio_data=None, transformation_data=None):
        """

        Args:
            audio_data:
            transformation_data:

        Returns:

        """

        if audio_data is not None and transformation_data is not None:
            raise TransformationBaseException('Cannot initialize with audio_data '
                                              'and transformation_data!')

        self._audio_data = audio_data
        self._transformation_data = transformation_data

    @property
    def is_empty(self):
        """
        Returns true if there is no :attr:`audio_data` and no :attr:`transformation_data`.

        Returns:
            `not (:attr:`has_audio_data` or :attr:`has_transformation_data`)`

        See Also:
            :attr:`has_audio_data`, :attr:`has_transformation_data`.

        """
        return not (self.has_audio_data or self.has_transformation_data)

    @property
    def has_audio_data(self):
        """
        Returns if this transformation has any information in :attr:`audio_data`.

        Returns:
            True if :attr:`audio_data` is not `None` and :attr:`audio_data.size` > 0.

        See Also:
            :attr:`has_transformation_data`, :attr:`is_empty`.

        """
        return self.audio_data is not None and self.audio_data.size > 0

    @property
    def has_transformation_data(self):
        """
        Returns if this transformation has any information in :attr:`transformation_data`.

        Returns:
            True if :attr:`transformation_data` is not `None`
            and :attr:`transformation_data.size` > 0.

        See Also:
            :attr:`has_audio_data`, :attr:`is_empty`.

        """
        return self.transformation_data is not None and self.transformation_data.size > 0

    def transform(self):
        """

        Returns:

        """

    def inverse_transform(self):
        """

        Returns:

        """

    def plot(self):
        """

        Returns:

        """

    @property
    def audio_data(self):
        """ (:obj:`np.ndarray`): Real-valued, uncompressed, time-domain transformation of the audio.
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
    def transformation_data(self):
        """ (:obj:`np.ndarray`): Complex-valued, time-frequency transformation of the audio.
            2D numpy array with shape `(n_frequency_bins, n_time_bins)`.
            ``None`` by default, this can be initialized at instantiation.
            Usually, this is expected to be floats. Some functions will convert to floats
            if not already.
        """

        return self._transformation_data

    @transformation_data.setter
    def transformation_data(self, value):
        self._transformation_data = utils._verify_transformation_data(value)

    @property
    def num_channels(self):
        """ (int): Number of channels this Transformation object has.
            Defaults to returning number of channels in :attr:`transformation_data`.
            If no data ``None`` then returns ``None``.
        """
        if self.transformation_data is not None:
            return self.transformation_data.shape[constants.TF_CHAN_INDEX]
        return None

    @property
    def is_mono(self):
        """
        PROPERTY
        Returns:
            (bool): Whether or not this signal is mono (i.e., has exactly `one` channel).

        """
        return self.num_channels == 1

    @property
    def is_stereo(self):
        """
        PROPERTY
        Returns:
            (bool): Whether or not this signal is stereo (i.e., has exactly `two` channels).

        """
        return self.num_channels == 2

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
         :attr:`transformation_data`.

        Returns:
            (:class:`MaskBase`): with the shape of :attr:`transformation_data`.
        """

        if mask_type == constants.BINARY_MASK:
            return nussl.separation.masks.BinaryMask.zeros(self.transformation_data.shape)
        else:
            return nussl.separation.masks.SoftMask.zeros(self.transformation_data.shape)

    def make_ones_mask(self, mask_type=constants.BINARY_MASK):
        """
        Creates a binary or soft mask filled with ONES that is the exact shape as
         :attr:`transformation_data`.

        Returns:
            (:class:`MaskBase`): with the shape of :attr:`transformation_data`.
        """

        if mask_type == constants.BINARY_MASK:
            return nussl.separation.masks.BinaryMask.ones(self.transformation_data.shape)
        else:
            return nussl.separation.masks.SoftMask.ones(self.transformation_data.shape)

    def apply_mask(self, mask, overwrite=False):
        """
        Applies a mask to :attr:`transformation_data`.
        Args:
            mask:
            overwrite:

        Returns:

        """

        if self.transformation_data is None or self.transformation_data.size == 0:
            raise TransformationBaseException('Cannot apply mask when '
                                              'self.transformation_data is empty!')

        if not isinstance(mask, nussl.separation.masks.MaskBase):
            raise TransformationBaseException('mask is {} but is expected to be a '
                                              'MaskBase-derived object!'.format(type(mask)))

        if mask.shape != self.transformation_data.shape:
            raise TransformationBaseException('Input mask and self.transformation_data are not the '
                                              'same shape! mask: {}, self.transformation_data: {}'.
                                              format(mask.shape, self.transformation_data.shape))

        masked_transformation = self.transformation_data * mask.mask

        if overwrite:
            self.transformation_data = masked_transformation

        return masked_transformation


class TransformationBaseException(Exception):
    """
    Exception class for RepresentationBase
    """
    pass
