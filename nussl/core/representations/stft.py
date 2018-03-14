#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Class for computing a Short-Time Fourier Transform (STFT) of audio and inverse STFT (iSTFT) from
STFT data to audio data.
"""
from __future__ import division
import warnings

import numpy as np
import scipy.fftpack as scifft

from ...core import constants
from ...core import utils
import invertible_representation_base


class STFT(invertible_representation_base.InvertibleRepresentationBase):
    """
    This class computes a Short-Time Fourier Transform (STFT) of an input signal and inverse
    Short-Time Fourier Transform (iSTFT) of STFT data.
    This will zero pad the signal by half a hop_length at the beginning to reduce the window
    tapering effect from the first window. It also will zero pad at the end to get an integer
    number of hops.

    By default, this function removes the FFT data that is a reflection from over Nyquist. There is
    an option to suppress this behavior and have this function include data from above Nyquist,
    but since the inverse STFT function expects data without the reflection, the onus is on
    the user to remember to set the reconstruct_reflection flag in e_istft() input.

    Additionally, this function assumes a single channeled audio signal and is not guaranteed to
    work on multichannel audio. If you want to do an STFT on multichannel audio see the AudioSignal
    object.

    Args:
        signal : 1D numpy array containing audio data.
        window_length: (int) number of samples per window
        hop_length: (int) number of samples between the start of adjacent windows, or "hop"
        window_type: (string) type of window to use. Using WindowType object is recommended.
        n_fft_bins (int): number of fft bins per time window.
        If not specified, defaults to next highest power of 2 above window_length
        remove_reflection(bool): if True, this will remove reflected STFT data above the Nyquist
        point.
        If not specified, defaults to True.
        remove_padding (bool): if True, this will remove the extra padding added when doing the
        STFT.
        Defaults to True.

    Returns:
        2D  numpy array with complex STFT data.
        Data is of shape (num_time_blocks, num_fft_bins). These numbers are determined by length of
        the input signal,
        on internal zero padding (explained at top), and n_fft_bins/remove_reflection input
        (see example below).

    Example:

    .. code-block:: python
        :linenos:


        # Set up sine wave parameters
        sr = nussl.Constants.DEFAULT_SAMPLE_RATE # 44.1kHz
        n_sec = 3 # seconds
        duration = n_sec * sr
        freq = 300 # Hz

        # Make sine wave array
        x = np.linspace(0, freq * 2 * np.pi, duration)
        x = np.sin(x)

        # Set up e_stft() parameters
        win_type = nussl.WindowType.HANN
        win_length = 2048
        hop_length = win_length / 2

        # Run e_stft()
        stft = nussl.e_stft(x, win_length, hop_length, win_type)
        # stft has shape (win_length // 2 + 1 , duration / hop_length)

        # Get reflection
        stft_with_reflection = nussl.e_stft(x, win_length, hop_length, win_type,
        remove_reflection=False)
        # stft_with_reflection has shape (win_length, duration / hop_length)

        # Change number of fft bins per hop
        num_bins = 4096
        stft_more_bins = e_stft(x, win_length, hop_length, win_type, n_fft_bins=num_bins)
        # stft_more_bins has shape (num_bins // 2 + 1, duration / hop_length)
    """

    NAME = __name__.lower()

    def __init__(self, audio_data=None, representation_data=None,
                 sample_rate=constants.DEFAULT_SAMPLE_RATE,
                 window_length=constants.DEFAULT_WIN_LENGTH,
                 hop_length=constants.DEFAULT_WIN_LENGTH // 2,
                 window_type=constants.WINDOW_DEFAULT,
                 n_fft_bins=constants.DEFAULT_WIN_LENGTH, reflection=True, pad=False,
                 original_length=None, dtype='float64'):

        super(STFT, self).__init__(audio_data=audio_data, representation_data=representation_data)

        self.window_length = window_length
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        self.n_fft_bins = n_fft_bins if n_fft_bins is not None else self.window_length
        self.reflection = reflection
        self.pad = pad
        self.dtype = dtype

        self.window_type = window_type
        self.window = self.make_window(self.window_type, self.window_length)

        if original_length is not None:
            self.original_length = original_length
        elif self.audio_data is not None:
            self.original_length = self.audio_data.shape[constants.LEN_INDEX]

    @property
    def stft_data(self):
        """
        STFT data, alias to self.representation_data
        Returns:

        """
        return self.representation_data

    @property
    def freq_vector(self):
        """ (:obj:`np.ndarray`): A 1D numpy array with frequency values that correspond
        to each frequency bin (vertical axis) for the STFT.

        Raises:
            AttributeError: If :attr:`stft_data` is ``None``.
            Run :func:`stft` before accessing this.

        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate freq_vector until self.stft() is run')
        return np.linspace(0.0, self.sample_rate // 2,
                           num=self.stft_data.shape[constants.STFT_VERT_INDEX])

    @property
    def time_bins_vector(self):
        """(:obj:`np.ndarray`): A 1D numpy array with time values that correspond
        to each time bin (horizontal axis) in the STFT.

        Raises:
            AttributeError: If :attr:`stft_data` is ``None``.
            Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate time_bins_vector until self.stft() is run')
        return np.linspace(0.0, self.self.audio_data.shape[constants.LEN_INDEX] / self.sample_rate,
                           num=self.stft_data.shape[constants.STFT_LEN_INDEX])

    @property
    def stft_length(self):
        """ (int): The number of time windows the STFT has.
        Raises:
            AttributeError: If ``self.stft_dat``a is ``None``.
            Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate stft_length until self.stft() is run')
        return self.stft_data.shape[constants.STFT_LEN_INDEX]

    @property
    def num_fft_bins(self):
        """ (int): Number of FFT bins in self.stft_data
        Raises:
            AttributeError: If :attr:`stft_data` is ``None``.
            Run :func:`stft` before accessing this.
        """
        if self.stft_data is None:
            raise AttributeError('Cannot calculate num_fft_bins until self.stft() is run')
        return self.stft_data.shape[constants.STFT_VERT_INDEX]

    def forward(self):
        """

        Returns:

        """
        if self.audio_data is None or self.audio_data.size == 0:
            raise STFTException('No time domain signal (self.audio_data) to compute STFT from!')

        if self._representation_data is not None:
            warnings.warn('Overwriting self.stft_data data')

        # check for reflection
        stft_bins = self.n_fft_bins // 2 + 1 if self.reflection else self.n_fft_bins

        stfts = []
        for i in range(self.audio_data.shape[constants.CHAN_INDEX]):
            signal = utils._get_axis(self.audio_data, constants.CHAN_INDEX, i)
            signal, num_blocks = self._add_zero_padding(signal, self.window_length,
                                                        self.hop_length)
            stfts.append(self._forward_process_channel(signal, num_blocks, stft_bins))

        # save with the correct shape
        self.representation_data = np.array(stfts).transpose((1, 2, 0))
        return self.representation_data

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

        if self.pad:
            stft = self._remove_padding(stft, self.original_length,
                                        self.window_length, self.hop_length)

        return stft

    def inverse(self, truncate_to_length=None):
        """

        Returns:

        """

        if self.representation_data is None or self.representation_data.size == 0:
            raise STFTException('No stft_data to compute iSTFT from!')

        audio_signal = []

        for i in range(self.representation_data[constants.STFT_CHAN_INDEX]):
            chan = utils._get_axis(self.representation_data,
                                              constants.STFT_CHAN_INDEX, i)
            audio_signal.append(self._inverse_process_channel(chan))

        audio_signal = np.array(audio_signal)
        audio_signal = np.expand_dims(audio_signal, -1) if audio_signal.ndim == 1 else audio_signal

        # if truncate_to_length isn't provided
        if truncate_to_length is None:
            if self.original_length is not None:
                truncate_to_length = self.original_length

        if truncate_to_length is not None and truncate_to_length > 0:
            audio_signal = audio_signal[:, :truncate_to_length]

        self.audio_data = audio_signal
        return self.audio_data

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


class STFTException(Exception):
    pass
