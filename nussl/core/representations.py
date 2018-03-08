#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Classes for spectral representations in nussl.
"""
import warnings

import numpy as np
import scipy.fftpack as scifft
import scipy.signal

import constants
import utils






class StftParams(object):
    """
    The StftParams class is a container for information needed to run an STFT or iSTFT.
    This is meant as a convenience and does not actually perform any calculations within. It should
    get "decomposed" by the time e_stft() or e_istft() are called, so that every attribute in this
    object is a parameter to one of those functions.

    Every class that inherits from the SeparationBase class has an StftParms object, and this
    is the only way that a top level user has access to the STFT parameter settings that
    all of the separation algorithms are built upon.
    This object will get passed around instead of each of these individual attributes.
    """
    def __init__(self, sample_rate, window_length=None, hop_length=None, window_type=None, n_fft_bins=None):
        self.sample_rate = int(sample_rate)

        # default to 40ms windows
        default_win_len = int(2 ** (np.ceil(np.log2(constants.DEFAULT_WIN_LEN_PARAM * sample_rate))))
        self._window_length = default_win_len if window_length is None else int(window_length)
        self._hop_length = self._window_length // 2 if hop_length is None else int(hop_length)
        self.window_type = constants.WINDOW_DEFAULT if window_type is None else window_type
        self._n_fft_bins = self._window_length if n_fft_bins is None else int(n_fft_bins)

        self._hop_length_needs_update = True
        self._n_fft_bins_needs_update = True

        if hop_length is not None:
            self._hop_length_needs_update = False

        if n_fft_bins is not None:
            self._n_fft_bins_needs_update = False

    @property
    def window_length(self):
        return self._window_length

    @window_length.setter
    def window_length(self, value):
        """
        Length of stft window in samples. If window_overlap
        or num_fft are not set manually, then changing this will update them to
        hop_length = window_length // 2, and and num_fft = window_length
        This property is settable.
        """
        self._window_length = value

        if self._n_fft_bins_needs_update:
            self._n_fft_bins = value

        if self._hop_length_needs_update:
            self._hop_length = value // 2

    @property
    def hop_length(self):
        return self._hop_length

    @hop_length.setter
    def hop_length(self, value):
        """
        Number of samples that e_stft will jump ahead for every time slice.
        By default, this is equal to half of self.window_length and will update when you
        change self.window_length to stay at self.window_length // 2. If you set self.hop_length directly
        then self.hop_length and self.window_length are unlinked.
        This property is settable.
        """
        self._hop_length_needs_update = False
        self._hop_length = value

    @property
    def n_fft_bins(self):
        """

        Returns:

        """
        return self._n_fft_bins

    @n_fft_bins.setter
    def n_fft_bins(self, value):
        """
        Number of fft bins per time slice in the stft. A time slice is of length window length.
        By default the number of FFT bins is equal to window_length (value of window_length),
        but if this is set manually then e_stft takes a window of length.
        If you give it a value lower than self.window_length, self.window_length will be used.
        This property is settable.

        """
        # TODO: add warning for this
        if value < self.window_length:
            value = self.window_length

        self._n_fft_bins_needs_update = False
        self._n_fft_bins = value

    @property
    def window_overlap(self):
        """
        Returns number of samples of overlap between adjacent time slices.
        This is calculated like self.window_length - self.hop_length
        This property is not settable.
        """
        return self.window_length - self.hop_length

    def to_json(self):
        return json.dumps(self, default=self._to_json_helper)

    def _to_json_helper(self, o):
        if not isinstance(o, StftParams):
            raise TypeError
        d = {'__class__': o.__class__.__name__,
             '__module__': o.__module__}
        d.update(o.__dict__)
        return d

    @staticmethod
    def from_json(json_string):
        return json.loads(json_string, object_hook=StftParams._from_json_helper)

    @staticmethod
    def _from_json_helper(json_dict):
        if '__class__' in json_dict:
            class_name = json_dict.pop('__class__')
            module = json_dict.pop('__module__')
            if class_name != StftParams.__name__ or module != StftParams.__module__:
                raise TypeError
            sr = json_dict['sample_rate']
            s = StftParams(sr)
            for k, v in json_dict.items():
                s.__dict__[k] = v if not isinstance(v, unicode) else v.encode('ascii')
            return s
        else:
            return json_dict

    def __eq__(self, other):
        return all([v == other.__dict__[k] for k, v in self.__dict__.items()])

    def __ne__(self, other):
        return not self == other

