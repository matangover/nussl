#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import warnings

import numpy as np
import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
jsonpickle_numpy.register_handlers()

from ..core.audio_signal import AudioSignal


class SeparationBase(object):
    """Base class for all separation algorithms in nussl.

    Do not call this. It will not do anything.

    Parameters:
        input_audio_signal (:class:`audio_signal.AudioSignal`). :class:`audio_signal.AudioSignal`
            object. This makes a copy of the provided AudioSignal object.
    """

    # Representations that this algorithm can run on. For more information see
    ALLOWED_TRANSFORMATIONS = {}

    def __init__(self, input_audio_signal, transformation=None):
        if not isinstance(input_audio_signal, AudioSignal):
            raise ValueError('input_audio_signal is not an AudioSignal object!')

        self._audio_signal = None
        self._transformation = None

        if input_audio_signal is not None:
            self.audio_signal = input_audio_signal
        else:
            self.audio_signal = AudioSignal()

        if transformation is None:
            self._transformation = self.audio_signal.transformation

        if not self.audio_signal.has_data:
            warnings.warn('input_audio_signal has no data!')

            # initialize to empty arrays so that we don't crash randomly
            self.audio_signal.audio_data = np.array([])
            self.audio_signal.stft_data = np.array([[]])

    def __str__(self):
        return self.__class__.__name__

    @property
    def sample_rate(self):
        """(int): Sample rate of :attr:`audio_signal`.
        Literally :attr:`audio_signal.sample_rate`.
        """
        return self.audio_signal.sample_rate

    @property
    def stft_params(self):
        """(:class:`spectral_utils.StftParams`): :class:`spectral_utils.StftParams` of :attr:`audio_signal`
        Literally :attr:`audio_signal.stft_params`.
        """
        return self.audio_signal.stft_params

    @property
    def transformation(self):
        """

        Returns:

        """
        return self._transformation

    @transformation.setter
    def transformation(self, value):

        self._transformation = value

    @property
    def audio_signal(self):
        """(:class:`audio_signal.AudioSignal`): Copy of the :class:`audio_signal.AudioSignal` object passed in 
        upon initialization.
        """
        return self._audio_signal

    @audio_signal.setter
    def audio_signal(self, input_audio_signal):
        self._audio_signal = copy.copy(input_audio_signal)

    def plot(self, output_name, **kwargs):
        """Plots relevant data for separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def run(self):
        """Runs separation algorithm

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def make_audio_signals(self):
        """Makes :class:`audio_signal.AudioSignal` objects after separation algorithm is run

        Raises:
            NotImplementedError: Cannot call base class
        """
        raise NotImplementedError('Cannot call base class.')

    def to_json(self):
        """
        Outputs JSON from the data stored in this object.
        
        Returns:
            (str) a JSON string containing all of the information to restore this object exactly as it was when this
            was called.
            
        See Also:
            :func:`from_json` to restore a JSON frozen object.

        """
        return jsonpickle.encode(self)

    @classmethod
    def from_json(cls, json_string):
        """
        Creates a new :class:`SeparationBase` object from the parameters stored in this JSON string.
        
        Args:
            json_string (str): A JSON string containing all the data to create a new :class:`SeparationBase` 
                object.

        Returns:
            (:class:`SeparationBase`) A new :class:`SeparationBase` object from the JSON string.
            
        See Also:
            :func:`to_json` to make a JSON string to freeze this object.

        """
        return jsonpickle.decode(json_string)

    def __call__(self):
        return self.run()

    def __repr__(self):
        return self.__class__.__name__ + ' instance'

    def __eq__(self, other):
        for k, v in self.__dict__.items():
            if isinstance(v, np.ndarray):
                if not np.array_equal(v, other.__dict__[k]):
                    return False
            elif k == 'self':
                pass
            elif v != other.__dict__[k]:
                return False
        return True

    def __ne__(self, other):
        return not self == other
