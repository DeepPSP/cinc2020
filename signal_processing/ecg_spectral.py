"""
spectral analysis on the sequence of rr intervals (hrv, etc.),
and on the ecg signal itself (heart rate, etc.)
"""
from numbers import Real
from typing import Union, Optional, NoReturn

import numpy as np
import scipy.signal as SS

from utils.misc import resample_irregular_timeseries


__all__ [
    "spectral_heart_rate",
]


def spectral_heart_rate(sig:np.ndarray, fs:Real) -> Real:
    """
    """
    raise NotImplementedError
