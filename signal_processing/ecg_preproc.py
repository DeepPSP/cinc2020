"""
preprocess of (single lead) ecg signal:
    band pass (dep. on purpose?) --> remove baseline --> find rpeaks --> wave delineation (?)

References:
-----------
[1] https://github.com/PIA-Group/BioSPPy
[2] to add
"""
import os
import multiprocessing as mp
from copy import deepcopy
from numbers import Real
from typing import Union, Optional, Any, List, Dict, Callable

import numpy as np
from easydict import EasyDict as ED
from scipy.ndimage.filters import median_filter
from scipy.signal.signaltools import resample
# from scipy.signal import medfilt
# https://github.com/scipy/scipy/issues/9680
from biosppy.signals.tools import filter_signal

from .ecg_rpeaks import (
    xqrs_detect, gqrs_detect, pantompkins,
    hamilton_detect, ssf_detect, christov_detect, engzee_detect, gamboa_detect,
)


__all__ = [
    "preprocess_signal",
]


QRS_DETECTORS = {
    "xqrs": xqrs_detect,
    "gqrs": gqrs_detect,
    "pantompkins": pantompkins,
    "hamilton": hamilton_detect,
    "ssf": ssf_detect,
    "christov": christov_detect,
    "engzee": engzee_detect,
    "gamboa": gamboa_detect,
}


def preprocess_signal(raw_sig:np.ndarray, fs:Real, bl_win:Optional[list]=None, band_fs:Optional[list]=None, rpeak_fn:Optional[Callable[[np.ndarray,Real], np.ndarray]]=None) -> Dict[str, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_sig`
    bl_win: list (of 2 real numbers), optional,
        window of baseline removal using `median_filter`
        if is None or empty, baseline removal will not be performed
    band_fs: list (of 2 real numbers), optional,
        frequency band of the bandpass filter,
        if is None or empty, bandpass filtering will not be performed
    rpeak_fn: callable, optional,
        the function detecting rpeaks,
        whose first parameter is the signal, second parameter the sampling frequency

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if 'rpeaks' in `config` is not set
    """
    filtered_ecg = raw_sig.copy()

    # remove baseline
    if baseline:
        window1 = 2 * (bl_win[0] // 2) + 1  # window size must be odd
        window2 = 2 * (bl_win[1] // 2) + 1
        baseline = median_filter(filtered_ecg, size=window1, mode='nearest')
        baseline = median_filter(baseline, size=window2, mode='nearest')
        filtered_ecg = filtered_ecg - baseline
    
    # filter signal
    if band_fs:
        filtered_ecg = filter_signal(
            signal=filtered_ecg,
            ftype='FIR',
            band='bandpass',
            order=int(0.3 * fs),
            sampling_rate=fs,
            frequency=band_fs,
        )['signal']

    if rpeak_fn:
        rpeaks = rpeak_fn(filtered_ecg, fs).astype(int)
    else:
        rpeaks = np.array([], dtype=int)

    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })
    
    return retval
