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
    "preprocess_single_lead_signal",
    "preprocess_12_lead_signal",
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


def preprocess_12_lead_signal(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first", bl_win:Optional[List[Real]]=None, band_fs:Optional[List[Real]]=None, rpeak_fn:Optional[Callable[[np.ndarray,Real], np.ndarray]]=None) -> Dict[str, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_sig`
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    bl_win: list (of 2 real numbers), optional,
        window (units in second) of baseline removal using `median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is [0.2, 0.6],
        if is None or empty, baseline removal will not be performed
    band_fs: list (of 2 real numbers), optional,
        frequency band of the bandpass filter,
        a typical pair is [0.5, 45],
        be careful when detecting paced rhythm,
        if is None or empty, bandpass filtering will not be performed
    rpeak_fn: callable, optional,
        the function detecting rpeaks,
        whose first parameter is the signal, second parameter the sampling frequency

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if `rpeak_fn` is not given
    """
    assert sig_fmt.lower() in ['channel_first', 'lead_first', 'channel_last', 'lead_last']
    if sig_fmt.lower() in ['channel_last', 'lead_last']:
        filtered_ecg = raw_sig.T
    else:
        filtered_ecg = raw_sig.copy()
    rpeaks_candidates = []
    for lead in range(filtered_ecg.shape[0]):
        filtered_metadata = preprocess_single_lead_signal(
            raw_sig=filtered_ecg[lead,...],
            fs=fs,
            bl_win=bl_win,
            band_fs=band_fs,
            rpeak_fn=rpeak_fn
        )
        filtered_ecg[lead,...] = filtered_metadata["filtered_ecg"]
        rpeaks_candidates.append(filtered_metadata["rpeaks"])
    # TODO: merge rpeaks detected in different leads
    rpeaks = np.array([], dtype=int)
    retval = ED({
        "filtered_ecg": filtered_ecg,
        "rpeaks": rpeaks,
    })
    return retval


def preprocess_single_lead_signal(raw_sig:np.ndarray, fs:Real, bl_win:Optional[List[Real]]=None, band_fs:Optional[List[Real]]=None, rpeak_fn:Optional[Callable[[np.ndarray,Real], np.ndarray]]=None) -> Dict[str, np.ndarray]:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw ecg signal
    fs: real number,
        sampling frequency of `raw_sig`
    bl_win: list (of 2 real numbers), optional,
        window (units in second) of baseline removal using `median_filter`,
        the first is the shorter one, the second the longer one,
        a typical pair is [0.2, 0.6],
        if is None or empty, baseline removal will not be performed
    band_fs: list (of 2 real numbers), optional,
        frequency band of the bandpass filter,
        a typical pair is [0.5, 45],
        be careful when detecting paced rhythm,
        if is None or empty, bandpass filtering will not be performed
    rpeak_fn: callable, optional,
        the function detecting rpeaks,
        whose first parameter is the signal, second parameter the sampling frequency

    Returns:
    --------
    retval: dict,
        with items
        - 'filtered_ecg': the array of the processed ecg signal
        - 'rpeaks': the array of indices of rpeaks; empty if `rpeak_fn` is not given
    """
    filtered_ecg = raw_sig.copy()

    # remove baseline
    if baseline:
        window1 = 2 * (int(bl_win[0] * fs) // 2) + 1  # window size must be odd
        window2 = 2 * (int(bl_win[1] * fs) // 2) + 1
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
