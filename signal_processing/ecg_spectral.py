"""
spectral analysis on the sequence of rr intervals (hrv, etc.),
and on the ecg signal itself (heart rate, etc.)
"""
from numbers import Real
from typing import Union, Optional, NoReturn

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import scipy.signal as SS

from cfg import FeatureCfg
from utils.utils_signal import resample_irregular_timeseries


__all__ = [
    "spectral_heart_rate",
]


def spectral_heart_rate(filtered_sig:np.ndarray, fs:Real, hr_fs_band:Optional[Sequence[Real]]=None, sig_fmt:str="channel_first", verbose:int=0) -> Real:
    """ finished, NOT checked,

    compute heart rate of a ecg signal using spectral method (from the frequency domain)

    Parameters:
    -----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `filtered_sig`
    hr_fs_band: sequence of real number, optional,
        frequency band (bounds) of heart rate
    sig_fmt: str, default "channel_first",
        format of the multi-lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    verbose: int, default 0,
        print verbosity
    
    Returns:
    --------
    hr: real number,
        heart rate of the ecg signal, with units in bpm
    """
    assert sig_fmt.lower() in ['channel_first', 'lead_first', 'channel_last', 'lead_last']
    if sig_fmt.lower() in ['channel_last', 'lead_last']:
        s = filtered_sig.T
    else:
        s = filtered_sig.copy()
    
    freqs, _, psd = SS.spectrogram(s, fs, axis=-1)
    fs_band = hr_fs_band or FeatureCfg.spectral_hr_fs_band
    assert len(fs_band) >= 2, "frequency band of heart rate should at least has 2 bounds"
    fs_band = sorted(fs_band)
    fs_band = [fs_band[0], fs_band[-1]]

    inds_of_interest = np.where((fs_band[0] <= freqs) & (freqs <= fs_band[1]))[0]
    freqs_of_interest = freqs[inds_of_interest]
    psd_of_interest = psd[...,inds_of_interest]
    peak_inds = np.argmax(psd_of_interest, axis=-1)

    # averaging at a neighborhood of `peak_idx`
    # TODO: ajust for multi-channel cases
    # psd_of_interest[:max(0,peak_idx-1)] = 0
    # psd_of_interest[peak_idx+2:] = 0
    # hr = 60 * np.dot(freqs_of_interest, psd_of_interest) / np.sum(psd_of_interest)
