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


def spectral_heart_rate(filtered_sig:np.ndarray, fs:Real, hr_fs_band:Optional[Sequence[Real]]=None, sig_fmt:str="channel_first", mode:str='hr', verbose:int=0) -> Real:
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
    mode: str, default 'hr',
        mode of computation (return mean heart rate or mean rr intervals),
        can also be 'heart_rate' (alias of 'hr'), and 'rr' (with an alias of 'rr_interval'),
        case insensitive
    verbose: int, default 0,
        print verbosity
    
    Returns:
    --------
    ret_val: real number,
        mean heart rate of the ecg signal, with units in bpm;
        or mean rr intervals, with units in ms
    """
    assert sig_fmt.lower() in ['channel_first', 'lead_first', 'channel_last', 'lead_last']
    if sig_fmt.lower() in ['channel_last', 'lead_last']:
        s = filtered_sig.T
    else:
        s = filtered_sig.copy()
    
    # psd of shape (c,n), freqs of shape (n,)
    # where n = length of signal, c = number of leads
    freqs, _, psd = SS.spectrogram(s, fs, axis=-1)
    fs_band = hr_fs_band or FeatureCfg.spectral_hr_fs_band
    assert len(fs_band) >= 2, "frequency band of heart rate should at least has 2 bounds"
    fs_band = sorted(fs_band)
    fs_band = [fs_band[0], fs_band[-1]]

    inds_of_interest = np.where((fs_band[0] <= freqs) & (freqs <= fs_band[1]))[0]
    # psd_of_interest of shape (c, m), freqs_of_interest of shape (m,)
    # where m = length of inds_of_interest
    freqs_of_interest = freqs[inds_of_interest]
    psd_of_interest = psd[...,inds_of_interest]
    peak_inds = np.argmax(psd_of_interest, axis=-1).flatten()

    if verbose >= 1:
        print(f"freqs_of_interest = {freqs_of_interest.tolist()}")
        print(f"peak_inds = {peak_inds.tolist()}")

    # averaging at a neighborhood of `peak_idx`
    n_nbh = 1
    psd_mask = np.zeros_like(psd_of_interest, dtype=int)
    for l in range(psd_mask.shape[0]):
        psd_mask[l, max(0,peak_inds[l]-n_nbh):min(psd_mask.shape[-1],peak_inds[l]+n_nbh)] = 1
    psd_of_interest = psd_of_interest * psd_mask
    # ret_val with units in second^{-1}
    ret_val = np.mean(np.dot(psd_of_interest, freqs_of_interest) / np.sum(psd_of_interest, axis=-1))
    if mode.lower() in ['hr', 'heart_rate']:
        ret_val = 60 * ret_val
    elif mode.lower() in ['rr', 'rr_interval']:
        ret_val = 1000 / ret_val
    return ret_val
