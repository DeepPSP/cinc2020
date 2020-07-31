"""
special detectors using rules,
for (perhaps auxiliarily) detecting PR, Brady (including SB), LQRSV, RAD, LAD, STach

pending arrhythmia classes: LPR, LQT
"""
from numbers import Real
from typing import Union, Optional, Any, List, Dict, Callable, Sequence

import numpy as np
from scipy.signal import peak_prominences, peak_widths
from biosppy.signals.tools import filter_signal

from cfg import FeatureCfg
from signal_processing.ecg_rpeaks import (
    pantompkins,
    xqrs_detect, gqrs_detect,
    hamilton_detect, ssf_detect, christov_detect, engzee_detect, gamboa_detect,
)
from utils.misc import ms2samples


__all__ = [
    "pace_rhythm_detector",
    "electrical_axis_detector",
    "brady_tachy_detector",
    "LQRSV_detector",
]


def pace_rhythm_detector(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first", verbose:int=0) -> bool:
    """

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    if sig_fmt.lower in ['channel_first', 'lead_first']:
        s = raw_sig.copy()
    else:
        s = raw_sig.T
    
    data_hp = np.array([
        filter_signal(
            raw_sig[lead],
            ftype='butter',
            band='highpass',
            order=5,
            frequency=FeatureCfg.pr_fs_lower_bound,
            sampling_rate=fs)['signal'] \
                for lead in range(12)
    ])
    # TODO: making decision using data_hp

    raise NotImplementedError


def electrical_axis_detector(filtered_sig:np.ndarray, rpeaks:np.ndarray, fs:Real, sig_fmt:str="channel_first", method:Optional[str]=None, verbose:int=0) -> str:
    """ finished, checked,

    detector of the heart electrical axis by means of '2-lead' method or '3-lead' method,
    NOTE that the extreme axis is not checked and treated as 'normal'

    Parameters:
    -----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    method: str, optional,
        method for detecting electrical axis, can be '2-lead', '3-lead',
        if not specified, `FeatureCfg.axis_method` will be used
    verbose: int, default 0,
        print verbosity

    Returns:
    --------
    axis: str,
        one of 'normal', 'LAD', 'RAD',
        the heart electrical axis
    """
    decision_method = method or FeatureCfg.axis_method
    decision_method = decision_method.lower()
    assert decision_method in ['2-lead', '3-lead',], f"Method {decision_method} not supported!"

    if sig_fmt.lower() in ['channel_first', 'lead_first']:
        s = filtered_sig.copy()
    else:
        s = filtered_sig.T
    
    lead_I = s[FeatureCfg.leads_ordering.index('I')]
    lead_II = s[FeatureCfg.leads_ordering.index('II')]
    lead_aVF = s[FeatureCfg.leads_ordering.index('aVF')]

    sig_len = s.shape[1]
    radius = ms2samples(FeatureCfg.axis_qrs_mask_radius, fs)
    l_qrs = []
    for r in rpeaks:
        l_qrs.append([max(0,r-radius), min(sig_len-1,r+radius)])

    if verbose >= 1:
        print(f"qrs mask radius = {radius}, sig_len = {sig_len}")
        print(f"l_qrs = {l_qrs}")
    
    # lead I
    lead_I_positive = sum([
        np.max(lead_I[qrs_itv[0]:qrs_itv[1]]) > np.abs(np.min(lead_I[qrs_itv[0]:qrs_itv[1]])) \
            for qrs_itv in l_qrs
    ]) >= len(l_qrs)//2 + 1

    # lead aVF
    lead_aVF_positive = sum([
        np.max(lead_aVF[qrs_itv[0]:qrs_itv[1]]) > np.abs(np.min(lead_aVF[qrs_itv[0]:qrs_itv[1]])) \
            for qrs_itv in l_qrs
    ]) >= len(l_qrs)//2 + 1
    
    # lead II
    lead_II_positive = sum([
        np.max(lead_II[qrs_itv[0]:qrs_itv[1]]) > np.abs(np.min(lead_II[qrs_itv[0]:qrs_itv[1]])) \
            for qrs_itv in l_qrs
    ]) >= len(l_qrs)//2 + 1

    # decision making
    if decision_method == '2-lead':
        if lead_I_positive and not lead_aVF_positive:
            axis = 'LAD'
        elif not lead_I_positive and lead_aVF_positive:
            axis = 'RAD'
        else:
            axis = 'normal'  # might also include extreme axis
    elif decision_method == '3-lead':
        if lead_I_positive and not lead_II_positive and not lead_aVF_positive:
            axis = 'LAD'
        elif not lead_I_positive and lead_aVF_positive:
            axis = 'RAD'
        else:
            axis = 'normal'  # might also include extreme axis

    return axis


def brady_tachy_detector(rpeaks:np.ndarray, fs:Real, normal_rr_range:Optional[Sequence[Real]]=None, verbose:int=0) -> str:
    """ finished, NOT checked,

    detemine if the ecg is bradycadia or tachycardia or normal,
    only by the mean rr interval.

    this detector can be used alone (e.g. for the arrhythmia `Brady`),
    or combined with other detectors (e.g. for the arrhythmia `STach`)

    Parameters:
    -----------
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    normal_rr_range: sequence of int, optional,
        the range of normal rr interval, with units in ms;
        if not given, default values from `FeatureCfg` will be used
    verbose: int, default 0,
        print verbosity

    Returns:
    --------
    conclusion: str,
        one of "T" (tachycardia), "B" (bradycardia), "N" (normal)
    """
    mean_rr = np.mean(np.diff(rpeaks))
    nrr = normal_rr_range or [FeatureCfg.tachy_threshold, FeatureCfg.brady_threshold]
    nrr = sorted(nrr)
    assert len(nrr) >= 2
    nrr = [ms2samples(nrr[0], fs), ms2samples(nrr[-1], fs)]
    if mean_rr < nrr[0]:
        conclusion = "T"
    elif mean_rr > nrr[1]:
        conclusion = "B"
    else:
        conclusion = "N"
    return conclusion


def LQRSV_detector(filtered_sig:np.ndarray, rpeaks:np.ndarray, fs:Real, sig_fmt:str="channel_first", verbose:int=0) -> str:
    """ NOT finished,

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    raise NotImplementedError
