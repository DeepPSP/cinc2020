"""
special detectors using rules,
for (perhaps auxiliarily) detecting PR, Brady (including SB), LQRSV, RAD, LAD, STach

pending arrhythmia classes: LPR, LQT
"""
from numbers import Real
from typing import Union, Optional, Any, List, Dict, Callable

import numpy as np
from biosppy.signals.tools import filter_signal
from .ecg_rpeaks import (
    xqrs_detect, gqrs_detect, pantompkins,
    hamilton_detect, ssf_detect, christov_detect, engzee_detect, gamboa_detect,
)
from cfg import FeatureCfg


def pace_rhythm_detector(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first") -> bool:
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


def electrical_axis_detector(filtered_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first", rpeaks:np.ndarray) -> str:
    """ finished, not checked,

    detector of the heart electrical axis by means of '2-lead' method or '3-lead' method,
    NOTE that the extreme axis is not checked and treated as 'normal'

    Parameters:
    -----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    rpeaks: ndarray,
        array of indices of the R peaks

    Returns:
    --------
    axis: str,
        one of 'normal', 'LAD', 'RAD'
    """
    if sig_fmt.lower in ['channel_first', 'lead_first']:
        s = filtered_sig.copy()
    else:
        s = filtered_sig.T
    
    lead_I = s[FeatureCfg.leads_ordering.index('I')]
    lead_II = s[FeatureCfg.leads_ordering.index('II')]
    lead_aVF = s[FeatureCfg.leads_ordering.index('aVF')]

    sig_len = s.shape[1]
    radius = int(FeatureCfg.axis_qrs_mask_radius * fs / 1000)
    l_qrs = []
    for r in rpeaks:
        l_qrs.append([max(0,r-radius),min(sig_len-1,r+radius)])
    
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
    if FeatureCfg.axis_method == '2-lead':
        if lead_I_positive and not lead_aVF_positive:
            axis = 'LAD'
        elif not lead_I_positive and lead_aVF_positive:
            axis = 'RAD'
        else:
            axis = 'normal'  # might also include extreme axis
    elif FeatureCfg.axis_method == '3-lead':
        if lead_I_positive and not lead_II_positive and not lead_aVF_positive:
            axis = 'LAD'
        elif not lead_I_positive and lead_aVF_positive:
            axis = 'RAD'
        else:
            axis = 'normal'  # might also include extreme axis
    return axis
