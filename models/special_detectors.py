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


def electric_axis_detector(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first") -> str:
    """

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write
    """
    raise NotImplementedError
