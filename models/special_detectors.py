"""
special detectors using rules,
for (perhaps auxiliarily) detecting PR, Brady (including SB), LQRSV, RAD, LAD, STach

pending arrhythmia classes: LPR, LQT

NOTE:
-----
1. ALL signals are assumed to have units in mV
2. almost all the rules can be found in `utils.ecg_arrhythmia_knowledge`
3. 'PR' is superior to electrical axis deviation, which should be considered in the final decision.
the co-occurrence of 'PR' and 'LAD' is 7; the co-occurrence of 'PR' and 'RAD' is 3, whose probabilities are both relatively low

TODO:
-----
currently all are binary detectors, --> detectors producing a probability?
"""
import multiprocessing as mp
from numbers import Real
from typing import Union, Optional, Any, List, Dict, Callable, Sequence

import numpy as np
from scipy.signal import peak_prominences, peak_widths
from biosppy.signals.tools import filter_signal
from easydict import EasyDict as ED

np.set_printoptions(precision=5, suppress=True)

from cfg import (
    FeatureCfg,
    Standard12Leads, ChestLeads, PrecordialLeads, LimbLeads
)
from signal_processing.ecg_rpeaks import (
    pantompkins,
    xqrs_detect, gqrs_detect,
    hamilton_detect, ssf_detect, christov_detect, engzee_detect, gamboa_detect,
)
from signal_processing.ecg_preproc import (
    preprocess_multi_lead_signal,
    rpeaks_detect_multi_leads,
)
from utils.utils_signal import detect_peaks
from utils.misc import ms2samples, samples2ms, get_mask


__all__ = [
    "special_detectors",
    "pacing_rhythm_detector",
    "electrical_axis_detector",
    "brady_tachy_detector",
    "LQRSV_detector",
]


def special_detectors(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first", verbose:int=0) -> np.ndarray:
    """ finished, checked,

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw 12-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    verbose: int, default 0,
        print verbosity
    """
    preprocess = preprocess_multi_lead_signal(
        raw_sig, fs, sig_fmt, rpeak_fn='xqrs', verbose=verbose
    )
    filtered_sig = preprocess["filtered_ecg"]
    rpeaks = preprocess["rpeaks"]
    is_PR = pacing_rhythm_detector(raw_sig, fs, sig_fmt, verbose)
    axis = electrical_axis_detector(filtered_sig, rpeaks, fs, sig_fmt, method='2-lead', verbose=verbose)
    brady_tachy = brady_tachy_detector(rpeaks, fs, verbose=verbose)
    is_LQRSV = LQRSV_detector(filtered_sig, rpeaks, fs, sig_fmt, verbose)
    is_LAD = (axis=='LAD')
    is_RAD = (axis=='RAD')
    is_brady = (brady_tachy=='B')
    is_tachy = (brady_tachy=='T')
    conclusion = ED(
        is_brady=is_brady, is_tachy=is_tachy,
        is_LAD=is_LAD, is_RAD=is_RAD,
        is_PR=is_PR, is_LQRSV=is_LQRSV,
    )
    return conclusion


def pacing_rhythm_detector(raw_sig:np.ndarray, fs:Real, sig_fmt:str="channel_first", verbose:int=0) -> bool:
    """ finished, checked, to be improved (fine-tuning hyper-parameters in cfg.py),

    Parameters:
    -----------
    raw_sig: ndarray,
        the raw 12-lead ecg signal, with units in mV
    fs: real number,
        sampling frequency of `sig`
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    verbose: int, default 0,
        print verbosity

    Returns:
    --------
    is_PR: bool,
        the ecg signal is of pacing rhythm or not
    """
    if sig_fmt.lower() in ['channel_first', 'lead_first']:
        s = raw_sig.copy()
    else:
        s = raw_sig.T
    
    data_hp = np.array([
        filter_signal(
            s[lead,...],
            ftype='butter',
            band='highpass',
            order=20,
            frequency=FeatureCfg.pr_fs_lower_bound,
            sampling_rate=fs)['signal'] \
                for lead in range(s.shape[0])
    ])
    # cpu_num = max(1, mp.cpu_count()-3)
    # with mp.Pool(processes=cpu_num) as pool:
    #     iterable = [(s[lead,...], 'butter', 'highpass', 20, FeatureCfg.pr_fs_lower_bound, fs) for lead in range(s.shape[0])]
    #     results = pool.starmap(
    #         func=filter_signal,
    #         iterable=iterable,
    #     )
    # data_hp = np.array([item['signal'] for item in results])

    # if the signal is 'PR', then there's only sharp spikes left in data_hp
    # however, 'xqrs' seems unable to pick out these spikes as R peaks

    # potential_spikes = rpeaks_detect_multi_leads(
    #     sig=data_hp,
    #     fs=fs,
    #     sig_fmt='channel_first',
    #     rpeak_fn='xqrs',
    #     verbose=verbose,
    # )

    potential_spikes = []
    sig_len = data_hp.shape[-1]
    for l in range(data_hp.shape[0]):
        lead_hp = np.abs(data_hp[l,...])
        mph = FeatureCfg.pr_spike_mph_ratio * np.sum(lead_hp) / sig_len
        lead_spikes = detect_peaks(
            x=lead_hp,
            mph=mph,
            mpd=ms2samples(FeatureCfg.pr_spike_mpd, fs),
            prominence=FeatureCfg.pr_spike_prominence,
            prominence_wlen=ms2samples(FeatureCfg.pr_spike_prominence_wlen, fs),
            verbose=0,
        )
        if verbose >= 2:
            print(f"for the {l}-th lead, its spike detecting mph = {round(mph, 4)} mV")
            print(f"lead_spikes = {lead_spikes.tolist()}")
            print(f"with prominences = {np.round(peak_prominences(lead_hp, lead_spikes, wlen=ms2samples(FeatureCfg.pr_spike_prominence_wlen, fs))[0], 5).tolist()}")
        potential_spikes.append(lead_spikes)
    
    # make decision using `potential_spikes`
    sig_duration_ms = samples2ms(sig_len, fs)
    lead_has_enough_spikes = [False if len(potential_spikes[l]) ==0 else sig_duration_ms / len(potential_spikes[l]) < FeatureCfg.pr_spike_inv_density_threshold for l in range(data_hp.shape[0])]
    if verbose >= 1:
        print(f"lead_has_enough_spikes = {lead_has_enough_spikes}")
        print(f"leads spikes density (units in ms) = {[len(potential_spikes[l]) / sig_duration_ms for l in range(data_hp.shape[0])]}")
    is_PR = sum(lead_has_enough_spikes) >= FeatureCfg.pr_spike_leads_threshold
    return is_PR


def electrical_axis_detector(filtered_sig:np.ndarray, rpeaks:np.ndarray, fs:Real, sig_fmt:str="channel_first", method:Optional[str]=None, verbose:int=0) -> str:
    """ finished, checked, to be improved (fine-tuning hyper-parameters in cfg.py),

    detector of the heart electrical axis by means of '2-lead' method or '3-lead' method,
    NOTE that the extreme axis is not checked and treated as 'normal'

    Parameters:
    -----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal, with units in mV
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
    assert decision_method in ['2-lead', '3-lead',], f"Method `{decision_method}` not supported!"

    if sig_fmt.lower() in ['channel_first', 'lead_first']:
        s = filtered_sig.copy()
    else:
        s = filtered_sig.T
    
    lead_I = s[Standard12Leads.index('I')]
    lead_II = s[Standard12Leads.index('II')]
    lead_aVF = s[Standard12Leads.index('aVF')]

    if len(rpeaks==0):
        # degenerate case
        # voltage might be too low to detect rpeaks
        lead_I_positive = np.max(lead_I) > np.abs(np.min(lead_I))
        lead_II_positive = np.max(lead_II) > np.abs(np.min(lead_II))
        lead_aVF_positive = np.max(lead_aVF) > np.abs(np.min(lead_aVF))
        # decision making
        if decision_method == '2-lead':
            if lead_I_positive and not lead_aVF_positive:
                axis = 'LAD'
            elif not lead_I_positive and lead_aVF_positive:
                axis = 'RAD'
            else:  # if `rpeaks` is empty, all conditions are False
                axis = 'normal'  # might also include extreme axis
        elif decision_method == '3-lead':
            if lead_I_positive and not lead_II_positive and not lead_aVF_positive:
                axis = 'LAD'
            elif not lead_I_positive and lead_aVF_positive:
                axis = 'RAD'
            else:
                axis = 'normal'  # might also include extreme axis
        return axis

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
        else:  # if `rpeaks` is empty, all conditions are False
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
    """ finished, checked, to be improved (fine-tuning hyper-parameters in cfg.py),

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
    if len(rpeaks) <= 1:
        # unable to make predictions
        # TODO: try using spectral method
        conclusion = "N"
        return conclusion
    rr_intervals = np.diff(rpeaks)
    mean_rr = np.mean(rr_intervals)
    if verbose >= 1:
        if len(rr_intervals) > 0:
            print(f"mean_rr = {round(samples2ms(mean_rr, fs), 1)} ms, with detailed rr_intervals (with units in ms) = {(np.vectorize(lambda item:samples2ms(item, fs))(rr_intervals)).tolist()}")
        else:
            print(f"not enough r peaks for computing rr intervals")
    nrr = normal_rr_range or [FeatureCfg.tachy_threshold, FeatureCfg.brady_threshold]
    nrr = sorted(nrr)
    assert len(nrr) >= 2
    nrr = [ms2samples(nrr[0], fs), ms2samples(nrr[-1], fs)]
    # if mean_rr is nan, then all conditions are False, hence the `else` branch is entered
    if mean_rr < nrr[0]:
        conclusion = "T"
    elif mean_rr > nrr[1]:
        conclusion = "B"
    else:
        conclusion = "N"
    return conclusion


def LQRSV_detector(filtered_sig:np.ndarray, rpeaks:np.ndarray, fs:Real, sig_fmt:str="channel_first", verbose:int=0) -> bool:
    """ finished, checked, to be improved (fine-tuning hyper-parameters in cfg.py),

    Parameters:
    -----------
    filtered_sig: ndarray,
        the filtered 12-lead ecg signal, with units in mV
    rpeaks: ndarray,
        array of indices of the R peaks
    fs: real number,
        sampling frequency of the ecg signal
    sig_fmt: str, default "channel_first",
        format of the 12 lead ecg signal,
        'channel_last' (alias 'lead_last'), or
        'channel_first' (alias 'lead_first', original)
    verbose: int, default 0,
        print verbosity

    Returns:
    --------
    is_LQRSV: bool,
        the ecg signal is of arrhythmia `LQRSV` or not
    """
    if sig_fmt.lower() in ['channel_first', 'lead_first']:
        sig_ampl = np.abs(filtered_sig)
    else:
        sig_ampl = np.abs(filtered_sig.T)
    qrs_mask_radius = ms2samples(FeatureCfg.lqrsv_qrs_mask_radius, fs)
    l_qrs = get_mask(
        shape=sig_ampl.shape,
        critical_points=rpeaks,
        left_bias=qrs_mask_radius,
        right_bias=qrs_mask_radius,
        return_fmt='intervals',
    )
    if verbose >= 2:
        print(f"qrs intervals = {l_qrs}")

    limb_lead_inds = [Standard12Leads.index(l) for l in LimbLeads]
    precordial_lead_inds = [Standard12Leads.index(l) for l in PrecordialLeads]

    l_qrs_limb_leads = []
    l_qrs_precordial_leads = []
    
    if len(l_qrs) == 0:
        # no rpeaks detected
        low_qrs_limb_leads = [np.max(sig_ampl[idx]) <= 0.5 + FeatureCfg.lqrsv_ampl_bias for idx in limb_lead_inds]
        low_qrs_limb_leads = sum(low_qrs_limb_leads) / len(low_qrs_limb_leads)  # to ratio
        low_qrs_precordial_leads = [np.max(sig_ampl[idx]) <= 1 + FeatureCfg.lqrsv_ampl_bias for idx in precordial_lead_inds]
        low_qrs_precordial_leads = sum(low_qrs_precordial_leads) / len(low_qrs_precordial_leads)
    else:
        for itv in l_qrs:
            for idx in limb_lead_inds:
                l_qrs_limb_leads.append(sig_ampl[idx, itv[0]:itv[1]].flatten())
            for idx in precordial_lead_inds:
                l_qrs_precordial_leads.append(sig_ampl[idx, itv[0]:itv[1]].flatten())

        if verbose >= 2:
            print("for limb leads, the qrs amplitudes are as follows:")
            for idx, lead_name in enumerate(LimbLeads):
                print(f"for limb lead {lead_name}, the qrs amplitudes are {[np.max(item) for item in l_qrs_limb_leads[idx*len(l_qrs): (idx+1)*len(l_qrs)]]}")
            for idx, lead_name in enumerate(PrecordialLeads):
                print(f"for precordial lead {lead_name}, the qrs amplitudes are {[np.max(item) for item in l_qrs_limb_leads[idx*len(l_qrs): (idx+1)*len(l_qrs)]]}")

        low_qrs_limb_leads = [np.max(item) <= 0.5 + FeatureCfg.lqrsv_ampl_bias for item in l_qrs_limb_leads]
        low_qrs_limb_leads = sum(low_qrs_limb_leads) / len(low_qrs_limb_leads)  # to ratio
        low_qrs_precordial_leads = [np.max(item) <= 1 + FeatureCfg.lqrsv_ampl_bias for item in l_qrs_precordial_leads]
        low_qrs_precordial_leads = sum(low_qrs_precordial_leads) / len(low_qrs_precordial_leads)

    if verbose >= 2:
        print(f"ratio of low qrs in limb leads = {low_qrs_limb_leads}")
        print(f"ratio of low qrs in precordial leads = {low_qrs_precordial_leads}")

    is_LQRSV = \
        (low_qrs_limb_leads >= FeatureCfg.lqrsv_ratio_threshold) \
        or (low_qrs_precordial_leads >= FeatureCfg.lqrsv_ratio_threshold)

    return is_LQRSV
