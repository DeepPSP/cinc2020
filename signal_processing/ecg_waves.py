"""
delineation of T, U, P waves of single lead ECG,
under the assumption that onsets and offset of the QRS complexes are already detected,
mainly relying on (spatial) peak detection algorithms

TODO:
-----
    1. pick out missed R peak (by slope?)
    2. find the second P wave in long RR intervals
    3. filter out incorrectly detected P wave caused by noises (noise should have many local peaks)
    4. filter out incorrectly detected BIPHASIC P wave caused by baseline bias
    5. further check the shape of the P wave, to distinguish normal, RAE, LAE, bi-AE
    6. ...

NOTE:
-----
    1. About T wave:
        1.1. duration of the T Wave is 0.10 to 0.25 seconds or greater
    2. About P wave:
        2.1. duration of a normal P wave is 0.12 to 0.20 seconds.
"""
import time
from typing import Union, Optional, Any, Dict, List, Sequence
from numbers import Real
import warnings

import numpy as np

from utils.utils_signal import smooth, detect_peaks


__all__ = [
    "ecg_tup_wave",
]


##################################
# constants for wave delineation
##################################

EPS = 10  # µV

STANDARD_ECG_AMPLITUDE = 1100  # µV

T_DETECTION_THRESHOLD = int(3.5 * EPS)
P_DETECTION_THRESHOLD = int(2.0 * EPS)
U_DETECTION_THRESHOLD = int(1.6 * EPS)

FLATTENED_T = 100  # 100 µV, i.e. 1mm
FLATTENED_P = 30  # 50 µV, i.e. 0.5mm
FLATTENED_U = FLATTENED_T//5

SLOPE_STEP_T = 16  # ms
P_ONSET_SEARCH_BIAS = 60  # ms
SMOOTH_WINDOW_LEN_MS = 60  # ms

DEFUALT_P_ONSET_OFFSET = 50  # ms
DEFUALT_T_OFFSET = 60  # ms
DEFUALT_T_ONSET = 110  # ms

DEFAULT_U_WIDTH = 100  # ms

DEFAULT_IDX = -9999


def ecg_tup_wave(ecg_curve:np.ndarray, rpeaks:Sequence[int], l_ecg_beats:list, freq:int, verbose:int=0, **kwargs) -> Dict[str,List[int]]:
    """ partly finished,

    detection of T, U, P waves of ECG,
    according to already detected r peaks and qrs onsets, offsets

    Parameters:
    -----------
    to write

    Returns:
    --------
    to write

    Implemented:
    ------------
        "t_peaks" (checked)
        "t_offsets" (checked)
        "t_onsets" (checked)
        "p_peaks" (checked)
        "p_onsets" (checked)
        "p_offsets" (checked)

        "u_peaks" (not checked, to be ameliorated)

    Not implemented:
    ----------------
        "possible_missing_r_peaks"
    """
    analysis_start = time.time()
    warnings.filterwarnings("ignore")

    MS20 = 20 * freq // 1000  # nb_points corr. to 20ms
    MS10 = 10 * freq // 1000  # nb_points corr. to 10ms
    MAX_WINR = int(0.8 * freq)
    MAX_WINL = int(0.5 * freq)

    ret = {
        "t_peaks": [],
        "t_onsets": [],
        "t_offsets": [],
        "p_peaks": [],
        "p_onsets": [],
        "p_offsets": [],
        "u_peaks": [],
        "possible_missing_r_peaks": []
    }

    if verbose >= 1:
        ret["t_separators"] = []
        ret["p_separators"] = []

    # no data
    if len(ecg_curve) * len(rpeaks) * len(l_ecg_beats) == 0:
        return ret

    signal_amplitude = np.max(ecg_curve) - np.min(ecg_curve)
    signal_ratio = np.nanmin([1.0, signal_amplitude/STANDARD_ECG_AMPLITUDE])

    if verbose >= 1:
        print(f'signal_amplitude = {signal_amplitude}, signal_ratio = {signal_ratio}')

    slope_step_t = int(round(SLOPE_STEP_T * freq / 1000))  # number of points
    # slope_step_t = 1
    slope_step_p = max(1, slope_step_t//2)
    p_onset_search_bias = int(round(P_ONSET_SEARCH_BIAS * freq / 1000))

    # parameter used for smoothing the curve within the RR interval
    smooth_window_len = SMOOTH_WINDOW_LEN_MS * freq // 1000

    beat_r_peak_indices = [int(b.r_peak_idx_abs) for b in l_ecg_beats if b.qrs_onset_idx!=DEFAULT_IDX and b.qrs_offset_idx!=DEFAULT_IDX]
    beat_r_peak_indices.sort()

    qrs_amplitudes = [np.max(np.abs(b.beat[b.qrs_onset_idx:b.qrs_offset_idx])) for b in l_ecg_beats if b.qrs_onset_idx!=DEFAULT_IDX and b.qrs_offset_idx!=DEFAULT_IDX]

    # get stats about RR interval
    rr = np.diff(beat_r_peak_indices)
    rr_mean = np.mean(rr)
    rr_std = np.std(rr)
    common_t_search_range = min(MAX_WINR, 3*(rr_mean+2*rr_std)//4)
    common_p_search_range = min(MAX_WINL, (rr_mean+2*rr_std)//2)

    second_p_threshold = min(0.6*rr_mean, 0.6*1200*freq/1000)  # time threshold
    # if distance of two consecutively detected T wave and P wave is greater than second_p_threshold,
    # then the search of the second P wave within this RR interval is to be performed

    # get stats about max abs slope (not standardized as in ecg_beat) of R peaks
    slope_step_r = int(round(SLOPE_STEP * freq / 1000))
    max_abs_slopes = [np.max(np.abs(diff_with_step(b.beat, step=slope_step_r))) for b in l_ecg_beats]
    mean_r_slope = np.mean(max_abs_slopes)

    t_peaks = []  # implemented
    t_onsets = []  # implemented
    t_offsets = []  # implemented
    p_peaks = []  # implemented
    p_onsets = []  # implemented
    p_offsets = []  # implemented
    u_peaks = []  # not implemented
    possible_missing_r_peaks = []

    # additional information
    t_separators = []
    p_separators = []

    if verbose >= 1:
        print('\n\necg_tup_wave starts working...')
        print('the constants are as follows')
        print(f'FLATTENED_T = {FLATTENED_T}, FLATTENED_P = {FLATTENED_P}, FLATTENED_U = {FLATTENED_U}')
        print(f'slope_step_r = {slope_step_r}, slope_step_t = {slope_step_t}, slope_step_p = {slope_step_p}')
        print(f'p_onset_search_bias = {p_onset_search_bias}')
        print(f'MAX_WINL = {MAX_WINL}, MAX_WINR = {MAX_WINR}')
        print(f'mean_r_slope = {mean_r_slope}')

    t_vals, p_vals, u_vals = [], [], []
    qt_intervals, pr_intervals = [], []
    # t_val_mean, p_val_mean, u_val_mean = -1, -1, -1
    t_val_mean, p_val_mean, u_val_mean = np.nan, np.nan, np.nan

    for idx, r in enumerate(beat_r_peak_indices[:-1]):
        if verbose >= 1:
            print(f"{'*'*20}  start detecting t, u, p waves of the {idx}-th beat...  {'*'*20}")
        qrs_offset_idx = l_ecg_beats[idx].qrs_offset_idx + r - l_ecg_beats[idx].r_peak_idx
        next_r = beat_r_peak_indices[idx+1]  # absolute index in ecg_curve
        qrs_onset_idx = l_ecg_beats[idx+1].qrs_onset_idx + next_r - l_ecg_beats[idx+1].r_peak_idx  # absolute index in self.ecg_curve
        # l_point_idx = qrs_offset_idx + l_point_bias  # to be used to filter out false T peaks
 
        if l_ecg_beats[idx].qrs_offset_idx == DEFAULT_IDX or l_ecg_beats[idx+1].qrs_onset_idx == DEFAULT_IDX:
            continue

        # if len(t_peaks) > 0:
        t_val_mean = np.mean([ecg_curve[pos] for pos in t_peaks])
        p_val_mean = np.mean([ecg_curve[pos] for pos in p_peaks])
        u_val_mean = np.mean([ecg_curve[pos] for pos in u_peaks])

        flattened_t = np.nanmin([FLATTENED_T*signal_ratio, t_val_mean/2, qrs_amplitudes[idx]//4])
        flattened_p = np.nanmin([FLATTENED_P*signal_ratio, p_val_mean/2, qrs_amplitudes[idx]//12])
        flattened_u = FLATTENED_U*signal_ratio

        # data_check_start, data_check_end are absolute indices in ecg_curve
        data_check_start, data_check_end = qrs_offset_idx+MS10, qrs_onset_idx-MS10

        if verbose >= 1:
            print(f'in the {idx}-th rr interval, data_check_start = {data_check_start}, data_check_end = {data_check_end}')
            print(f'for the former beat of this rr interval, its relative qrs_offset = {l_ecg_beats[idx].qrs_offset_idx}, relative r_peak_idx = {l_ecg_beats[idx].r_peak_idx}, absolute r_peak_idx_abs = {r}')
            print(f'for the latter beat of this rr interval, its relative qrs_onset = {l_ecg_beats[idx+1].qrs_onset_idx}, relative r_peak_idx = {l_ecg_beats[idx+1].r_peak_idx}, absolute r_peak_idx_abs = {next_r}')

        if data_check_end <= data_check_start:
            if verbose >= 1:
                print(f'data_check_end (={data_check_end}) >= ({data_check_start}=) data_check_start, which is absurd. Might due to noise.')
            continue
        
        data_smoothed = smooth(
            x=ecg_curve[data_check_start:data_check_end],
            window_len=smooth_window_len,
            window='hanning',
            keep_dtype=True
        )

        if len(data_smoothed) == 0:
            if verbose >= 1:
                print('len(data_smoothed) == 0, which is absurd. Might due to noise.')
            continue

        if verbose >= 1:
            print(f'this qrs_offset_idx from this R peak is {qrs_offset_idx-r}')
            print(f'next qrs_onset_idx from this R peak is {qrs_onset_idx-r}')
            print(f'the already detected t_val_mean = {t_val_mean}, p_val_mean = {p_val_mean}, u_val_mean = {u_val_mean}')
            if verbose >= 2:
                print(f'data_smoothed = {data_smoothed}')

        ##########################################################################
        #########    start detecting t peaks and t offset and t onset    #########
        ##########################################################################
        t_separator = int(min(r+common_t_search_range, (data_check_start+2*data_check_end)//3))
        
        nb_iter_t_wave = 0
        while nb_iter_t_wave <= 1:
            t_separators.append(t_separator)
            nb_iter_t_wave += 1

            if verbose >= 1:
                print(f'nb_iter_t_wave = {nb_iter_t_wave}')
                print(f'separator from this R peak for t_data is {t_separator-r}')
            
            t_separator_rel = t_separator-data_check_start
            
            t_data = data_smoothed[:t_separator_rel]
            if verbose >= 2:
                print(f't_data = {t_data}')

            t_wave = delineate_t_wave(
                t_data=t_data,
                freq=freq,
                t_val_mean=t_val_mean,
                flattened_t=flattened_t,
                signal_ratio=signal_ratio,
                verbose=verbose
            )

            if len(t_wave['t_peaks']) == 0:
                # no t peaks found
                # search again by moving t_separator to qrs onset of the next beat
                # t_separator = min(r+common_t_search_range, data_check_end)
                t_separator = int(data_check_end)
            else:
                break

        if len(t_wave['t_peaks']) > 0:
            t_wave['t_peaks'].sort()
            t_peaks += [pos+data_check_start for pos in t_wave['t_peaks']]
            t_vals += [data_smoothed[pos] for pos in t_wave['t_peaks']]
        if t_wave['t_onset'] >= 0:
            t_onsets.append(t_wave['t_onset']+data_check_start)
        if t_wave['t_offset'] >= 0:
            t_offsets.append(t_wave['t_offset']+data_check_start)
            # qt_intervals.append(t_wave['t_offset']+data_check_start-qrs_onset_idx)

        ###########################################################################
        #########    end of detecting t peaks and t offset and t onset    #########
        ###########################################################################

        # if this T peak is very close to the next R peak,
        # and the latter beat is possibly a PVC beat,
        # then skip searching for P wave and U wave within this RR interval
        if nb_iter_t_wave == 2 and len(t_wave['t_peaks']) > 0:
            data_len_left = len(data_smoothed) - t_wave['t_peaks'][-1]
            if data_len_left < 100 * freq // 1000 and l_ecg_beats[idx+1].qrs_width > 130:
                if verbose >= 1:
                    print('this T peak is very close to the next R peak, and the latter beat is possibly a PVC beat,')
                    print('hence searching for P wave and U wave within this RR interval is skipped')
                continue

        #######################################################################
        #########    start detecting p peaks and p onset, p offset    #########
        #######################################################################

        if t_wave['t_offset'] > 0:
            p_separator = int(max(0, next_r-common_p_search_range, min(t_wave['t_offset']+data_check_start+MS10, (data_check_start + 2*data_check_end)//3)))
        else:
            p_separator = int(max(0, next_r-common_p_search_range, (data_check_start + 2*data_check_end)//3))

        p_search_end = data_check_end

        if verbose >= 1:
            print(f'start searching for p wave, with p_separator = {p_separator}, p_search_end = {p_search_end}')

        nb_iter_p_wave = 0
        while nb_iter_p_wave <= 1:
            p_separators.append(p_separator)
            nb_iter_p_wave += 1

            p_separator_rel = p_separator - data_check_start  # rel. index in data_smoothed
            p_search_end_rel = p_search_end - data_check_start

            if verbose >= 1:
                print(f'in the {nb_iter_p_wave}-th iteration of searching for p wave')
                print(f'p_separator_rel = {p_separator_rel}, p_search_end_rel = {p_search_end_rel}')
                print(f'p_separator from next R peak for p_data is {next_r-p_separator}')

            if p_search_end_rel <= p_separator_rel:
                p_wave = {
                    "p_onset": -1,
                    "p_peaks": [],
                    "p_offset": -1
                }
                break

            p_data_ori = data_smoothed[p_separator_rel:p_search_end_rel]
            p_data = p_data_ori - p_data_ori[-1]

            p_wave = delineate_p_wave(
                p_data=p_data,
                freq=freq,
                p_val_mean=p_val_mean,
                flattened_p=flattened_p,
                signal_ratio=signal_ratio,
                verbose=verbose
            )

            if len(p_wave['p_peaks']) == 0 and t_wave['t_offset'] > 0 and p_separator-t_wave['t_offset']-data_check_start >= 4*MS20:
                # no p peaks found, but t wave exists
                # search again by moving searching interval backward
                p_search_end = min(p_separator+4*MS20, p_search_end)
                p_separator = int(t_wave['t_offset']+data_check_start+MS10)
                if verbose >= 1:
                    print(f'the {nb_iter_p_wave-1}-th iteration fails to find p wave, hence we change p_separator to {p_separator} and p_search_end to {p_search_end}')
                if p_search_end <= p_separator:
                    break
            else:
                if verbose >= 1:
                    print(f'the {nb_iter_p_wave-1}-th iteration delineates p wave successfully, with p_wave = {p_wave}')
                break
        
        if len(p_wave['p_peaks']) >= 0:
            p_wave['p_peaks'].sort()
            p_peaks += [pos+p_separator for pos in p_wave['p_peaks']]
            p_vals += [data_smoothed[pos+p_separator_rel] for pos in p_wave['p_peaks']]
        if p_wave['p_onset'] >= 0:
            p_onsets.append(p_wave['p_onset']+p_separator)
        if p_wave['p_offset'] >= 0:
            p_offsets.append(p_wave['p_offset']+p_separator)
            pr_intervals.append(p_wave['p_offset'])
        
        # start detecting the second P wave in the long RR intervals
        dist_t_p_waves = np.nan
        if t_wave['t_offset'] >= 0 and p_wave['p_onset'] >= 0:
            dist_t_p_waves = p_wave['p_onset'] + p_separator - t_wave['t_offset'] - data_check_start
        
        # if not np.isnan(dist_t_p_waves) and dist_t_p_waves > second_p_threshold:
        if dist_t_p_waves > second_p_threshold:
            # TODO: perform the searching for the second P wave
            if verbose >= 1:
                print(f'distance of the current consecutive t, p waves is {dist_t_p_waves}, which is greater than second_p_threshold = {second_p_threshold}, hence start detecting the possible second p wave within this rr interval')
            pass

        ########################################################################
        #########    end of detecting p peaks and p onset, p offset    #########
        ########################################################################

    
        # TODO: camel, u_wave
        #################################################
        #########    start detecting u waves    #########
        #################################################
        # if not (p_wave['p_onset'] > 0 and t_wave['t_offset'] > p_wave['p_onset']+p_separator_rel):
        #     continue

        if t_wave['camel_peak'] > 0:
            u_peaks.append(t_wave['camel_peak']+data_check_start)
            continue
        
        u_start_rel = t_wave['t_offset']
        u_end_rel = p_wave['p_onset']+p_separator_rel
        
        if verbose >= 1:
            print(f"p_wave['p_onset'] = {p_wave['p_onset']}, t_wave['t_offset'] = {t_wave['t_offset']}, p_separator_rel = {p_separator_rel}")
            print(f"u_start_rel = {u_start_rel}, u_end_rel = {u_end_rel}")

        if u_end_rel - u_start_rel < DEFAULT_U_WIDTH*freq//1000:
            continue

        u_data = data_smoothed[u_start_rel: u_end_rel]

        u_wave = delineate_u_wave(
            u_data=u_data,
            freq=freq,
            u_val_mean=u_val_mean,
            flattened_u=flattened_u,
            signal_ratio=signal_ratio,
            verbose=verbose
        )

        if u_wave['u_peak'] >= 0:
            u_peaks.append(u_wave['u_peak'])        

        #####################################################
        #########    end of detection of u waves    #########
        #####################################################


    ret = {
        "t_peaks": sorted(t_peaks),
        "t_onsets": sorted(t_onsets),
        "t_offsets": sorted(t_offsets),
        "p_peaks": sorted(p_peaks),
        "p_onsets": sorted(p_onsets),
        "p_offsets": sorted(p_offsets),
        "u_peaks": sorted(u_peaks),
        "possible_missing_r_peaks": sorted(possible_missing_r_peaks)
    }

    if verbose >= 1:
        ret["t_separators"] = sorted(t_separators)
        ret["p_separators"] = sorted(p_separators)

    if verbose >= 1:
        print(f'ecg_tup_wave spent {time.time()-analysis_start} second(s) on analyzing ecg data of duration {len(ecg_curve)/freq} second(s)')

    return ret


def delineate_t_wave(t_data:np.ndarray, freq:int, t_val_mean:Real, flattened_t:Real, signal_ratio:Real, verbose:int=0, **kwargs) -> dict:
    """ finished, needs enhancement,

    delineate t wave, detecting t onset, t peaks (possibly 2), t offset,
    and possibly an additional camel peak
    """
    if verbose >= 1:
        print('\nstart delineating T wave...')

    ret = {
        "t_onset": -1,
        "t_peaks": [],
        "t_offset": -1,
        "camel_peak": -1
    }
    # constants
    MS20 = 20 * freq // 1000  # nb pts corr. to 20ms
    # MS10 = 10 * freq // 1000  # nb pts corr. to 10ms

    # duration of the T Wave is 0.10 to 0.25 seconds or greater
    t_mpd = 4 * MS20
    t_detection_threshold = np.nanmin([T_DETECTION_THRESHOLD*signal_ratio, 0.5*t_val_mean])

    slope_step_t = int(round(SLOPE_STEP_T * freq / 1000))  # number of points
    # slope_step_t = 1
    l_point_bias = int(round(60 * freq / 1000))  # 60 ms

    # TODO: flattened_p to be reconsidered
    flattened_p = FLATTENED_P

    flattened_t_idx = np.nan

    t_separator_rel = len(t_data)

    t_candidate_positive_peaks = detect_peaks(
        x=t_data,
        mpd=t_mpd,
        threshold=t_detection_threshold,
        valley=False,
        verbose=verbose
    )  # relateive indices
    t_candidate_positive_peaks = [rel_t for rel_t in t_candidate_positive_peaks if rel_t > l_point_bias]

    t_candidate_negative_peaks = detect_peaks(
        x=t_data,
        mpd=t_mpd,
        threshold=t_detection_threshold,
        valley=True,
        verbose=verbose
    )  # relateive indices
    t_candidate_negative_peaks = [rel_t for rel_t in t_candidate_negative_peaks if rel_t > l_point_bias]

    if verbose >= 1:
        print(f"t_candidate_positive_peaks = {t_candidate_positive_peaks}, t_candidate_negative_peaks = {t_candidate_negative_peaks}")

    if len(t_candidate_positive_peaks) > 0:
        t_positive_peak_idx = t_candidate_positive_peaks[np.argmax([t_data[rel_t] for rel_t in t_candidate_positive_peaks])]
        t_positive_peak_val = t_data[t_positive_peak_idx]
        # t_positive_peak_idx += t_start_rel
    else:
        t_positive_peak_idx, t_positive_peak_val = np.nan, np.nan
    
    if len(t_candidate_negative_peaks) > 0:
        t_negative_peak_idx = t_candidate_negative_peaks[np.argmin([t_data[rel_t] for rel_t in t_candidate_negative_peaks])]
        t_negative_peak_val = t_data[t_negative_peak_idx]
        # t_negative_peak_idx += t_start_rel
    else:
        t_negative_peak_idx, t_negative_peak_val = np.nan, np.nan
    
    flattened_t_candidates = []
    if t_positive_peak_idx > 0 and abs(t_positive_peak_val) < flattened_t:
        # flattened_t_idx = t_positive_peak_idx
        # t_positive_peak_idx = -1
        flattened_t_candidates.append([t_positive_peak_idx, t_positive_peak_val])
        t_positive_peak_idx, t_positive_peak_val = np.nan, np.nan
    if t_negative_peak_idx > 0 and abs(t_negative_peak_val) < flattened_t:
        # t_negative_peak_idx = -1
        flattened_t_candidates.append([t_negative_peak_idx, t_negative_peak_val])
        t_negative_peak_idx, t_negative_peak_val = np.nan, np.nan

    if verbose >= 1:
        print(f'flattened_t_candidates = {flattened_t_candidates}')

    if np.isnan(t_positive_peak_idx) and np.isnan(t_negative_peak_idx) and len(flattened_t_candidates)>0:
        flattened_t_idx = max(flattened_t_candidates, key=lambda c:abs(c[1]))[0]

    if verbose >= 1:
        print(f"t_positive_peak_idx = {t_positive_peak_idx}, t_positive_peak_val = {t_positive_peak_val}")
        print(f"t_negative_peak_idx = {t_negative_peak_idx}, t_negative_peak_val = {t_negative_peak_val}")

    t_onset = -1
    t_offset = -1

    # NOTE: the following block searching for onset and offset of t wave
    # might be redundant, try simplying it latter!
    if not np.isnan(t_positive_peak_idx) and np.isnan(t_negative_peak_idx):
        # normal T wave
        # further check if is 'camel', i.e. two concecutive T peaks (or T wave followed by P wave)
        # t_peaks.append(t_positive_peak_idx+data_check_start)
        ret['t_peaks'].append(t_positive_peak_idx)
        peak_vals = np.array([t_data[p] for p in t_candidate_positive_peaks])
        if len(peak_vals) > 1:
            ind = np.argpartition(peak_vals, -2)[-2:]
            ind = ind[np.argsort(peak_vals[ind])]
            second_t_candidate = t_candidate_positive_peaks[ind[-2]]
        else:
            second_t_candidate = np.nan
        
        if verbose >= 1:
            print(f"second_t_candidate = {second_t_candidate}")
        
        if second_t_candidate > t_positive_peak_idx:
            camel_flag = (t_data[t_positive_peak_idx: second_t_candidate+1] > flattened_p).all()
            fused_flag = (t_data[t_positive_peak_idx: second_t_candidate+1] > flattened_t).all()
        elif t_positive_peak_idx > second_t_candidate:
            camel_flag = (t_data[second_t_candidate: t_positive_peak_idx+1] > flattened_p).all()
            fused_flag = (t_data[second_t_candidate: t_positive_peak_idx+1] > flattened_t).all()
        else:  # second_t_candidate == np.nan
            camel_flag = False
            fused_flag = False
        
        if verbose >= 1:
            print(f"camel_flag = {camel_flag}, fused_flag = {fused_flag}")
        
        if camel_flag:
            if second_t_candidate > t_positive_peak_idx:
                # in this case, the second_t_candidate is probably a U wave or a P wave
                ret['camel_peak'] = second_t_candidate
                # t_peaks.append(second_t_candidate)
            else:
                # in this case, the second_t_candidate is probably noise
                # t_peaks.append(second_t_candidate)
                pass
        
        i = t_positive_peak_idx + 1
        next_zero_point = t_separator_rel
        while i < t_separator_rel-1:
            if (t_data[i] - zero_threshold_muV) * (t_data[i+1] - zero_threshold_muV) < 0:  # 'sign' changed
                next_zero_point = i
                break
            i += 1
        if fused_flag:
            dx_start = max(t_positive_peak_idx, second_t_candidate)
        else:
            dx_start = t_positive_peak_idx
        dx = diff_with_step(t_data[dx_start:next_zero_point], step=slope_step_t) / slope_step_t
        # find the 'critical point' with the laregest negative slope
        if len(dx) > 0:
            critical_x = np.argmin(dx)
            critical_slope = dx[critical_x]
            critical_x += dx_start
            critical_y = t_data[critical_x]
            # t_offset = intercept at the x axis of the tangent line at the critical point
            if critical_slope < 0:
                critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                t_offset = min(t_separator_rel, critical_intercept_x)
            if verbose >= 2:
                print(f'in searching for t_offset, critical_slope = {critical_slope}')
        else:
            t_offset = next_zero_point
            if verbose >= 2:
                print('when searching for t_offset, len(dx) is 0!')
        if 0 < t_offset < t_separator_rel:
            ret['t_offset'] = t_offset
    elif t_positive_peak_idx > 0 and t_negative_peak_idx > 0:
        # biphasic
        # find where self.ecg_curve 'return to' zero value
        next_zero_point = t_separator_rel
        if t_positive_peak_idx > t_negative_peak_idx:
            ret['t_peaks'].append(t_negative_peak_idx)
            ret['t_peaks'].append(t_positive_peak_idx)
            i = t_positive_peak_idx + 1
            while i < t_separator_rel-1:
                if (t_data[i] - zero_threshold_muV) * (t_data[i+1] - zero_threshold_muV) < 0:  # 'sign' changed
                    next_zero_point = i
                    break
                i += 1
            # dx = np.diff(t_data[t_positive_peak_idx:next_zero_point], n=slope_step_t) / slope_step_t
            dx = diff_with_step(t_data[t_positive_peak_idx:next_zero_point], step=slope_step_t) / slope_step_t
            # find the 'critical point' with the laregest negative slope
            if len(dx) > 0:
                critical_x = np.argmin(dx)
                critical_slope = dx[critical_x]
                critical_x += t_positive_peak_idx
                critical_y = t_data[critical_x]
                if critical_slope < 0:
                    # t_offset = intercept at the x axis of the tangent line at the critical point
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    t_offset = min(t_separator_rel, critical_intercept_x)
                    if verbose >= 2:
                        print(f'in searching for t_offset, critical_slope = {critical_slope}')
            else:
                t_offset = next_zero_point
                if verbose >= 2:
                    print('when searching for t_offset, len(dx) is 0!')
        else:  # t_positive_peak_idx < t_negative_peak_idx
            ret['t_peaks'].append(t_positive_peak_idx)
            ret['t_peaks'].append(t_negative_peak_idx)
            i = t_negative_peak_idx + 1
            while i < t_separator_rel-1:
                if (t_data[i] + zero_threshold_muV) * (t_data[i+1] + zero_threshold_muV) < 0:  # 'sign' changed
                    next_zero_point = i
                    break
                i += 1
            # dx = np.diff(t_data[t_negative_peak_idx: next_zero_point], n=slope_step_t) / slope_step_t
            dx = diff_with_step(t_data[t_negative_peak_idx: next_zero_point], step=slope_step_t) / slope_step_t
            # find the 'critical point' with the laregest slope
            if len(dx) > 0:
                critical_x = np.argmax(dx)
                critical_slope = dx[critical_x]
                critical_x += t_negative_peak_idx
                critical_y = t_data[critical_x]
                if critical_slope > 0:
                    # t_offset = intercept at the x axis of the tangent line at the critical point
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    t_offset = min(t_separator_rel, critical_intercept_x)
                    if verbose >= 2:
                        print(f'in searching for t_offset, critical_slope = {critical_slope}')
            else:
                t_offset = next_zero_point
                if verbose >= 2:
                    print('when searching for t_offset, len(dx) is 0!')
        if 0 < t_offset < t_separator_rel:
            ret['t_offset'] = t_offset
    elif np.isnan(t_positive_peak_idx) and not np.isnan(t_negative_peak_idx):
        # inverted
        # find where ecg_curve 'return to' zero value
        ret['t_peaks'].append(t_negative_peak_idx)
        next_zero_point = t_separator_rel
        i = t_negative_peak_idx + 1
        while i < t_separator_rel-1:
            if (t_data[i] + zero_threshold_muV) * (t_data[i+1] + zero_threshold_muV) < 0:  # 'sign' changed
                next_zero_point = i
                break
            i += 1
        # dx = np.diff(t_data[t_negative_peak_idx:next_zero_point], n=slope_step_t) / slope_step_t
        dx = diff_with_step(t_data[t_negative_peak_idx:next_zero_point], step=slope_step_t) / slope_step_t
        # find the 'critical point' with the laregest slope
        if len(dx) > 0:
            critical_x = np.argmax(dx)
            critical_slope = dx[critical_x]
            critical_x += t_negative_peak_idx
            critical_y = t_data[critical_x]
            # t_offset = intercept at the x axis of the tangent line at the critical point
            if critical_slope > 0:
                critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                t_offset = min(t_separator_rel, critical_intercept_x)
            if verbose >= 2:
                print(f'in searching for t_offset, critical_slope = {critical_slope}')
        else:
            t_offset = next_zero_point
            if verbose >= 2:
                print('when searching for t_offset, len(dx) is 0!')
        if 0 < t_offset < t_separator_rel:
            ret['t_offset'] = t_offset
    elif not np.isnan(flattened_t_idx):  # flattened
        # flattend T waves should be further treated?
        # pass
        ret['t_peaks'].append(flattened_t_idx)
        i = flattened_t_idx + 1
        next_zero_point = t_separator_rel
        while i < t_separator_rel-1:
            if (t_data[i] - zero_threshold_muV) * (t_data[i+1] - zero_threshold_muV) < 0:  # 'sign' changed
                next_zero_point = i
                break
            i += 1
        dx = diff_with_step(t_data[flattened_t_idx:next_zero_point], step=slope_step_t) / slope_step_t
        negative_flag = True if t_data[flattened_t_idx] < 0 else False
        # find the 'critical point' with the laregest negative or positive slope
        if len(dx) > 0:
            critical_x = np.argmax(dx) if negative_flag else np.argmin(dx)
            critical_slope = dx[critical_x]
            critical_x += flattened_t_idx
            critical_y = t_data[critical_x]
            # t_offset = intercept at the x axis of the tangent line at the critical point
            if (not negative_flag and critical_slope < 0) or (negative_flag and critical_slope > 0):
                critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                t_offset = min(t_separator_rel, critical_intercept_x)
            if verbose >= 2:
                print(f'in searching for t_offset, critical_slope = {critical_slope}')
        else:
            t_offset = next_zero_point
            if verbose >= 2:
                print('when searching for t_offset, len(dx) is 0!')
        if 0 < t_offset < t_separator_rel:
            ret['t_offset'] = t_offset

    if verbose >= 1:
        print(f't_offset = {t_offset}')

    # detecting t_onset
    if t_negative_peak_idx > 0 and t_positive_peak_idx > 0:
        t_onset_search_end = min(t_positive_peak_idx, t_positive_peak_idx)-MS20
    elif t_negative_peak_idx > 0:
        t_onset_search_end = t_negative_peak_idx-MS20
    elif t_positive_peak_idx > 0:
        t_onset_search_end = t_positive_peak_idx-MS20
    else:
        t_onset_search_end = np.nan
    
    if not np.isnan(t_onset_search_end):
        t_val = np.max(np.abs(t_data))
        # change t_onset_search_end if possible, to make if more precise
        tmp = np.where(np.abs(t_data[:t_onset_search_end]-t_data[0])<0.2*t_val)[0]
        if len(tmp) <= 1:
            tmp = np.where(np.abs(t_data[:t_onset_search_end]-t_data[0])>=0.2*t_val)[0]
            if len(tmp) > 1:
                t_onset_search_end = tmp[0]
        else:
            t_onset_search_end = tmp[-1]
        t_onset_search_data = t_data[:t_onset_search_end]
        dx = np.diff(t_onset_search_data)
        if len(dx) > 0:
            t_onset_candidate1 = np.argmin(np.abs(dx))+1
            tmp = np.where((dx<0.02*t_val) & (np.cumsum(dx)<0.2*t_val))[0]
            t_onset_candidate2 = tmp[-1]+1 if len(tmp) > 0 else np.nan
            if verbose >= 1:
                print(f't_onset_candidate1 = {t_onset_candidate1}, t_onset_candidate2 = {t_onset_candidate2}')
            t_onset = max(t_onset_candidate1, t_onset_candidate2) if not np.isnan(t_onset_candidate2) else t_onset_candidate1
        else:
            t_onset = 0
    else:
        t_onset = 0
    
    ret['t_onset'] = t_onset

    if verbose >= 1:
        print(f't_onset = {t_onset}')

        print(f'by the end of delineation of T wave, the result is {ret}')

    return ret


def delineate_p_wave(p_data:np.ndarray, freq:int, p_val_mean:Real, flattened_p:Real, signal_ratio:Real, verbose:int=0, **kwargs) -> dict:
    """ finished, needs enhancement,

    delineate p wave, detecting p onset, p peaks (possibly 2), p offset
    """
    if verbose >= 1:
        print('\nstart delineating P wave...')

    ret = {
        "p_onset": -1,
        "p_peaks": [],
        "p_offset": -1
    }

    MS20 = 20*freq//1000  # nb_points corr. to 20ms
    # MS10 = 10*freq//1000  # nb pts corr. to 10ms

    # normal duration of p_wave 0.12-0.20 seconds
    p_mpd = 50*freq//1000  # nb_points corr. to 50ms
    p_detection_threshold = np.nanmin([P_DETECTION_THRESHOLD*signal_ratio, 0.5*p_val_mean])

    slope_step_p = max(1, int(round(SLOPE_STEP_T * freq / 2000)))  # number of points
    p_onset_search_bias = int(round(P_ONSET_SEARCH_BIAS * freq / 1000))

    # default_p_onset_offset = 3*MS20
    default_p_onset_offset = DEFUALT_P_ONSET_OFFSET*freq//1000

    p_candidate_positive_peaks = detect_peaks(
        x=p_data,
        mpd=p_mpd,
        # mpb=int(1.5*MS20),
        threshold=p_detection_threshold,
        valley=False,
        verbose=verbose
    )  # relateive indices
    p_candidate_negative_peaks = detect_peaks(
        x=p_data,
        mpd=p_mpd,
        # mpb=int(1.5*MS20),
        threshold=p_detection_threshold,
        valley=True,
        verbose=verbose
    )  # relateive indices

    if verbose >= 1:
        print(f"p_candidate_positive_peaks = {p_candidate_positive_peaks}, p_candidate_negative_peaks = {p_candidate_negative_peaks}")

    p_start_negative_flag = False
    p_end_negative_flag = False

    # if non negative, p_positive_peak_idx and p_negative_peak_idx will be indices in p_data
    if len(p_candidate_positive_peaks) > 0:
        p_positive_peak_idx = p_candidate_positive_peaks[np.argmax([p_data[rel_p] for rel_p in p_candidate_positive_peaks])]
        p_positive_peak_val = p_data[p_positive_peak_idx]
    else:
        p_positive_peak_idx, p_positive_peak_val = np.nan, np.nan
    if len(p_candidate_negative_peaks) > 0:
        p_negative_peak_idx = p_candidate_negative_peaks[np.argmin([p_data[rel_p] for rel_p in p_candidate_negative_peaks])]
        p_negative_peak_val = p_data[p_negative_peak_idx]
    else:
        p_negative_peak_idx, p_negative_peak_val = np.nan, np.nan

    if p_positive_peak_val < flattened_p:
        p_positive_peak_idx, p_positive_peak_val = np.nan, np.nan
    if -p_negative_peak_val < flattened_p:
        p_negative_peak_idx, p_negative_peak_val = np.nan, np.nan

    if verbose >= 1:
        print(f"p_positive_peak_idx = {p_positive_peak_idx}, p_positive_peak_val = {p_positive_peak_val}")
        print(f"p_negative_peak_idx = {p_negative_peak_idx}, p_negative_peak_val = {p_negative_peak_val}")

    # if non negative, p_positive_peak_idx and p_negative_peak_idx are indices in p_data
    if not np.isnan(p_positive_peak_idx) and np.isnan(p_negative_peak_idx):
        # normal P wave
        # TODO: further check the shape of the P wave, to distinguish normal, RAE, LAE, bi-AE
        ret['p_peaks'].append(p_positive_peak_idx)

        # ranges for searching for p_onset and p_offset
        # _rel refer to relative index in p_data or p_data_ori
        onset_search_end_rel = p_positive_peak_idx
        offset_search_start_rel = p_positive_peak_idx
    elif not np.isnan(p_positive_peak_idx) and not np.isnan(p_negative_peak_idx):
        # biphasic
        ret['p_peaks'] += sorted([p_positive_peak_idx, p_negative_peak_idx])
        if p_positive_peak_idx > p_negative_peak_idx:
            p_start_negative_flag = True
        else:
            p_end_negative_flag = True

        onset_search_end_rel = ret['p_peaks'][-2]
        offset_search_start_rel = ret['p_peaks'][-1]
    elif np.isnan(p_positive_peak_idx) and not np.isnan(p_negative_peak_idx):
        # inverted
        ret['p_peaks'].append(p_negative_peak_idx)
        p_start_negative_flag = True
        p_end_negative_flag = True

        onset_search_end_rel = p_negative_peak_idx
        offset_search_start_rel = p_negative_peak_idx
    else:  # no significant P wave
        pass

    if len(ret['p_peaks']) == 0:
        return ret

    # start detecting p_onset and p_offset
    p_onset = -1
    p_offset = -1

    if not np.isnan(p_positive_peak_idx) or not np.isnan(p_negative_peak_idx):
        onset_search_start_rel = max(0, onset_search_end_rel-p_onset_search_bias)
        if verbose >= 2:
            print(f'onset_search_start = {onset_search_start_rel}, onset_search_end = {onset_search_end_rel}')
            print(f'offset_search_start = {offset_search_start_rel}')
        # searching for p_onset
        onset_search_data = p_data[onset_search_start_rel: onset_search_end_rel]
        dx = diff_with_step(onset_search_data, step=slope_step_p) / slope_step_p
        if len(dx) > 0:
            p_onset_candidate1 = np.nan
            if p_start_negative_flag:
                critical_x = np.argmin(dx)
                critical_slope = dx[critical_x]
                # critical_x = the relative index in p_data_ori
                critical_x += onset_search_start_rel
                critical_y = p_data[critical_x]
                critical_y = critical_y - np.percentile(p_data[:onset_search_end_rel],10)
                if critical_slope < 0:
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    p_onset_candidate1 = critical_intercept_x
            else:
                critical_x = np.argmax(dx)
                critical_slope = dx[critical_x]
                # critical_x = the relative index in p_data_ori
                critical_x += onset_search_start_rel
                critical_y = p_data[critical_x]
                critical_y = critical_y - np.percentile(p_data[:onset_search_end_rel],10)
                if critical_slope > 0:
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    p_onset_candidate1 = critical_intercept_x
            if verbose >= 2:
                print(f'in the searching for p_onset_candidate1, critical_slope = {critical_slope}')
        else:
            p_onset_candidate1 = np.nan
            # TODO: this case might could be finer analyzed

        if p_start_negative_flag:
            p_val = -np.min(p_data) + np.max(onset_search_data)
            dx = np.diff(-onset_search_data)
            tmp = np.where((dx<0.05*p_val) & (np.cumsum(dx)<0.2*p_val))[0]
            p_onset_candidate2 = tmp[-1]+1 if len(tmp) > 0 else np.nan
            p_onset_candidate2 += onset_search_start_rel
        else:
            p_val = np.max(p_data) - np.min(onset_search_data)
            dx = np.diff(onset_search_data)
            tmp = np.where((dx<0.05*p_val) & (np.cumsum(dx)<0.2*p_val))[0]
            p_onset_candidate2 = tmp[-1]+1 if len(tmp) > 0 else np.nan
            p_onset_candidate2 += onset_search_start_rel

        if verbose >= 1:
            print(f'p_onset_candidate1 = {p_onset_candidate1}, p_onset_candidate2 = {p_onset_candidate2}')

        # p_onset = np.nanmax([p_onset_candidate1, p_onset_candidate2]) if p_onset_candidate2>0 else p_onset_candidate1  # TODO
        p_onset = int(np.nanmax([p_onset_candidate1, p_onset_candidate2]))
        if np.isnan(p_onset):
            p_onset = max(0, ret['p_peaks'][0] - default_p_onset_offset)

        ret['p_onset'] = p_onset
        
        # searching for p_offset
        dx = diff_with_step(p_data[offset_search_start_rel:], step=slope_step_p) / slope_step_p
        if len(dx) > 0:
            if p_end_negative_flag:
                critical_x = np.argmax(dx)
                critical_slope = dx[critical_x]
                # critical_x = the relative index in p_data
                critical_x += offset_search_start_rel
                critical_y = p_data[critical_x] - 0.2*p_data[offset_search_start_rel]  # allow for some error
                if critical_slope > 0:
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    p_offset = critical_intercept_x
            else:
                critical_x = np.argmin(dx)
                critical_slope = dx[critical_x]
                # critical_x = the relative index in p_data
                critical_x += offset_search_start_rel
                critical_y = p_data[critical_x] - 0.2*p_data[offset_search_start_rel]  # allow for some error
                if critical_slope < 0:
                    critical_intercept_x = int(round(critical_x - critical_y/critical_slope))
                    p_offset = critical_intercept_x
            if verbose >= 2:
                print(f'in searching for p_offset, critical_slope = {critical_slope}')
        else:
            pass
            # TODO: this case might could be finer analyzed
        
        if p_offset < 0:
            p_offset = min(len(p_data)-1, ret['p_peaks'][-1] + default_p_onset_offset)

        ret['p_offset'] = p_offset

    if verbose >= 1:
        print(f'by the end of delineation of P wave, the result is {ret}')
    
    return ret


def delineate_u_wave(u_data:np.ndarray, freq:int, u_val_mean:Real, flattened_u:Real, signal_ratio:Real, verbose:int=0, **kwargs) -> dict:
    """ NOT finished,

    delineate U wave, the peak and onset, offset
    """
    if verbose >= 1:
        print('\nstart delineating U wave...')

    ret = {
        'u_onset': -1,
        'u_peak': -1,
        'u_offset': -1
    }

    MS20 = 20 * freq // 1000  # nb_points corr. to 20ms
    u_mpd = 2 * MS20

    u_candidate_positive_peaks = detect_peaks(
        x=u_data,
        mpd=u_mpd,
        # mpb=MS20,
        threshold=U_DETECTION_THRESHOLD,
        valley=False,
        verbose=verbose
    )  # relateive indices
    u_candidate_negative_peaks = detect_peaks(
        x=u_data,
        mpd=u_mpd,
        # mpb=MS20,
        threshold=U_DETECTION_THRESHOLD,
        valley=True,
        verbose=verbose
    )  # relateive indices

    if verbose >= 1:
        print(f"u_candidate_positive_peaks = {u_candidate_positive_peaks}, u_candidate_negative_peaks={u_candidate_negative_peaks}")

    u_peak = -1
    # if non negative, u_positive_peak_idx and u_negative_peak_idx will be absolute indices
    if len(u_candidate_positive_peaks) > 0:
        u_positive_peak_idx = u_candidate_positive_peaks[np.argmax([u_data[rel_u] for rel_u in u_candidate_positive_peaks])]
        u_positive_peak_val = u_data[u_positive_peak_idx]
    else:
        u_positive_peak_idx, u_positive_peak_val = np.nan, np.nan
    if len(u_candidate_negative_peaks) > 0:
        u_negative_peak_idx = u_candidate_negative_peaks[np.argmin([u_data[rel_u] for rel_u in u_candidate_negative_peaks])]
        u_negative_peak_val = u_data[u_negative_peak_idx]
    else:
        u_negative_peak_idx, u_negative_peak_val = np.nan, np.nan

    if u_positive_peak_val < flattened_u:
        u_positive_peak_idx, u_positive_peak_val = np.nan, np.nan
    if -u_negative_peak_val < flattened_u:
        u_negative_peak_idx, u_negative_peak_val = np.nan, np.nan

    if verbose >= 1:
        print(f"u_positive_peak_idx = {u_positive_peak_idx}, u_positive_peak_val = {u_positive_peak_val}")
        print(f"u_negative_peak_idx = {u_negative_peak_idx}, u_negative_peak_val = {u_negative_peak_val}")

    if not np.isnan(u_positive_peak_idx) and np.isnan(u_negative_peak_idx):
        # positive U wave
        u_peak = u_positive_peak_idx
    elif np.isnan(u_positive_peak_idx) and not np.isnan(u_negative_peak_idx):
        # positive U wave
        u_peak = u_negative_peak_idx
    elif not (u_positive_peak_idx) and not np.isnan(u_negative_peak_idx):
        # biphasic ?
        # currently, choose a more significant one
        if u_positive_peak_val >= -u_negative_peak_val:
            u_peak = u_positive_peak_idx
        else:
            u_peak = u_negative_peak_idx
    else:  # no significant U wave
        pass

    if verbose >= 1:
        print(f'u_peak = {u_peak}')

    ret['u_peak'] = u_peak

    if verbose >= 1:
        print('by the end of delineation of U wave, the result is {ret}')

    return ret


def is_truly_biphasic(to_check_data:np.ndarray) -> bool:
    """

    """
    return True


def tp_interval_has_another_wave(to_check_data:np.ndarray, threshold_val:Real, threshold_pos:Real) -> bool:
    """

    check if there is another wave (normal, inverted, or biphasic) within tp interval
    """
    positive_wave_pos = np.where(np.array(to_check_data)>threshold_val)[0].tolist()
    negative_wave_pos = np.where(np.array(to_check_data)<-threshold_val)[0].tolist()
    if any([np.any(np.diff(positive_wave_pos)>1), np.any(np.diff(negative_wave_pos)>1)]):
        return False
    tot_pos = positive_wave_pos + negative_wave_pos
    if max(tot_pos) - min(tot_pos) > threshold_pos:
        return False
    return True
