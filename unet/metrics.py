"""

Reference:
----------
[1] Moskalenko, Viktor, Nikolai Zolotykh, and Grigory Osipov. "Deep Learning for ECG Segmentation." International Conference on Neuroinformatics. Springer, Cham, 2019.

Section 3.2 of ref. [1] describes the metrics

KEY points:
-----------
1. an onset or an offset are detected correctly, if their deviation from the doctor annotations does not exceed in absolute value the tolerance of 150 ms
2. if there is no corresponding significant point (onsets and offset of ECG waveforms P, QRS, T) in the test sample in the neighborhood of ±tolerance of the detected significant point, then the I type error is counted (false positive, FP)
3. if the algorithm does not detect a significant point, then the II type error is counted (false negative, FN)
"""
from numbers import Real
from typing import Union, Optional, Sequence, Dict, Tuple

import numpy as np
from easydict import EasyDict as ED

from .data_reader import ECGWaveForm


__all__ = [
    "compute_metrics",
]


__TOLERANCE = 150  # ms
__WaveNames = ["pwave", "qrs", "twave"]


def compute_metrics(truths:Sequence[ECGWaveForm], preds:Sequence[ECGWaveForm], freq:Real) -> Dict[str, Dict[str, float]]:
    """ finished, checked,

    compute the mean error and standard deviation of the mean errors

    Parameters:
    -----------
    truths: sequence of `ECGWaveForm`s,
        the ground truth
    preds: sequence of `ECGWaveForm`s,
        the predictions

    Returns:
    --------
    scorings: dict,
        with scorings of onsets and offsets of pwaves, qrs complexes, twaves,
        each scoring is a dict consisting of the following metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation
    """
    pwave_onset_truths, pwave_offset_truths, pwave_onset_preds, pwave_offset_preds = \
        [], [], [], []
    qrs_onset_truths, qrs_offset_truths, qrs_onset_preds, qrs_offset_preds = \
        [], [], [], []
    twave_onset_truths, twave_offset_truths, twave_onset_preds, twave_offset_preds = \
        [], [], [], []

    for item in ["truths", "preds"]:
        for w in eval(item):
            for term in ["onset", "offset"]:
                eval(f"{w.name}_{term}_{item}.append(w.{term})")

    scorings = ED()
    for wave in ["pwave", "qrs", "twave",]:
        for term in ["onset", "offset"]:
            sensitivity, precision, f1_score, mean_error, standard_deviation = \
                _compute_metrics(
                    eval(f"{wave}_{term}_truths"),
                    eval(f"{wave}_{term}_preds"),
                    freq
                )
            scorings[f"{wave}_{term}"] = ED(
                sensitivity=sensitivity,
                precision=precision,
                f1_score=f1_score,
                mean_error=mean_error,
                standard_deviation=standard_deviation,
            )
    return scorings


def _compute_metrics(truths:Sequence[Real], preds:Sequence[Real], freq:Real) -> Tuple[float, ...]:
    """ finished, checked,

    Parameters:
    -----------
    truths: sequence of real numbers,
        ground truth of indices of corresponding significant points
    preds: sequence of real numbers,
        predicted indices of corresponding significant points

    Returns:
    --------
    tuple of metrics:
        sensitivity, precision, f1_score, mean_error, standard_deviation
        see ref. [1]
    """
    _tolerance = __TOLERANCE * freq / 1000
    _truths = np.array(truths)
    _preds = np.array(preds)
    truth_positive, false_positive, false_negative = 0, 0, 0
    errors = []
    n_included = 0
    for point in truths:
        _pred = _preds[np.where(np.abs(_preds-point)<=_tolerance)[0].tolist()]
        if len(_pred) > 0:
            truth_positive += 1
            errors.append(_pred[0]-point)
        else:
            false_negative += 1
        n_included += len(_pred)
    
    # false_positive = len(_preds) - n_included
    false_positive = len(_preds) - truth_positive

    # print(f"""
    # truth_positive = {truth_positive}
    # false_positive = {false_positive}
    # false_negative = {false_negative}
    # """)

    # print(f"len(truths) = {len(truths)}, truth_positive + false_negative = {truth_positive + false_negative}")
    # print(f"len(preds) = {len(preds)}, truth_positive + false_positive = {truth_positive + false_positive}")

    sensitivity = truth_positive / (truth_positive + false_negative)
    precision = truth_positive / (truth_positive + false_positive)
    f1_score = 2 * sensitivity * precision / (sensitivity + precision)
    mean_error = np.mean(errors) * 1000 / freq
    standard_deviation = np.std(errors) * 1000 / freq

    return sensitivity, precision, f1_score, mean_error, standard_deviation
