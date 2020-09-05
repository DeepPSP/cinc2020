#!/usr/bin/env python

import os, sys
import joblib
from copy import deepcopy
from typing import Optional, Dict, Tuple, List

import numpy as np
np.set_printoptions(precision=5, suppress=True)
import wfdb
import torch
from torch import nn
from scipy.signal import resample, resample_poly
from easydict import EasyDict as ED

from get_12ECG_features import get_12ECG_features
from models.special_detectors import special_detectors
from models.ecg_crnn import ECG_CRNN
from model_configs.ecg_crnn import ECG_CRNN_CONFIG
from utils.misc import rdheader, ensure_lead_fmt, extend_predictions
from utils.utils_signal import butter_bandpass_filter
from cfg import ModelCfg, TrainCfg

if ModelCfg.torch_dtype.lower() == "double":
    torch.set_default_tensor_type(torch.DoubleTensor)


__all__ = [
    "load_12ECG_model",
    "run_12ECG_classifier",
]


def run_12ECG_classifier(data:np.ndarray, header_data:List[str], loaded_model:Dict[str, nn.Module], verbose:int=0) -> Tuple[List[int], List[float], List[str]]:
    """
    """
    dtype = np.float32 if ModelCfg.torch_dtype == "float" else np.float64

    header = rdheader(header_data)
    raw_data = ensure_lead_fmt(data.copy(), fmt="lead_first")
    baseline = np.array(header.baseline).reshape(raw_data.shape[0], -1)
    adc_gain = np.array(header.adc_gain).reshape(raw_data.shape[0], -1)
    raw_data = (raw_data - baseline) / adc_gain

    freq = header.fs
    if freq != ModelCfg.freq:
        raw_data = resample_poly(raw_data, ModelCfg.freq, freq, axis=1)
        freq = ModelCfg.freq

    dl_data = raw_data.copy()
    if TrainCfg.bandpass is not None:
        # bandpass
        dl_data = butter_bandpass_filter(
            dl_data,
            lowcut=TrainCfg.bandpass[0],
            highcut=TrainCfg.bandpass[1],
            order=5,
            fs=TrainCfg.fs,
        )
    if TrainCfg.normalize_data:
        # normalize
        dl_data = ((data - np.mean(dl_data)) / np.std(dl_data)).astype(dtype)

    final_scores, final_conclusions = [], []

    partial_conclusion = special_detectors(raw_data, freq, sig_fmt="lead_first")
    is_brady = partial_conclusion.is_brady
    is_tachy = partial_conclusion.is_tachy
    is_LAD = partial_conclusion.is_LAD
    is_RAD = partial_conclusion.is_RAD
    is_PR = partial_conclusion.is_PR
    is_LQRSV = partial_conclusion.is_LQRSV

    tmp = np.zeros(shape=(len(ModelCfg.full_classes,)))
    tmp[ModelCfg.full_classes.index('Brady')] = int(is_brady)
    tmp[ModelCfg.full_classes.index('LAD')] = int(is_LAD)
    tmp[ModelCfg.full_classes.index('RAD')] = int(is_RAD)
    tmp[ModelCfg.full_classes.index('PR')] = int(is_PR)
    tmp[ModelCfg.full_classes.index('LQRSV')] = int(is_LQRSV)
    partial_conclusion = tmp

    final_scores.append(partial_conclusion)
    final_conclusions.append(partial_conclusion)
    
    # TODO: DL models to make decisions
    dl_scores = []
    for subset, model in loaded_model.items():
        subset_scores, subset_bin = model.inference(torch.from_numpy(normalized_data))
        # subset_scores = extend_predictions(
        #     subset_scores,
        #     TrainCfg.tranche_classes[subset],
        #     ModelCfg.full_classes,
        # )
        # subset_bin = extend_predictions(
        #     subset_bin,
        #     TrainCfg.tranche_classes[subset],
        #     ModelCfg.full_classes,
        # )
        # scores.append(subset_scores)
        # conclusions.append(subset_bin)
        subset_scores = extend_predictions(
            subset_scores,
            ModelCfg.tranche_classes[subset],
            ModelCfg.dl_classes,
        )
        dl_scores.append(subset_scores)

    if "NSR" in ModelCfg.dl_classes:
        dl_nsr_cid = ModelCfg.dl_classes.index("NSR")
    elif "426783006" in ModelCfg.dl_classes:
        dl_nsr_cid = ModelCfg.dl_classes.index("426783006")
    else:
        dl_nsr_cid = None

    # TODO: make a classifier using the scores from the 4 different dl models
    dl_scores = np.max(np.array(dl_scores), axis=0)
    dl_conclusions = (dl_scores >= ModelCfg.bin_pred_thr).astype(int)

    # treat exceptional cases
    max_prob = dl_scores.max()
    if max_prob < ModelCfg.bin_pred_nsr_thr and dl_nsr_cid is not None:
        dl_conclusions[row_idx, dl_nsr_cid] = 1
    elif dl_conclusions.sum() == 0:
        dl_conclusions = ((dl_scores+ModelCfg.bin_pred_look_again_tol) >= max_prob)
        dl_conclusions = dl_conclusions & (dl_scores >= ModelCfg.bin_pred_nsr_thr)
        dl_conclusions = dl_conclusions.astype(int)

    dl_scores = extend_predictions(
        dl_scores,
        ModelCfg.dl_classes,
        ModelCfg.full_classes,
    )
    dl_conclusions = extend_predictions(
        dl_conclusions,
        ModelCfg.dl_classes,
        ModelCfg.full_classes,
    )

    final_scores.append(dl_scores)
    final_conclusions.append(dl_conclusions)
    final_scores = np.max(final_scores, axis=0)
    final_conclusions = np.max(final_conclusions, axis=0)

    current_label = final_conclusions
    current_score = final_scores
    classes = ModelCfg.full_classes

    return current_label, current_score, classes


def load_12ECG_model(input_directory:Optional[str]=None):
    """
    """
    loaded_model = ED()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    for k in ["AB", "E", "F"]:
        model_config = deepcopy(ECG_CRNN_CONFIG)
        classes = ModelCfg.tranche_classes[k]
        loaded_model[k] = ECG_CRNN(
            classes=classes,
            config=model_config,
        )
        model_weight_path = ModelCfg.tranche_model[k]
        loaded_model[k].load_state_dict(torch.load(model_weight_path, map_location=device))

    loaded_model["all"] = ECG_CRNN(
        classes=ModelCfg.dl_classes,
        config=deepcopy(ECG_CRNN_CONFIG),
    )
    loaded_model["all"].load_state_dict(torch.load(ModelCfg.tranche_model["all"]))

    return loaded_model
