#!/usr/bin/env python

import os, sys
import joblib
from io import StringIO

import numpy as np
import wfdb
import torch
from scipy.signal import resample, resample_poly

from get_12ECG_features import get_12ECG_features
from models.special_detectors import special_detectors
from utils.misc import rdheader, ensure_lead_fmt
from cfg import ModelCfg, TrainCfg


def run_12ECG_classifier(data, header_data, loaded_model):
    """
    """
    classes = TrainCfg.classes + ['Brady', 'LAD', 'RAD', 'PR', 'LQRSV']
    header = rdheader(header_data)
    data = ensure_lead_fmt(data, fmt="lead_first")
    baseline = np.array(header.baseline).reshape(data.shape[0], -1)
    adc_gain = np.array(header.adc_gain).reshape(data.shape[0], -1)
    data = (data - baseline) / adc_gain

    freq = header.fs
    if freq != ModelCfg.freq:
        data = resample_poly(data, ModelCfg.freq, freq)
        freq = ModelCfg.freq

    partial_conclusion = special_detectors(data, freq, sig_fmt="lead_first")
    is_brady = partial_conclusion.is_brady
    is_tachy = partial_conclusion.is_tachy
    is_LAD = partial_conclusion.is_LAD
    is_RAD = partial_conclusion.is_RAD
    is_PR = partial_conclusion.is_PR
    is_LQRSV = partial_conclusion.is_LQRSV

    tmp = np.zeros(shape=(len(classes,)))
    tmp[classes.index('Brady')] = int(is_brady)
    tmp[classes.index('LAD')] = int(is_LAD)
    tmp[classes.index('RAD')] = int(is_RAD)
    tmp[classes.index('PR')] = int(is_PR)
    tmp[classes.index('LQRSV')] = int(is_LQRSV)
    partial_conclusion = tmp

    # normalize
    normalized_data = ((data - np.mean(data)) / np.std(data)).astype(np.float32)
    
    # TODO: DL models to make decisions

    current_score, _ = \
        loaded_model.inference(torch.from_numpy(normalized_data))

    # merge results from DL models with special detectors
    # in a 'max pooling' manner
    current_score = np.max(np.array([partial_conclusion, scalar_pred]), axis=0)

    current_label = np.where(scalar_pred >= TrainCfg.bin_pred_thr)[0].astype(int).tolist()
    if sum(current_label) == 0:
        current_label[np.argmax(current_score)] = 1

    return current_label, current_score, classes


def load_12ECG_model(input_directory):
    # # load the model from disk 
    # f_out='finalized_model.sav'
    # filename = os.path.join(input_directory,f_out)

    # loaded_model = joblib.load(filename)

    return loaded_model
