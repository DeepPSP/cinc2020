#!/usr/bin/env python

import os, sys
import joblib
from io import StringIO

import numpy as np
import wfdb
from scipy.signal import resample, resample_poly

from get_12ECG_features import get_12ECG_features
from models.special_detectors import special_detectors
from utils.misc import rdheader, ensure_lead_fmt
from cfg import ModelCfg


def run_12ECG_classifier(data, header_data, loaded_model):
    """
    """
    raise NotImplementedError
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

    # normalize
    normalized_data = (data - np.mean(data)) / np.std(data)
    
    # TODO: DL models to make decisions

    # TODO: join results from DL models and special detectors

    return current_label, current_score, classes


def load_12ECG_model(input_directory):
    # load the model from disk 
    f_out='finalized_model.sav'
    filename = os.path.join(input_directory,f_out)

    loaded_model = joblib.load(filename)

    return loaded_model
