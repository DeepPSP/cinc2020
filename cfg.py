"""
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "PreprocCfg",
    "TrainCfg",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONE_MINUTE_IN_MS = 60 * 1000

PreprocCfg = ED()
# PreprocCfg.fs = 500
PreprocCfg.leads_ordering = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',]
PreprocCfg.rpeak_mask_radius = 50  # ms
PreprocCfg.rpeak_threshold = 8
PreprocCfg.beat_winL = 250
PreprocCfg.beat_winR = 250

FeatureCfg = ED()
FeatureCfg.leads_ordering = deepcopy(PreprocCfg.leads_ordering)
FeatureCfg.pr_fs_lower_bound = 50  # Hz
FeatureCfg.axis_qrs_mask_radius = 70  # ms
FeatureCfg.axis_method = '2-lead'  # can also be '3-lead'
FeatureCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
FeatureCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm

TrainCfg = ED()
