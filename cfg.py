"""
configurations for signal preprocess, feature extraction, training, etc.
along with some constants
"""
import os
from copy import deepcopy

from easydict import EasyDict as ED


__all__ = [
    "PreprocCfg",
    "FeatureCfg",
    "TrainCfg",
    "PlotCfg",
    "Standard12Leads",
    "InferiorLeads", "LateralLeads", "SeptalLeads",
    "ChestLeads", "PrecordialLeads", "LimbLeads",
]


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_ONE_MINUTE_IN_MS = 60 * 1000


# names of the 12 leads
Standard12Leads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6',]
InferiorLeads = ['II', 'III', 'aVF',]
LateralLeads = ['I', 'aVL',] + [f'V{i}' for i in range(5,7)]
SeptalLeads = ['aVR', 'V1',]
AnteriorLeads = [f'V{i}' for i in range(2,5)]
ChestLeads = [f'V{i}' for i in range(1, 7)]
PrecordialLeads = ChestLeads
LimbLeads = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',]


# ecg signal preprocessing configurations
PreprocCfg = ED()
# PreprocCfg.fs = 500
PreprocCfg.leads_ordering = deepcopy(Standard12Leads)
PreprocCfg.rpeak_mask_radius = 50  # ms
PreprocCfg.rpeak_num_threshold = 8  # used for merging rpeaks detected from 12 leads
PreprocCfg.beat_winL = 250
PreprocCfg.beat_winR = 250


# feature extracting configurations,
# mainly used by the special detectors
FeatureCfg = ED()
FeatureCfg.leads_ordering = deepcopy(PreprocCfg.leads_ordering)
FeatureCfg.pr_fs_lower_bound = 47  # Hz
FeatureCfg.pr_spike_mph_ratio = 15  # ratio to the average amplitude of the signal
FeatureCfg.pr_spike_mpd = 300  # ms
FeatureCfg.pr_spike_prominence = 0.3
FeatureCfg.pr_spike_prominence_wlen = 120  # ms
FeatureCfg.pr_spike_inv_density_threshold = 2000  # inverse density (1/density), one spike per 2000 ms
FeatureCfg.pr_spike_leads_threshold = 7
FeatureCfg.axis_qrs_mask_radius = 70  # ms
FeatureCfg.axis_method = '2-lead'  # can also be '3-lead'
FeatureCfg.brady_threshold = _ONE_MINUTE_IN_MS / 60  # ms, corr. to 60 bpm
FeatureCfg.tachy_threshold = _ONE_MINUTE_IN_MS / 100  # ms, corr. to 100 bpm
FeatureCfg.lqrsv_qrs_mask_radius = 60  # ms
FeatureCfg.lqrsv_ampl_bias = 0.02  # mV, TODO: should be further determined by resolution, etc.
FeatureCfg.lqrsv_ratio_threshold = 0.8
FeatureCfg.spectral_hr_fs_band = [0.5, 4]  # corr. to hr from 30 to 240


# configurations for visualization
PlotCfg = ED()
# default const for the plot function in dataset.py
# used only when corr. values are absent
# all values are time bias w.r.t. corr. peaks, with units in ms
PlotCfg.p_onset = -40
PlotCfg.p_offset = 40
PlotCfg.q_onset = -20
PlotCfg.s_offset = 40
PlotCfg.qrs_radius = 60
PlotCfg.t_onset = -100
PlotCfg.t_offset = 60


# configurations for building deep learning models
# terminologies of stanford ecg repo. will be adopted
# NOTE: has been moved to the folder `model_configs`
ModelCfg = ED()
ModelCfg.fs = 500
ModelCfg.spacing = 1000 / ModelCfg.fs
ModelCfg.classes = []


# training configurations for machine learning and deep learning
TrainCfg = ED()

TrainCfg.db_dir = "/media/cfs/wenhao71/data/cinc2020_data/"
TrainCfg.log_dir = os.path.join(_BASE_DIR, 'log')
TrainCfg.checkpoints = os.path.join(_BASE_DIR, "checkpoints")
TrainCfg.keep_checkpoint_max = 100

TrainCfg.tranche_class_weights = ED({
    "AB": {
        'IAVB': 828, 'AF': 1374, 'AFL': 54, 'IRBBB': 86, 'LBBB': 274, 'PAC': 742, 'PVC': 196, 'RBBB': 1971, 'SB': 45, 'NSR': 922, 'STach': 303, 'TAb': 22,
    },
    "E": {
        'IAVB': 797, 'AF': 1514, 'AFL': 73, 'RBBB': 542, 'IRBBB': 1118, 'LAnFB': 1626, 'LBBB': 536, 'NSIVCB': 789, 'PAC': 555, 'LPR': 340, 'LQT': 118, 'QAb': 548, 'SA': 772, 'SB': 637, 'NSR': 18092, 'STach': 826, 'TAb': 2345, 'TInv': 294,
    },
    "F": {
        'IAVB': 769, 'AF': 570, 'AFL': 186, 'RBBB': 570, 'IRBBB': 407, 'LAnFB': 180, 'LBBB': 231, 'NSIVCB': 203, 'PAC': 640, 'LQT': 1391, 'QAb': 464, 'SA': 455, 'SB': 1677, 'NSR': 1752, 'STach': 1261, 'TAb': 2306, 'TInv': 812, 'PVC': 357,
    },
})
TrainCfg.tranche_class_weights = ED({
    t: {k: sum(t_cw.values())/v for k, v in t_cw.items()} \
        for t, t_cw in TrainCfg.tranche_class_weights.items()
})
TrainCfg.tranche_class_weights = ED({
    t: {k: v/min(t_cw.values()) for k, v in t_cw.items()} \
        for t, t_cw in TrainCfg.tranche_class_weights.items()
})
TrainCfg.tranche_classes = ED({
    t: list(t_cw.keys()) for t,t_cw in TrainCfg.tranche_class_weights.items()
})
TrainCfg.class_weights = ED({
    'IAVB': 2394, 'AF': 3473, 'AFL': 314, 'RBBB': 3083, 'IRBBB': 1611, 'LAnFB': 1806, 'LBBB': 1041, 'NSIVCB': 996, 'PAC': 1937, 'PVC': 553, 'LPR': 340, 'LQT': 1513, 'QAb': 1013, 'SA': 1238, 'SB': 2359, 'NSR': 20846, 'STach': 2391, 'TAb': 4673, 'TInv': 1111,
})  # count
TrainCfg.class_weights = ED({
    k: sum(TrainCfg.class_weights.values())/v for k, v in TrainCfg.class_weights.items()
})
TrainCfg.class_weights = ED({
    k: v/min(TrainCfg.class_weights.values()) for k, v in TrainCfg.class_weights.items()
})  # normalize so that the smallest weight equals 1
TrainCfg.classes = list(TrainCfg.class_weights.keys())
TrainCfg.tranches_for_training = ''  # one of '', 'AB', 'E', 'F'

TrainCfg.TRAIN_EPOCHS = 60000
TrainCfg.TRAIN_OPTIMIZER = "adam"  # "sgd"

TrainCfg.momentum = 0.949
TrainCfg.decay = 0.0005
TrainCfg.learning_rate = 0.00261
TrainCfg.burn_in = 1000
TrainCfg.max_batches = 500500
TrainCfg.steps = [40000, 45000]
TrainCfg.batch = 32
TrainCfg.loss = 'bce'

TrainCfg.cnn_name = "resnet"
