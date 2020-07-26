"""
spectral analysis on the sequence of rr intervals (hrv, etc.),
and on the ecg signal itself (heart rate, etc.)
"""
import numpy as np
from scipy.signal import spectrogram, welch

from utils.misc import resample_irregular_timeseries
