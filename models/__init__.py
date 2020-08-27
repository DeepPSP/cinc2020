"""
Resources:
----------
1. ECG CRNN
2. special detectors
3. to add more

Rules:
------
to write
"""

from .ecg_crnn import ECG_CRNN
from .ecg_unet import ECG_UNET


__all__ = [s for s in dir() if not s.startswith('_')]
