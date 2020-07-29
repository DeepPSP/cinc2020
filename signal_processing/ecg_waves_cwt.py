"""

12-lead ECG wave delineation, using algorithms proposed in ref. [1]

Update:
no existing implementation of daubechies cwt,
priority of implementation of this algorithm is set LOW

References:
-----------
[1] Yochum, Maxime, Charlotte Renaud, and Sabir Jacquir. "Automatic detection of P, QRS and T patterns in 12 leads ECG signal based on CWT." Biomedical Signal Processing and Control 25 (2016): 46-52.
[2] Li, Cuiwei, Chongxun Zheng, and Changfeng Tai. "Detection of ECG characteristic points using wavelet transforms." IEEE Transactions on biomedical Engineering 42.1 (1995): 21-28.
[3] https://encyclopediaofmath.org/wiki/Daubechies_wavelets
"""
from typing import Union, Optional, NoReturn

import numpy as np
from pywt import cwt


def continuous_daubechies():
    """
    """
    raise NotImplementedError
