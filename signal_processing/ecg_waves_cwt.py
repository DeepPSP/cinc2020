"""

12-lead ECG wave delineation, using algorithms proposed in ref. [1]

References:
-----------
[1] Yochum, Maxime, Charlotte Renaud, and Sabir Jacquir. "Automatic detection of P, QRS and T patterns in 12 leads ECG signal based on CWT." Biomedical Signal Processing and Control 25 (2016): 46-52.
"""
from typing import Union, Optional, NoReturn

import numpy as np
from pywt import cwt

