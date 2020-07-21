"""
"""
import os
from copy import deepcopy
from typing import Union, Optional, List, Sequence

import numpy as np


__all__ = [
    "get_record_list_recursive",
    "dict_to_str",
    "diff_with_step",
    "smooth",
    "MovingAverage",
]


def get_record_list_recursive(db_dir:str, rec_ext:str) -> List[str]:
    """ finished, checked,

    get the list of records in `db_dir` recursively,
    for example, there are two folders 'patient1', 'patient2' in `db_dir`,
    and there are records 'A0001', 'A0002', ... in 'patient1'; 'B0001', 'B0002', ... in 'patient2',
    then the output would be 'patient1{sep}A0001', ..., 'patient2{sep}B0001', ...,
    sep is determined by the system

    Parameters:
    -----------
    db_dir: str,
        the parent (root) path of the whole database
    rec_ext: str,
        extension of the record files

    Returns:
    --------
    res: list of str,
        list of records, in lexicographical order
    """
    res = []
    db_dir = os.path.join(db_dir, "tmp").replace("tmp", "")  # make sure `db_dir` ends with a sep
    roots = [db_dir]
    while len(roots) > 0:
        new_roots = []
        for r in roots:
            tmp = [os.path.join(r, item) for item in os.listdir(r)]
            res += [item for item in tmp if os.path.isfile(item)]
            new_roots += [item for item in tmp if os.path.isdir(item)]
        roots = deepcopy(new_roots)
    res = [os.path.splitext(item)[0].replace(db_dir, "") for item in res if item.endswith(rec_ext)]
    res = sorted(res)

    return res


def dict_to_str(d:Union[dict, list, tuple], current_depth:int=1, indent_spaces:int=4) -> str:
    """ finished, checked,

    convert a (possibly) nested dict into a `str` of json-like formatted form,
    this nested dict might also contain lists or tuples of dict (and of str, int, etc.)

    Parameters:
    -----------
    d: dict, or list, or tuple,
        a (possibly) nested `dict`, or a list of `dict`
    current_depth: int, default 1,
        depth of `d` in the (possible) parent `dict` or `list`
    indent_spaces: int, default 4,
        the indent spaces of each depth

    Returns:
    --------
    s: str,
        the formatted string
    """
    assert isinstance(d, (dict, list, tuple))
    if len(d) == 0:
        s = f"{{}}" if isinstance(d, dict) else f"[]"
        return s
    s = "\n"
    unit_indent = " "*indent_spaces
    prefix = unit_indent*current_depth
    if isinstance(d, (list, tuple)):
        for v in d:
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{dict_to_str(v, current_depth+1)}\n"
            else:
                val = f'\042{v}\042' if isinstance(v, str) else v
                s += f"{prefix}{val}\n"
    elif isinstance(d, dict):
        for k, v in d.items():
            key = f'\042{k}\042' if isinstance(k, str) else k
            if isinstance(v, (dict, list, tuple)):
                s += f"{prefix}{key}: {dict_to_str(v, current_depth+1)}\n"
            else:
                val = f'\042{v}\042' if isinstance(v, str) else v
                s += f"{prefix}{key}: {val}\n"
    s += unit_indent*(current_depth-1)
    s = f"{{{s}}}" if isinstance(d, dict) else f"[{s}]"
    return s


def diff_with_step(a:np.ndarray, step:int=1, **kwargs) -> np.ndarray:
    """ finished, checked,

    compute a[n+step] - a[n] for all valid n

    Parameters:
    -----------
    a: ndarray,
        the input data
    step: int, default 1,
        the step to compute the difference
    kwargs: dict,

    Returns:
    --------
    d: ndarray:
        the difference array
    """
    if step >= len(a):
        raise ValueError(f"step ({step}) should be less than the length ({len(a)}) of `a`")
    d = a[step:] - a[:-step]
    return d


def smooth(x:np.ndarray, window_len:int=11, window:str='hanning', mode:str='valid', keep_dtype:bool=True) -> np.ndarray:
    """ finished, checked
    
    smooth the 1d data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    Parameters:
    -----------
    x: ndarray,
        the input signal 
    window_len: int, default 11,
        the length of the smoothing window,
        (previously should be an odd integer, currently can be any (positive) integer)
    window: str, default 'hanning',
        the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman',
        flat window will produce a moving average smoothing
    mode: str, default 'valid',
        ref. np.convolve
    keep_dtype: bool, default True,
        dtype of the returned value keeps the same with that of `x` or not

    Returns:
    --------
    y: ndarray,
        the smoothed signal
        
    Example:
    --------
    >>> t = linspace(-2, 2, 0.1)
    >>> x = sin(t) + randn(len(t)) * 0.1
    >>> y = smooth(x)
    
    See also: 
    ---------
    np.hanning, np.hamming, np.bartlett, np.blackman, np.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string

    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.

    References:
    -----------
    [1] https://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    """
    radius = min(len(x), window_len)
    radius = radius if radius%2 == 1 else radius-1

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    # if x.size < radius:
    #     raise ValueError("Input vector needs to be bigger than window size.")

    if radius < 3:
        return x
    
    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")

    s = np.r_[x[radius-1:0:-1], x, x[-2:-radius-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w = np.ones(radius,'d')
    else:
        w = eval('np.'+window+'(radius)')

    y = np.convolve(w/w.sum(), s, mode=mode)
    y = y[(radius//2-1):-(radius//2)-1]
    assert len(x) == len(y)

    if keep_dtype:
        y = y.astype(x.dtype)
    
    return y


class MovingAverage(object):
    """

    moving average

    References:
    -----------
    [1] https://en.wikipedia.org/wiki/Moving_average
    """
    def __init__(self, data:Sequence, **kwargs):
        """
        Parameters:
        -----------
        data: sequence,
            the series data to compute its moving average
        """
        self.data = np.array(data)
        self.verbose = kwargs.get("verbose", 0)

    def cal(self, method:str, **kwargs) -> np.ndarray:
        """
        Parameters:
        -----------
        method: str,
            method for computing moving average, can be one of
            - 'sma', 'simple', 'simple moving average'
            - 'ema', 'ewma', 'exponential', 'exponential weighted', 'exponential moving average', 'exponential weighted moving average'
            - 'cma', 'cumulative', 'cumulative moving average'
            - 'wma', 'weighted', 'weighted moving average'
        """
        m = method.lower().replace('_', ' ')
        if m in ['sma', 'simple', 'simple moving average']:
            func = self._sma
        elif m in ['ema', 'ewma', 'exponential', 'exponential weighted', 'exponential moving average', 'exponential weighted moving average']:
            func = self._ema
        elif m in ['cma', 'cumulative', 'cumulative moving average']:
            func = self._cma
        elif m in ['wma', 'weighted', 'weighted moving average']:
            func = self._wma
        else:
            raise NotImplementedError
        return func(**kwargs)

    def _sma(self, window:int=5, center:bool=False, **kwargs) -> np.ndarray:
        """
        simple moving average

        Parameters:
        -----------
        window: int, default 5,
            window length of the moving average
        center: bool, default False,
            if True, when computing the output value at each point, the window will be centered at that point;
            otherwise the previous `window` points of the current point will be used
        """
        smoothed = []
        if center:
            hw = window//2
            window = hw*2+1
        for n in range(window):
            smoothed.append(np.mean(self.data[:n+1]))
        prev = smoothed[-1]
        for n, d in enumerate(self.data[window:]):
            s = prev + (d - self.data[n]) / window
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        if center:
            smoothed[hw:-hw] = smoothed[window-1:]
            for n in range(hw):
                smoothed[n] = np.mean(self.data[:n+hw+1])
                smoothed[-n-1] = np.mean(self.data[-n-hw-1:])
        return smoothed

    def _ema(self, weight:float=0.6, **kwargs) -> np.ndarray:
        """
        exponential moving average,
        which is also the function used in Tensorboard Scalar panel,
        whose parameter `smoothing` is the `weight` here

        Parameters:
        -----------
        weight: float, default 0.6,
            weight of the previous data point
        """
        smoothed = []
        prev = self.data[0]
        for d in self.data:
            s = prev * weight + (1 - weight) * d
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _cma(self, **kwargs) -> np.ndarray:
        """
        cumulative moving average
        """
        smoothed = []
        prev = 0
        for n, d in enumerate(self.data):
            s = prev + (d - prev) / (n+1)
            prev = s
            smoothed.append(s)
        smoothed = np.array(smoothed)
        return smoothed

    def _wma(self, window:int=5, **kwargs) -> np.ndarray:
        """
        weighted moving average

        Parameters:
        -----------
        window: int, default 5,
            window length of the moving average
        """
        # smoothed = []
        # total = []
        # numerator = []
        conv = np.arange(1, window+1)[::-1]
        deno = np.sum(conv)
        smoothed = np.convolve(conv, self.data, mode='same') / deno
        return smoothed
