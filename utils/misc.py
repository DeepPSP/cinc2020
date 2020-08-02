"""
"""
import os
from copy import deepcopy
from typing import Union, Optional, List, Dict, Sequence, NoReturn, Any
from numbers import Real

import numpy as np
from scipy import interpolate

np.set_printoptions(precision=5, suppress=True)


__all__ = [
    "get_record_list_recursive",
    "dict_to_str",
    "diff_with_step",
    "ms2samples",
    "get_mask",
    "plot_single_lead",
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


def ms2samples(t:Real, fs:Real) -> int:
    """ finished, checked,

    Parameters:
    -----------
    t: real number,
        time with units in ms
    fs: real number,
        sampling frequency of a signal

    Returns:
    --------
    n_samples: int,
        number of samples corresponding to time `t`
    """
    n_samples = t * fs // 1000
    return n_samples


def get_mask(shape:Union[int, Sequence[int]], critical_points:np.ndarray, left_bias:int, right_bias:int, return_fmt:str="mask") -> Union[np.ndarray,list]:
    """ finished, checked,

    get the mask around the `critical_points`

    Parameters:
    -----------
    shape: int, or sequence of int,
        shape of the mask (and the original data)
    critical_points: ndarray,
        indices (of the last dimension) of the points around which to be masked (value 1)
    left_bias: int, non-negative
        bias to the left of the critical points for the mask
    right_bias: int, non-negative
        bias to the right of the critical points for the mask
    return_fmt: str, default "mask",
        format of the return values,
        "mask" for the usual mask,
        can also be "intervals", which consists of a list of intervals

    Returns:
    --------
    mask: ndarray or list,
    """
    if isinstance(shape, int):
        shape = (shape,)
    l_itv = [[max(0,cp-left_bias),min(shape[-1],cp+right_bias)] for cp in critical_points]
    if return_fmt.lower() == "mask":
        mask = np.zeros(shape=shape, dtype=int)
        for itv in l_itv:
            mask[..., itv[0]:itv[1]] = 1
    elif return_fmt.lower() == "intervals":
        mask = l_itv
    return mask


def plot_single_lead(t:np.ndarray, sig:np.ndarray, ax:Optional[Any]=None, ticks_granularity:int=0, **kwargs) -> NoReturn:
    """ finished, NOT checked,

    Parameters:
    -----------
    to write
    """
    if 'plt' not in dir():
        import matplotlib.pyplot as plt
    palette = {'p_waves': 'green', 'qrs': 'red', 't_waves': 'pink',}
    plot_alpha = 0.4
    y_range = np.max(np.abs(sig)) + 100
    if ax is None:
        fig_sz_w = int(round(4.8 * (t[-1]-t[0])))
        fig_sz_h = 6 * y_range / 1500
        fig, ax = plt.subplots(figsize=(fig_sz_w, fig_sz_h))
    if kwargs.get('label', None):
        ax.plot(t, sig, label=kwargs.get('label'))
    else:
        ax.plot(t, sig)
    ax.axhline(y=0, linestyle='-', linewidth='1.0', color='red')
    # NOTE that `Locator` has default `MAXTICKS` equal to 1000
    if ticks_granularity >= 1:
        ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
        ax.yaxis.set_major_locator(plt.MultipleLocator(500))
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    if ticks_granularity >= 2:
        ax.xaxis.set_minor_locator(plt.MultipleLocator(0.04))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(100))
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
    
    waves = kwargs.get('waves', {'p_waves':[], 'qrs':[], 't_waves':[]})
    for w, l_itv in waves.items():
        for itv in l_itv:
            ax.axvspan(itv[0], itv[1], color=palette[w], alpha=plot_alpha)
    ax.legend(loc='upper left')
    ax.set_xlim(t[0], t[-1])
    ax.set_ylim(-y_range, y_range)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Voltage [μV]')
