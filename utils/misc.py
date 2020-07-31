"""
"""
import os
from copy import deepcopy
from typing import Union, Optional, List, Sequence, NoReturn
from numbers import Real

import numpy as np
from scipy import interpolate

np.set_printoptions(precision=5, suppress=True)


__all__ = [
    "get_record_list_recursive",
    "dict_to_str",
    "diff_with_step",
    "ms2samples",
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
