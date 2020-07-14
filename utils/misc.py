"""
"""
import os
from copy import deepcopy


__all__ = [
    "get_record_list_recursive",
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
