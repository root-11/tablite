from tablite.base import Table
from tablite.config import Config
from tablite.utils import sub_cls_check
import numpy as np
from tqdm import tqdm as _tqdm


def reindex(T, index, names=None, tqdm=_tqdm, pbar=None):
    """Constant Memory helper for reindexing pages.

    Memory usage is set by datatype and Config.PAGE_SIZE

    Args:
        T (Table): subclass of Table
        index (np.array): int64.
        names (list, str): list of names from T to reindex.
        tqdm (tqdm, optional): Defaults to _tqdm.
        pbar (pbar, optional): Defaults to None.

    Returns:
        _type_: _description_
    """
    if names is None:
        names = list(T.columns.keys())

    if pbar is None:
        total = len(names)
        pbar = tqdm(total=total, desc="join", disable=Config.TQDM_DISABLE)

    sub_cls_check(T, Table)
    cls = type(T)
    result = cls()
    for name in names:
        result.add_column(name)
        col = result[name]

        for start, end in Config.page_steps(len(index)):
            indices = index[start:end]
            values = T[name].get_by_indices(indices)
            # in these values, the index of -1 will be wrong.
            # so if there is any -1 in the indices, they will
            # have to be replaced with Nones
            mask = indices == -1
            if np.any(mask):
                nones = np.full(index.shape, fill_value=None)
                values = np.where(mask, nones, values)
            col.extend(values)
        pbar.update(1)

    return result
