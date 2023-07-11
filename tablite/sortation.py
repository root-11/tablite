import os
import numpy as np
import psutil
from mplite import Task, TaskManager
from tablite.mp_utils import share_mem, reindex_task, select_processing_method
from tablite.datatypes import multitype_set, numpy_to_python
from tablite.reindex import reindex as _reindex
from tablite.sort_utils import modes as sort_modes
from tablite.sort_utils import rank as sort_rank
from tablite.base import Table, Column, Page
from tablite.utils import sub_cls_check, type_check

from tqdm import tqdm as _tqdm


def sort_index(T, mapping, sort_mode="excel", tqdm=_tqdm, pbar=None):
    """
    helper for methods `sort` and `is_sorted`

    param: sort_mode: str: "alphanumeric", "unix", or, "excel" (default)
    param: **kwargs: sort criteria. See Table.sort()
    """

    sub_cls_check(T, Table)

    if not isinstance(mapping, dict) or not mapping:
        raise TypeError("Expected mapping (dict)?")

    for k, v in mapping.items():
        if k not in T.columns:
            raise ValueError(f"no column {k}")
        if not isinstance(v, bool):
            raise ValueError(f"{k} was mapped to {v} - a non-boolean")

    if sort_mode not in sort_modes:
        raise ValueError(f"{sort_mode} not in list of sort_modes: {list(sort_modes)}")

    rank = {i: tuple() for i in range(len(T))}  # create index and empty tuple for sortation.

    _pbar = tqdm(total=len(mapping.items()), desc="creating sort index") if pbar is None else pbar

    for key, reverse in mapping.items():
        col = T[key][:]
        ranks = sort_rank(values=[numpy_to_python(v) for v in multitype_set(col)], reverse=reverse, mode=sort_mode)
        assert isinstance(ranks, dict)
        for ix, v in enumerate(col):
            v2 = numpy_to_python(v)
            rank[ix] += (ranks[v2],)  # add tuple for each sortation level.

        _pbar.update(1)

    del col
    del ranks

    new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
    del rank  # free memory.

    new_order.sort()
    sorted_index = [i for _, i in new_order]  # new index is extracted.
    new_order.clear()
    return np.array(sorted_index, dtype=np.int64)


def reindex(T, index):
    """
    index: list of integers that declare sort order.

    Examples:

        Table:  ['a','b','c','d','e','f','g','h']
        index:  [0,2,4,6]
        result: ['b','d','f','h']

        Table:  ['a','b','c','d','e','f','g','h']
        index:  [0,2,4,6,1,3,5,7]
        result: ['a','c','e','g','b','d','f','h']

    """
    sub_cls_check(T, Table)
    if isinstance(index, list):
        index = np.array(index, dtype=int)
    type_check(index, np.ndarray)
    if max(index) >= len(T):
        raise IndexError("index out of range: max(index) > len(self)")
    if min(index) < -len(T):
        raise IndexError("index out of range: min(index) < -len(self)")

    fields = len(T) * len(T.columns)
    m = select_processing_method(fields, _reindex, _mp_reindex)
    return m(T, index)


def _sp_reindex(T, index):
    return _reindex(T, index)


def _mp_reindex(T, index):
    assert isinstance(index, np.ndarray)
    return _sp_reindex(T, index)

    index, shm = share_mem(index, dtype=index.dtype)
    # shm = shared_memory.SharedMemory(create=True, size=index.nbytes)  # the co_processors will read this.
    # sort_index = np.ndarray(index.shape, dtype=index.dtype, buffer=shm.buf)
    # sort_index[:] = index

    new = {}
    tasks = []
    for name in T.columns:
        col = T[name]
        new[name] = []

        start, end = 0, 0
        for page in col.pages:
            start, end = end, start + len(page)
            src = page.path
            dst = page.path.parent / f"{next(Page.ids)}.npy"
            t = Task(reindex_task, src, dst, shm.name, index.shape, start, end)
            new[name].append(dst)
            tasks.append(t)

    cpus = min(len(tasks), psutil.cpu_count(logical=False))
    with TaskManager(cpu_count=cpus) as tm:
        errs = tm.execute(tasks)
        if any(errs):
            raise Exception("\n".join(filter(lambda x: x is not None, errs)))

    shm.close()
    shm.unlink()

    t = type(T)()
    for name in T.columns:
        t[name] = Column(t.path)
        for dst in new[name]:
            data = np.load(dst, allow_pickle=True, fix_imports=False)
            t[name].extend(data)
            os.remove(dst)
    return t


def sort(T, mapping, sort_mode="excel", tqdm=_tqdm, pbar: _tqdm = None):
    """Perform multi-pass sorting with precedence given order of column names.
    sort_mode: str: "alphanumeric", "unix", or, "excel"
    kwargs:
        keys: columns,
        values: 'reverse' as boolean.

    examples:
    Table.sort('A'=False) means sort by 'A' in ascending order.
    Table.sort('A'=True, 'B'=False) means sort 'A' in descending order, then (2nd priority)
    sort B in ascending order.
    """
    sub_cls_check(T, Table)

    index = sort_index(T, mapping, sort_mode=sort_mode, tqdm=_tqdm, pbar=pbar)
    m = select_processing_method(len(T) * len(T.columns), _sp_reindex, _mp_reindex)
    return m(T, index)


def is_sorted(T, mapping, sort_mode="excel"):
    """Performs multi-pass sorting check with precedence given order of column names.

    Args:
        mapping: sort criteria. See Table.sort()
        sort_mode = sort mode. See Table.sort()

    Returns:
        bool
    """
    index = sort_index(T, mapping, sort_mode=sort_mode)
    match = np.arange(len(T))
    return np.all(index == match)
