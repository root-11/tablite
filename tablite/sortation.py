import os
from config import Config
import numpy as np
import psutil
from mplite import Task, TaskManager
from mp_utils import shared_memory, reindex_task
from sort_utils import modes as sort_modes
from sort_utils import rank as sort_rank
from base import Table, Column, Page
from utils import sub_cls_check, type_check


def sort_index(T, sort_mode="excel", tqdm=_tqdm, pbar=None, **kwargs):
    """
    helper for methods `sort` and `is_sorted`

    param: sort_mode: str: "alphanumeric", "unix", or, "excel" (default)
    param: **kwargs: sort criteria. See Table.sort()
    """

    sub_cls_check(T, Table)

    if not isinstance(kwargs, dict):
        raise ValueError("Expected keyword arguments, did you forget the ** in front of your dict?")
    if not kwargs:
        kwargs = {c: False for c in T.columns}

    for k, v in kwargs.items():
        if k not in T.columns:
            raise ValueError(f"no column {k}")
        if not isinstance(v, bool):
            raise ValueError(f"{k} was mapped to {v} - a non-boolean")

    if sort_mode not in sort_modes:
        raise ValueError(f"{sort_mode} not in list of sort_modes: {list(sort_modes)}")

    rank = {i: tuple() for i in range(len(T))}  # create index and empty tuple for sortation.

    _pbar = tqdm(total=len(kwargs.items()), desc="creating sort index") if pbar is None else pbar

    for key, reverse in kwargs.items():
        col = T[key][:]
        col = col.tolist() if isinstance(col, np.ndarray) else col
        ranks = sort_rank(values=set(col), reverse=reverse, mode=sort_mode)
        assert isinstance(ranks, dict)
        for ix, v in enumerate(col):
            rank[ix] += (ranks[v],)  # add tuple

        _pbar.update(1)

    new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
    rank.clear()  # free memory.
    new_order.sort()
    sorted_index = [i for _, i in new_order]  # new index is extracted.
    new_order.clear()
    return np.array(sorted_index)


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
        index = np.ndarray(index, dtype=np.int64)
    type_check(index, np.ndarray)
    if max(index) >= len(T):
        raise IndexError("index out of range: max(index) > len(self)")
    if min(index) < -len(T):
        raise IndexError("index out of range: min(index) < -len(self)")

    cpus = max(psutil.cpu_count(), 1)
    if cpus < 2 or Config.MULTIPROCESSING_MODE == Config.FALSE:
        return _sp_reindex(T, index)
    else:
        return _mp_reindex(T, index)


def _sp_reindex(T, index):
    t = type(T)()
    for name in T.columns:
        t[name] = np.take(T[name][:], index)
        np.tak
    return t


def _mp_reindex(T, index):
    shm = shared_memory.SharedMemory(create=True, size=index.nbytes)  # the co_processors will read this.
    sort_index = np.ndarray(index.shape, dtype=index.dtype, buffer=shm.buf)
    sort_index[:] = index

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
            t = Task(reindex_task, src, dst, shm, start, end)
            new.append(dst)

    cpus = psutil.cpu_count()
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


def sort(T, sort_mode="excel", **kwargs):
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

    index = sort_index(sort_mode=sort_mode, **kwargs)

    cpus = max(psutil.cpu_count(), 1)
    if cpus < 2 or Config.MULTIPROCESSING_MODE == Config.FALSE:
        return _sp_reindex(T, index)
    else:
        return _mp_reindex(T, index)


def is_sorted(T, **kwargs):
    """Performs multi-pass sorting check with precedence given order of column names.
    **kwargs: optional: sort criteria. See Table.sort()
    :return bool
    """
    index = sort_index(**kwargs)
    match = np.arange(len(T))
    return np.all(index == match)
