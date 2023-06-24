import math
import numpy as np
from itertools import product
from tablite.config import Config
from tablite.base import Table
from tablite.reindex import reindex
from tablite.utils import sub_cls_check, unique_name
from tablite.mp_utils import share_mem, map_task, select_processing_method
from mplite import TaskManager, Task
import psutil

from tqdm import tqdm as _tqdm


def _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns):
    sub_cls_check(T, Table)
    sub_cls_check(other, Table)

    if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
        raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
    if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
        raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

    if any(key not in T.columns for key in left_keys):
        raise ValueError(f"left key(s) not found: {[k for k in left_keys if k not in T.columns]}")
    if any(key not in other.columns for key in right_keys):
        raise ValueError(f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}")

    if len(left_keys) != len(right_keys):
        raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

    for L, R in zip(left_keys, right_keys):
        Lcol, Rcol = T[L], other[R]
        if not set(Lcol.types()).intersection(set(Rcol.types())):
            left_types = tuple(t.__name__ for t in list(Lcol.types().keys()))
            right_types = tuple(t.__name__ for t in list(Rcol.types().keys()))
            raise TypeError(f"Type mismatch: Left key '{L}' {left_types} will never match right keys {right_types}")

    if not isinstance(left_columns, list) or not left_columns:
        raise TypeError("left_columns (list of strings) are required")
    if any(column not in T.columns for column in left_columns):
        raise ValueError(f"Column not found: {[c for c in left_columns if c not in T.columns]}")

    if not isinstance(right_columns, list) or not right_columns:
        raise TypeError("right_columns (list or strings) are required")
    if any(column not in other.columns for column in right_columns):
        raise ValueError(f"Column not found: {[c for c in right_columns if c not in other.columns]}")
    # Input is now guaranteed to be valid.


def join(T, other, left_keys, right_keys, left_columns, right_columns, kind="inner", tqdm=_tqdm, pbar=None):
    """
    short-cut for all join functions.
    kind: 'inner', 'left', 'outer', 'cross'
    """
    kinds = {
        "inner": T.inner_join,
        "left": T.left_join,
        "outer": T.outer_join,
        "cross": T.cross_join,
    }
    if kind not in kinds:
        raise ValueError(f"join type unknown: {kind}")
    f = kinds.get(kind, None)
    return f(other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def _sp_join(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=_tqdm, pbar=None):
    """
    private helper for single processing join
    """
    assert len(LEFT) == len(RIGHT)
    if pbar is None:
        total = len(left_columns) + len(right_columns)
        pbar = tqdm(total=total, desc="join", disable=Config.TQDM_DISABLE)

    result = reindex(T, LEFT, left_columns, tqdm=tqdm, pbar=pbar)
    second = reindex(other, RIGHT, right_columns, tqdm=tqdm, pbar=pbar)
    for name in right_columns:
        revised_name = unique_name(name, result.columns)
        result[revised_name] = second[name]

    return result


def _mp_join(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=_tqdm, pbar=None):
    return _sp_join(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)

    assert len(LEFT) == len(RIGHT)
    assert isinstance(LEFT, np.ndarray) and isinstance(RIGHT, np.ndarray)

    result = type(T)()
    cpus = max(psutil.cpu_count(logical=False), 2)
    step_size = math.ceil(len(LEFT) / cpus)

    with TaskManager(cpu_count=cpus) as tm:  # keeps the CPU pool alive during the whole join.
        for table, columns, side in ([T, left_columns, LEFT], [other, right_columns, RIGHT]):
            for name in columns:
                data = table[name][:]
                # TODO         ^---- determine how much memory is free and then decide
                # either to mmap the source or keep it in RAM.

                data, data_shm = share_mem(data, data.dtype)  # <-- this is source
                index, index_shm = share_mem(side, np.int64)  # <-- this is index
                # As all indices in `index` are positive, -1 is used as replacement for None.
                destination, dest_shm = share_mem(np.empty(shape=(len(side),)), data.dtype)  # <--this is destination.

                tasks = []
                start, end = 0, step_size
                for _ in range(cpus):
                    tasks.append(Task(map_task, data_shm, index_shm, dest_shm, start, end))
                    start, end = end, end + step_size
                # All CPUS now work on the same column and memory footprint is predetermined.
                results = tm.execute(tasks)
                if any(i is not None for i in results):
                    raise Exception("\n".join(filter(lambda x: x is not None, results)))

                # As the data and index no longer is needed, then can be closed.
                index_shm.close()
                data_shm.close()

                # As all the tasks have been completed, the Column can handle the pagination at once.
                name = unique_name(name, set(result.columns))

                # deal with Nones, before storing.
                nones = np.empty(shape=destination.shape, dtype=object)
                result[name] = np.where(index == -1, nones, destination)

                dest_shm.close()  # finally close dest.

    return result


def left_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param T: Table (left)
    :param other: Table (right)
    :param left_keys: list of keys for the join
    :param right_keys: list of keys for the join
    :param left_columns: list of left columns to retain, if None, all are retained.
    :param right_columns: list of right columns to retain, if None, all are retained.
    :return: new Table

    Example:
    SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
    Tablite: left_join = numbers.left_join(
        letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
    )
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    LEFT, RIGHT = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                LEFT.append(left_ix)
                RIGHT.append(right_ix)

    LEFT, RIGHT = np.array(LEFT), np.array(RIGHT)  # compress memory of python list to array.
    f = select_processing_method(len(LEFT) * len(left_columns + right_columns), _sp_join, _mp_join)
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def inner_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param T: Table (left)
    :param other: Table (right)
    :param left_keys: list of keys for the join
    :param right_keys: list of keys for the join
    :param left_columns: list of left columns to retain, if None, all are retained.
    :param right_columns: list of right columns to retain, if None, all are retained.
    :return: new Table
    Example:
    SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
    Tablite: inner_join = numbers.inner_join(
        letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
        )
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    LEFT, RIGHT = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, None)
        if right_ixs is None:
            continue
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                LEFT.append(left_ix)
                RIGHT.append(right_ix)

    LEFT, RIGHT = np.array(LEFT), np.array(RIGHT)
    f = select_processing_method(len(LEFT) * len(left_columns + right_columns), _sp_join, _mp_join)
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def outer_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param T: Table (left)
    :param other: Table (right)
    :param left_keys: list of keys for the join
    :param right_keys: list of keys for the join
    :param left_columns: list of left columns to retain, if None, all are retained.
    :param right_columns: list of right columns to retain, if None, all are retained.
    :return: new Table
    Example:
    SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
    Tablite: outer_join = numbers.outer_join(
        letters, left_keys=['colour'], right_keys=['color'], left_columns=['number'], right_columns=['letter']
        )
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    LEFT, RIGHT, RIGHT_UNUSED = [], [], set(right_index.keys())
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                LEFT.append(left_ix)
                RIGHT.append(right_ix)
                RIGHT_UNUSED.discard(left_key)

    for right_key in RIGHT_UNUSED:
        for right_ix in right_index[right_key]:
            LEFT.append(-1)
            RIGHT.append(right_ix)

    LEFT, RIGHT = np.array(LEFT), np.array(RIGHT)
    f = select_processing_method(len(LEFT) * len(left_columns + right_columns), _sp_join, _mp_join)
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def cross_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param T: Table (left)
    :param other: Table (right)
    :param left_keys: list of keys for the join
    :param right_keys: list of keys for the join
    :param left_columns: list of left columns to retain, if None, all are retained.
    :param right_columns: list of right columns to retain, if None, all are retained.
    :return: new Table

    CROSS JOIN returns the Cartesian product of rows from tables in the join.
    In other words, it will produce rows which combine each row from the first table
    with each row from the second table
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    LEFT, RIGHT = zip(*product(range(len(T)), range(len(other))))

    LEFT, RIGHT = np.array(LEFT), np.array(RIGHT)
    f = select_processing_method(len(LEFT), _sp_join, _mp_join)
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
