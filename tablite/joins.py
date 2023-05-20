import numpy as np
from base import Table
from itertools import product
from utils import sub_cls_check, unique_name
from config import Config
from mp_utils import share_mem, maskify, indexing_task
from mplite import TaskManager, Task
import psutil


def _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns):
    sub_cls_check(T, Table)

    if not issubclass(other, Table):
        raise TypeError(f"other expected other to be type Table, not {type(other)}")

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
    result = T()

    if pbar is None:
        total = len(left_columns) + len(right_columns)
        pbar = tqdm(total=total, desc="join")

    for col_name in left_columns:
        col_data = T[col_name][:]
        result[col_name] = [col_data[k] if k is not None else None for k in LEFT]
        pbar.update(1)
    for col_name in right_columns:
        col_data = other[col_name][:]
        revised_name = unique_name(col_name, result.columns)
        result[revised_name] = [col_data[k] if k is not None else None for k in RIGHT]
        pbar.update(1)
    return result


def _mp_join(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=_tqdm, pbar=None):
    """
    private helper for multiprocessing join
    TODO: better memory management when processes share column chunks (requires masking Nones)
    """
    LEFT_NONE_MASK, RIGHT_NONE_MASK = (maskify(arr) for arr in (LEFT, RIGHT))

    left_arr, left_shm = share_mem(LEFT, np.int64)
    right_arr, right_shm = share_mem(RIGHT, np.int64)
    left_msk_arr, left_msk_shm = share_mem(LEFT_NONE_MASK, np.bool8)
    right_msk_arr, right_msk_shm = share_mem(RIGHT_NONE_MASK, np.bool8)

    final_len = len(LEFT)

    assert len(LEFT) == len(RIGHT)

    tasks = []
    columns_refs = {}

    rows_per_page = Config.PAGE_SIZE

    for name in left_columns:
        col = T[name]
        container = columns_refs[name] = []

        offset = 0

        while offset < final_len or final_len == 0:  # create an empty page
            new_offset = min(offset + rows_per_page, final_len)
            slice_ = slice(offset, new_offset)
            d_key = mem.new_id("/column")
            container.append(d_key)
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=left_shm.name,
                    shm_name_for_sort_index_mask=left_msk_shm.name,
                    shape=left_arr.shape,
                    slice_=slice_,
                )
            )

            offset = new_offset

            if final_len == 0:
                break

    for name in right_columns:
        revised_name = unique_name(name, columns_refs.keys())
        col = other[name]
        container = columns_refs[revised_name] = []

        offset = 0

        while offset < final_len or final_len == 0:  # create an empty page
            new_offset = min(offset + rows_per_page, final_len)
            slice_ = slice(offset, new_offset)
            d_key = mem.new_id("/column")
            container.append(d_key)
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=right_shm.name,
                    shm_name_for_sort_index_mask=right_msk_shm.name,
                    shape=right_arr.shape,
                    slice_=slice_,
                )
            )

            offset = new_offset

            if final_len == 0:
                break

    if pbar is None:
        total = len(tasks)
        pbar = tqdm(total=total, desc="join")

    with TaskManager(cpu_count=min(psutil.cpu_count(), total)) as tm:
        results = tm.execute(tasks, tqdm=tqdm, pbar=pbar)

        if any(i is not None for i in results):
            raise Exception("\n".join(filter(lambda x: x is not None, results)))

    merged_column_refs = {k: mem.mp_merge_columns(v) for k, v in columns_refs.items()}

    with h5py.File(mem.path, "r+") as h5:
        table_key = mem.new_id("/table")
        dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty("f"))
        dset.attrs["columns"] = json.dumps(merged_column_refs)
        dset.attrs["saved"] = False

    left_shm.close()
    left_shm.unlink()
    right_shm.close()
    right_shm.unlink()

    left_msk_shm.close()
    left_msk_shm.unlink()
    right_msk_shm.close()
    right_msk_shm.unlink()

    t = Table.load(path=mem.path, key=table_key)
    return t


def _select_processing_method(fields):
    assert isinstance(fields, int)
    if Config.MULTIPROCESSING_MODE == Config.FALSE:  # tcfg.PROCESSING_PRIORITY == "sp":
        f = _sp_join
    elif Config.MULTIPROCESSING_MODE == Config.FORCE:  # tcfg.PROCESSING_PRIORITY == "mp":
        f = _mp_join
    elif fields < Config.SINGLE_PROCESSING_LIMIT:
        f = _sp_join
    else:  # use_mp:
        f = _mp_join
    return f


def left_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param other: T, other = (left, right)
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
        right_ixs = right_index.get(left_key, (None,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                LEFT.append(left_ix)
                RIGHT.append(right_ix)

    f = _select_processing_method(fields=len(LEFT) * len(left_columns + right_columns))
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def inner_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param other: T, other = (left, right)
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

    f = _select_processing_method(fields=len(LEFT) * len(left_columns + right_columns))
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def outer_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
    :param other: T, other = (left, right)
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
        right_ixs = right_index.get(left_key, (None,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                LEFT.append(left_ix)
                RIGHT.append(right_ix)
                RIGHT_UNUSED.discard(left_key)

    for right_key in RIGHT_UNUSED:
        for right_ix in right_index[right_key]:
            LEFT.append(None)
            RIGHT.append(right_ix)

    f = _select_processing_method(fields=len(LEFT) * len(left_columns + right_columns))
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)


def cross_join(T, other, left_keys, right_keys, left_columns=None, right_columns=None, tqdm=_tqdm, pbar=None):
    """
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

    f = _select_processing_method(fields=len(LEFT))
    return f(T, other, LEFT, RIGHT, left_columns, right_columns, tqdm=tqdm, pbar=pbar)
