from typing import List
import numpy as np
from pathlib import Path
from tablite.config import Config, TaskManager, Task
from tablite.base import Table, Column
from tablite.utils import sub_cls_check, unique_name, type_check


from tqdm import tqdm as _tqdm


def _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns):
    sub_cls_check(T, Table)
    sub_cls_check(other, Table)

    if not isinstance(left_keys, list) and all(isinstance(k, str) for k in left_keys):
        raise TypeError(f"Expected keys as list of strings, not {type(left_keys)}")
    if not isinstance(right_keys, list) and all(isinstance(k, str) for k in right_keys):
        raise TypeError(f"Expected keys as list of strings, not {type(right_keys)}")

    if any(key not in T.columns for key in left_keys):
        e = f"left key(s) not found: {[k for k in left_keys if k not in T.columns]}"
        raise ValueError(e)
    if any(key not in other.columns for key in right_keys):
        e = f"right key(s) not found: {[k for k in right_keys if k not in other.columns]}"
        raise ValueError(e)

    if len(left_keys) != len(right_keys):
        raise ValueError(f"Keys do not have same length: \n{left_keys}, \n{right_keys}")

    for L, R in zip(left_keys, right_keys):
        Lcol, Rcol = T[L], other[R]
        if not set(Lcol.types()).intersection(set(Rcol.types())):
            left_types = tuple(t.__name__ for t in list(Lcol.types().keys()))
            right_types = tuple(t.__name__ for t in list(Rcol.types().keys()))
            e = f"Type mismatch: Left key '{L}' {left_types} will never match right keys {right_types}"
            raise TypeError(e)

    if not isinstance(left_columns, list) or not left_columns:
        raise TypeError("left_columns (list of strings) are required")
    if any(column not in T.columns for column in left_columns):
        e = f"Column not found: {[c for c in left_columns if c not in T.columns]}"
        raise ValueError(e)

    if not isinstance(right_columns, list) or not right_columns:
        raise TypeError("right_columns (list or strings) are required")
    if any(column not in other.columns for column in right_columns):
        e = f"Column not found: {[c for c in right_columns if c not in other.columns]}"
        raise ValueError(e)
    # Input is now guaranteed to be valid.


def join(
    T: Table,
    other: Table,
    left_keys: List[str],
    right_keys: List[str],
    left_columns: List[str] | None,
    right_columns: List[str] | None,
    kind: str = "inner",
    tqdm=_tqdm,
    pbar=None,
):
    """short-cut for all join functions.

    Args:
        T (Table): left table
        other (Table): right table
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
        left_columns (list): list of columns names to retain from left table.
            If None, all are retained.
        right_columns (list): list of columns names to retain from right table.
            If None, all are retained.
        kind (str, optional): 'inner', 'left', 'outer', 'cross'. Defaults to "inner".
        tqdm (tqdm, optional): tqdm progress counter. Defaults to _tqdm.
        pbar (tqdm.pbar, optional): tqdm.progressbar. Defaults to None.

    Raises:
        ValueError: if join type is unknown.

    Returns:
        Table: joined table.
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
    return f(
        other, left_keys, right_keys, left_columns, right_columns, tqdm=tqdm, pbar=pbar
    )


def _where(
    T: Table,
    mapping: Table,
    field: str,
    left: str,
    right: str,
    new: str,
    start: int,
    end: int,
    path: Path,
):
    """takes from LEFT where criteria is True else RIGHT.

    Args:
        T (Table): Table to change.
        mapping (Table): mapping table.
        field (str): bool field name in mapping
            if True take left column
            else take right column
        left (str): column name
        right (str): column name
        new (str): new name
        start (int): start index
        end (int): end index

    :returns: None
    """
    criteria = mapping[field][start:end]
    left_values = T[left][start:end]
    right_values = T[right][start:end]
    new_values = np.where(criteria, left_values, right_values)
    return Table({new: new_values}, _path=path)


def _mapping(
    T: Table,
    other: Table,
    left_slice: slice,
    right_slice: slice,
    left_keys: List[str],
    right_keys: List[str],
    path: Path,
):
    """create mapping for left and right keys.

    Args:
        T (Table): left table
        Other (Table): right table
        left_slice (slice): slice of left table to perform mapping on.
        right_slice (slice): slice of right table to perform mapping on.
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
        path (pathlib.Path): directory of the main process

    Returns:
        Table: table initiated in the main process' working directory
    """
    left = T[left_keys + [left_slice]]
    right = other[right_keys + [right_slice]]
    left_index = left.index(*left_keys)
    right_index = right.index(*right_keys)
    # Note: The memory footprint on indexing is unpredictable.
    # If this crashes the user would have to create a key manually
    # on each table and use that as index.
    del left, right

    _left, _right = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)

    mapping = Table({"left": _left, "right": _right}, _path=path)
    return mapping


def _reindex_page(
    T: Table,
    column_name: str,
    mapping: Table,
    index: str,
    start: int,
    end: int,
    path: Path,
):
    """reindexes T on column_name using mapping between start and end.

    Args:
        T (Table): table to reindex
        column_name (str): column to reindex
        mapping (Table): table with mapping
        index (str): column name of index in mapping
        start (int): start of range
        end (int): end of range
        path (Path): directory of the main process

    Returns:
        Table: table initiated in the main process' working directory
    """
    part = slice(start, end)
    ix_arr = mapping[index][part]
    array = T[column_name][part]
    if np.all(ix_arr == np.arange(start, end)):
        pass  # if the range is not reordered, just copy the reference.
    else:  # the range needs reordering ...
        array = np.take(array, ix_arr)

    remapped_T = Table({column_name: array}, _path=path)

    return remapped_T


def left_join(
    T: Table,
    other: Table,
    left_keys: List[str],
    right_keys: List[str],
    left_columns: List[str] | None = None,
    right_columns: List[str] | None = None,
    merge_keys: bool = False,
    tqdm=_tqdm,
    pbar=None,
    task_manager=None,
):
    """perform left join on two tables.

    Args:
        T (Table): _description_
        other (Table): _description_
        left_keys (list of column names): left keys to join on.
        right_keys (list of column names): right keys to join on.
        left_columns (list of column names, optional): Columns to keep. Defaults to None.
        right_columns (list of column names, optional): Columns to keep. Defaults to None.
        merge_keys (bool, optional): merges keys to the left, so that cases where right key is None, a key exists.
            Defaults to False.
        tqdm (tqdm, optional): _description_. Defaults to _tqdm.
        pbar (tqdm.pbar, optional): _description_. Defaults to None.
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)
    assert merge_keys in {True, False}

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    if pbar is None:
        _pbar_created_here = True
        pbar = tqdm(total=5, desc="join", disable=Config.TQDM_DISABLE)
    else:
        _pbar_created_here = False
    pbar.update(0)  # pbar start

    class ProgressBar(object):
        def update(self, n):
            pbar.update(n / len(tasks))

    # create left and right index
    if task_manager is None:
        task_manager = TaskManager

    tasks = []

    _pid_dir = Path(Config.workdir) / Config.pid

    # step 1: create mapping tasks
    step = Config.PAGE_SIZE
    for left in range(0, len(T) + 1, step):
        for right in range(0, len(other) + 1, step):
            left_slice = slice(left, left + step, 1)
            right_slice = slice(right, right + step, 1)
            task = Task(
                _mapping,
                T=T,
                other=other,
                left_slice=left_slice,
                right_slice=right_slice,
                left_keys=left_keys,
                right_keys=right_keys,
                path=_pid_dir,
            )
            tasks.append(task)

    with task_manager() as tm:
        results = tm.execute(tasks, pbar=ProgressBar())

    # step 2: assemble mapping from tasks
    mapping = Table({"left": [], "right": []})
    for result in results:
        assert isinstance(result, Table)
        mapping += result

    pbar.update(1)

    # step 3: initiate reindexing tasks
    tasks = []
    names = []  # will store (old name, new name) for derefences during assemble.
    new_table = Table()
    n = len(mapping)
    for name in T.columns:
        new_table.add_column(name)
        names.append((name, name))

        for start in range(0, len(T) + 1, step):
            task = Task(
                _reindex_page,
                T=T,
                column_name=name,
                mapping=mapping,
                index="left",
                start=start,
                end=min(start + step, n),
                path=_pid_dir,
            )
            tasks.append(task)

    for name in other.columns:
        new_name = unique_name(name, new_table.columns)
        new_table.add_column(new_name)
        names.append((new_name, name))

        for start in range(0, len(other) + 1, step):
            task = Task(
                _reindex_page,
                T=other,
                column_name=name,
                mapping=mapping,
                index="right",
                start=start,
                end=min(start + step, n),
                path=_pid_dir,
            )
            tasks.append(task)

    with task_manager() as tm:
        results = tm.execute(tasks, pbar=ProgressBar())

    # step 4: assemble the result
    for result, (old, new) in zip(results, names):
        new_table[old].extend(result[new])

    pbar.update(1)

    # step 5: merge keys (if required)
    if merge_keys is True:
        mapping["boolean map"] = np.array(mapping[right] == -1, dtype=bool)
        step = 1 / len(left_keys)
        tasks = []
        for left_name, right_name in zip(left_keys, right_keys):
            right_name = unique_name(right_name, T.columns)
            for start, end in Config.page_steps(len(mapping)):
                task = Task(
                    _where,
                    T=new_table,
                    mapping=mapping,
                    field="boolean map",
                    left=left_name,
                    right=right_name,
                    new="bmap",
                    start=start,
                    end=end,
                    path=_pid_dir,
                )
                tasks.append(task)

        with task_manager() as tm:
            results = tm.execute(tasks, pbar=ProgressBar())

        for task, result in zip(tasks, results):
            col, start, end = task.gets("left", "start", "end")
            new_table[col][start:end] = result["bmap"][:]

        for name in right_keys:
            del new_table[name]

    else:
        pbar.update(1)

    if _pbar_created_here:
        pbar.close()

    return new_table
