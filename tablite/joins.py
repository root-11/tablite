import psutil
import numpy as np
from typing import List, Union
from itertools import product
from pathlib import Path
from tablite.config import Config
from tablite.reindex import reindex
from tablite.merge import where
from tablite.base import BaseTable, Column
from tablite.utils import sub_cls_check, unique_name, type_check
from tablite.mp_utils import select_processing_method
from mplite import Task, TaskManager as _TaskManager
from tqdm import tqdm as _tqdm


def join(
    T: BaseTable,
    other: BaseTable,
    left_keys: List[str],
    right_keys: List[str],
    left_columns: Union[List[str], None],
    right_columns: Union[List[str], None],
    kind: str = "inner",
    merge_keys: bool = False,
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
    
    Example: "inner"
    ```
    SQL:   SELECT number, letter FROM numbers JOIN letters ON numbers.colour == letters.color
    ```
    Tablite: 
    ```
    >>> inner_join = numbers.inner_join(
        letters, 
        left_keys=['colour'], 
        right_keys=['color'], 
        left_columns=['number'], 
        right_columns=['letter']
    )
    ```
    
    Example: "left" 
    ```
    SQL:   SELECT number, letter FROM numbers LEFT JOIN letters ON numbers.colour == letters.color
    ```
    Tablite: 
    ```
    >>> left_join = numbers.left_join(
        letters, 
        left_keys=['colour'], 
        right_keys=['color'], 
        left_columns=['number'], 
        right_columns=['letter']
    )
    ```

    Example: "outer"
    ```
    SQL:   SELECT number, letter FROM numbers OUTER JOIN letters ON numbers.colour == letters.color
    ```

    Tablite: 
    ```
    >>> outer_join = numbers.outer_join(
        letters, 
        left_keys=['colour'], 
        right_keys=['color'], 
        left_columns=['number'], 
        right_columns=['letter']
        )
    ```

    Example: "cross"

    CROSS JOIN returns the Cartesian product of rows from tables in the join.
    In other words, it will produce rows which combine each row from the first table
    with each row from the second table
    """
    if left_columns is None:
        left_columns = list(T.columns)
    if right_columns is None:
        right_columns = list(other.columns)
    assert merge_keys in {True,False}

    _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns)

    fields = len(T)*len(T.columns) + len(other)*len(other.columns)
    m = select_processing_method(fields, sp=_sp_join, mp=_mp_join)

    return m(kind, T,other,left_keys, right_keys, left_columns, right_columns, merge_keys=merge_keys,
             tqdm=tqdm, pbar=pbar)

# fmt:off
def inner_join(T: BaseTable, other: BaseTable, left_keys: List[str], right_keys: List[str], 
              left_columns: Union[List[str], None], right_columns: Union[List[str], None],
              merge_keys: bool = False, tqdm=_tqdm, pbar=None):
    return join(T, other, left_keys, right_keys, left_columns, right_columns, kind="inner", merge_keys=merge_keys, tqdm=tqdm,pbar=pbar)

def left_join(T: BaseTable, other: BaseTable, left_keys: List[str], right_keys: List[str], 
              left_columns: Union[List[str], None], right_columns: Union[List[str], None],
              merge_keys: bool = False, tqdm=_tqdm, pbar=None):
    return join(T, other, left_keys, right_keys, left_columns, right_columns, kind="left", merge_keys=merge_keys, tqdm=tqdm,pbar=pbar)

def outer_join(T: BaseTable, other: BaseTable, left_keys: List[str], right_keys: List[str], 
              left_columns: Union[List[str], None], right_columns: Union[List[str], None],
              merge_keys: bool = False, tqdm=_tqdm, pbar=None):
    return join(T, other, left_keys, right_keys, left_columns, right_columns, kind="outer", merge_keys=merge_keys, tqdm=tqdm,pbar=pbar)

def cross_join(T: BaseTable, other: BaseTable, left_keys: List[str], right_keys: List[str], 
              left_columns: Union[List[str], None], right_columns: Union[List[str], None],
              merge_keys: bool = False, tqdm=_tqdm, pbar=None):
    return join(T, other, left_keys, right_keys, left_columns, right_columns, kind="cross", merge_keys=merge_keys, tqdm=tqdm,pbar=pbar)
# fmt: on


def _vpus(tasks):
    """private helper to determine how many VPUs there is memory for.

    Args:
        tasks (list): list of tasks

    Returns:
        integer: number of VPUs
    """
    if Config.MULTIPROCESSING_MODE == Config.FALSE:
        raise TypeError("Config.MULTIPROCESSING_MODE == Config.FALSE")
    else:
        memory_per_join = 300e6
        max_vpus = psutil.virtual_memory().free // memory_per_join
        return int(min(Config.vpus, len(tasks), max_vpus))


def _jointype_check(T, other, left_keys, right_keys, left_columns, right_columns):
    sub_cls_check(T, BaseTable)
    sub_cls_check(other, BaseTable)

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


# -------------------------
# SINGLE PROCESSING SECTION
# -------------------------


def _sp_left_mapping(T, other, left_keys, right_keys, tqdm, pbar):
    """
    Args:
        T (Table): left table
        other (Table): right table
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
    
    Returns: 
        Table: joined table
    """
    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    _left, _right = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)

    _left, _right = np.array(_left, dtype=int), np.array(_right, dtype=int)
    return _left,_right


def _sp_inner_mapping(T, other, left_keys, right_keys, tqdm, pbar):
    """
    Args:
        T (Table): left table
        other (Table): right table
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
    
    Returns: 
        Table: joined table

    """
    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    _left, _right = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, None)
        if right_ixs is None:
            continue
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)

    _left, _right = np.array(_left, dtype=int), np.array(_right, dtype=int)
    return _left, _right


def _sp_outer_mapping(T, other, left_keys, right_keys, tqdm, pbar):
    """
    Args:
        T (Table): left table
        other (Table): right table
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
    
    Returns: 
        Table: joined table

    """
    left_index = T.index(*left_keys)
    right_index = other.index(*right_keys)
    _left, _right, _right_unused = [], [], set(right_index.keys())
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)
                _right_unused.discard(left_key)

    for right_key in _right_unused:
        for right_ix in right_index[right_key]:
            _left.append(-1)
            _right.append(right_ix)

    _left, _right = np.array(_left, dtype=int), np.array(_right, dtype=int)
    return _left, _right


def _sp_cross_mapping(T, other, left_keys, right_keys, tqdm, pbar):
    """
    Args:
        T (Table): left table
        other (Table): right table
        left_keys (list): list of keys for the join from left table.
        right_keys (list): list of keys for the join from right table.
    
    Returns: 
        Table: joined table

    """
    _left, _right = zip(*product(range(len(T)), range(len(other))))
    _left, _right = np.array(_left, dtype=int), np.array(_right, dtype=int)
    return _left,_right


_sp_mapping_methods = {
    "inner": _sp_inner_mapping,
    "left":  _sp_left_mapping, 
    "outer": _sp_outer_mapping,
    "cross": _sp_cross_mapping,
}

def _sp_join(kind, T, other, left_keys, right_keys, left_columns, right_columns, merge_keys=None, tqdm=_tqdm, pbar=None):
    _mapping = _sp_mapping_methods.get(kind, None)
    if _mapping is None:
        raise ValueError(f"join type unknown: {kind}")
    
    _left,_right = _mapping(T, other, left_keys, right_keys, tqdm=tqdm, pbar=pbar)
    assert len(_left) == len(_right)

    if pbar is None:
        total = len(left_columns) + len(right_columns)
        pbar = tqdm(total=total, desc="join", disable=Config.TQDM_DISABLE)

    result = reindex(T, _left, left_columns, tqdm=tqdm, pbar=pbar)
    second = reindex(other, _right, right_columns, tqdm=tqdm, pbar=pbar)
    for name in right_columns:
        revised_name = unique_name(name, result.columns)
        result[revised_name] = second[name]

    if merge_keys is True:
        result = _merge_keys(kind, T, result, _left,_right, left_keys, right_keys)
    return result

def _merge_keys(kind, T, result, left, right, left_keys, right_keys):
    if kind in ["inner", "cross"]:
        for right_name in right_keys:
            right_name = unique_name(right_name, T.columns)
            if right_name in result.columns:
                del result[right_name]
    else:
        if kind == "outer":
            boolean_map = (left != -1)
        elif kind == "left":
            boolean_map = (right == -1)
        else:
            raise TypeError(f"bad join type: {kind}")
        
        for left_name, right_name in zip(left_keys,right_keys):
            right_name = unique_name(right_name, T.columns)
            result = where(result, boolean_map, left_name, right_name, new=left_name)
    return result


# -----------------------
# MULTIPROCESSING SECTION
# -----------------------


def _mp_where(
    T: BaseTable,
    mapping: BaseTable,
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
    Constr = type(T)
    criteria = mapping[field][start:end]
    left_values = T[left][start:end]
    right_values = T[right][start:end]
    new_values = np.where(criteria, left_values, right_values)
    return Constr({new: new_values}, _path=path)


def _mp_left_mapping(
    T: BaseTable,
    other: BaseTable,
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
    del left, right

    _left, _right = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)

    Constr = type(T)
    mapping = Constr({"left": _left, "right": _right}, _path=path)
    return mapping

def _mp_inner_mapping(
    T: BaseTable,
    other: BaseTable,
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
    del left, right

    _left, _right = [], []
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, None)
        if right_ixs is None:
            continue
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)

    Constr = type(T)
    mapping = Constr({"left": _left, "right": _right}, _path=path)
    return mapping


def _mp_outer_mapping(
    T: BaseTable,
    other: BaseTable,
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
    del left, right

    _left, _right, _right_unused = [], [], set(right_index.keys())
    for left_key, left_ixs in left_index.items():
        right_ixs = right_index.get(left_key, (-1,))
        for left_ix in left_ixs:
            for right_ix in right_ixs:
                _left.append(left_ix)
                _right.append(right_ix)
                _right_unused.discard(left_key)

    for right_key in _right_unused:
        for right_ix in right_index[right_key]:
            _left.append(-1)
            _right.append(right_ix)

    Constr = type(T)
    mapping = Constr({"left": _left, "right": _right}, _path=path)
    return mapping


def _mp_cross_mapping(
    T: BaseTable,
    other: BaseTable,
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
    Constr = type(T)
    rr = np.arange(right_slice.start, min(right_slice.stop, len(other)))
    tmp = Constr({"right": rr}, _path=path)
    right = tmp["right"]
    rr_shape = rr.shape
    del rr

    lr = np.arange(left_slice.start, min(left_slice.stop, len(T)))
    mapping = Constr({"left": [], "right": []}, _path=path)
    for a in lr:
        mapping["right"].extend(right)  
        # by using the right filepointer above, the page-id is 
        # repeated, costing 0 Mb in storage.
        mapping["left"].extend(np.full(rr_shape, a))
    return mapping


def _mp_reindex_page(
    T: BaseTable,
    column_name: str,
    mapping: BaseTable,
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
    if ix_arr[0]==start and ix_arr[-1] == end-1 and np.all(ix_arr == np.arange(start,end)):  
        array = T[column_name][part]
    else:
        array = T[column_name].get_by_indices(ix_arr)
        # in the array, the index of -1 will be wrong.
        # so if there is any -1 in the indices, they will
        # have to be replaced with Nones
        mask = ix_arr == -1
        if np.any(mask):
            nones = np.full(ix_arr.shape, fill_value=None)
            array = np.where(mask, nones, array)


    Constr = type(T)
    remapped_T = Constr({column_name: array}, _path=path)

    return remapped_T


def _gets(task, *args):
    """helper to get kwargs of a task

    *Args:
        names from kwargs to retrieve.

    Returns:
        tuple: tuple with kw-values in same order as args

    Examples:

    Verbose way:
    ```
    >>> col = task.kwarg.get("left")
    >>> right = task.kwarg.get("right")
    >>> end = task.kwarg.get("end")
    ```
    Compact way:
    ```
    >>> col, start, end = task.gets("left", "start", "end")
    ```
    """
    result = tuple()
    for arg in args:
        result += (task.kwargs.get(arg),)
    return result


_mp_mapping_methods = {
    'left': _mp_left_mapping,
    'inner': _mp_inner_mapping,
    'outer': _mp_outer_mapping,
    'cross': _mp_cross_mapping
}


def _mp_join(
        kind: str,
        T: BaseTable,
        other: BaseTable,
        left_keys: List[str],
        right_keys: List[str],
        left_columns: Union[List[str], None] = None,
        right_columns: Union[List[str], None] = None,
        merge_keys: bool = False,
        tqdm=_tqdm,
        pbar=None,
        TaskManager=None,
    ):

    Constr = type(T)

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
    if TaskManager is None:
        TaskManager = _TaskManager
    
    tasks = []

    _pid_dir = Path(Config.workdir) / Config.pid

    # step 1: create mapping tasks
    _mapping = _mp_mapping_methods.get(kind)
    if _mapping is None:
        raise ValueError(f"join type unknown: {kind}")

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

    with TaskManager(cpu_count=_vpus(tasks)) as tm:
        results = tm.execute(tasks, pbar=ProgressBar())
        errors = [s for s in results if isinstance(s, str)]

        if len(errors) > 0:
            raise Exception(errors[0])

    # step 2: assemble mapping from tasks
    mapping = Constr({"left": [], "right": []})
    for result in results:
        assert isinstance(result, BaseTable)
        mapping += result

    pbar.update(1)

    # step 3: initiate reindexing tasks
    tasks = []
    names = []  # will store (old name, new name) for derefences during assemble.
    new_table = Constr()
    n = len(mapping)
    for name in left_columns:
        new_table.add_column(name)

        for start in range(0, n + 1, step):
            names.append((name, name))
            task = Task(
                _mp_reindex_page,
                T=T,
                column_name=name,
                mapping=mapping,
                index="left",
                start=start,
                end=min(start + step, n),
                path=_pid_dir,
            )
            tasks.append(task)

    for name in right_columns:
        new_name = unique_name(name, new_table.columns)
        new_table.add_column(new_name)

        for start in range(0, n + 1, step):
            names.append((new_name, name))
            task = Task(
                _mp_reindex_page,
                T=other,
                column_name=name,
                mapping=mapping,
                index="right",
                start=start,
                end=min(start + step, n),
                path=_pid_dir,
            )
            tasks.append(task)

    with TaskManager(cpu_count=_vpus(tasks)) as tm:
        results = tm.execute(tasks, pbar=ProgressBar())
        errors = [s for s in results if isinstance(s, str)]

        if len(errors) > 0:
            raise Exception(errors[0])

    # step 4: assemble the result
    for task, result, (new, old) in zip(tasks, results, names):
        arr = result[old]
        new_table[new].extend(arr)

    pbar.n = pbar.total - 1  # needed to overcome floating point error.
    pbar.refresh()

    # step 5: merge keys (if required)
    if merge_keys is True:
        if kind in ["outer", "left"]:
            mapping["boolean map"] = np.array(mapping["left"]) != -1 \
                if kind == "outer" else np.array(mapping["right"]) == -1

            step = 1 / len(left_keys)
            tasks = []
            for left_name, right_name in zip(left_keys, right_keys):
                right_name = unique_name(right_name, T.columns)
                for start, end in Config.page_steps(len(mapping)):
                    task = Task(
                        _mp_where,
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

            with TaskManager(cpu_count=_vpus(tasks)) as tm:
                results = tm.execute(tasks, pbar=ProgressBar())
                errors = [s for s in results if isinstance(s, str)]

                if len(errors) > 0:
                    raise Exception(errors[0])

            for task, result in zip(tasks, results):
                col, start, end = _gets(task, "left", "start", "end")
                new_table[col][start:end] = result["bmap"][:]
        elif kind not in ["inner", "cross"]:
            raise TypeError(f"bad join type: {kind}")

        for right_name in right_keys:
            del new_table[unique_name(right_name, T.columns)]
    else:
        pbar.update(1)

    if _pbar_created_here:
        pbar.close()

    return new_table
