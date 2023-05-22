from tablite.base import Table
import numpy as np
import psutil
import itertools
from tablite.config import Config
from tablite.utils import sub_cls_check, type_check, expression_interpreter
from tablite.mp_utils import filter_ops, shared_memory
from mplite import Task, TaskManager

from tqdm import tqdm as _tqdm


def _filter(T, expression):
    """
    filters based on an expression, such as:

        "all((A==B, C!=4, 200<D))"

    which is interpreted using python's compiler to:

        def _f(A,B,C,D):
            return all((A==B, C!=4, 200<D))
    """
    sub_cls_check(T, Table)
    type_check(expression, str)

    try:
        _f = expression_interpreter(expression, T.columns)
    except Exception as e:
        raise ValueError(f"Expression could not be compiled: {expression}:\n{e}")

    req_columns = [i for i in T.columns if i in expression]
    bitmap = [bool(_f(*r)) for r in T.__getitem__(*req_columns).rows]
    inverse_bitmap = [not i for i in bitmap]

    cls = type(T)
    if len(T) * len(T.columns) < Config.SINGLE_PROCESSING_LIMIT:
        true, false = cls(), cls()
        for col_name in T.columns:
            data = T[col_name][:]
            true[col_name] = list(itertools.compress(data, bitmap))
            false[col_name] = list(itertools.compress(data, inverse_bitmap))
        return true, false
    else:
        mask = np.array(bitmap, dtype=bool)
        return _mp_compress(T, mask), _mp_compress(T, np.invert(mask))  # true, false


def _mp_filter_evaluation_task(c1, c2, bitmap_shm, bitmap_shape, bitmap_ix, start, end, expression):
    """evaluation tasks.

    Args:
        c1 (str): path of a page from column1
        c2 (str): path of a page from column2
        bitmap_shm (shared_memory name): _description_
        bitmap_ix (int): bitmap column index - the index to which the
                         evaluation of expression should be written.
        start (int): bitmap start index
        end (int): bitmap end index
        expression: expression to evaluate
    """
    assert isinstance(expression, dict)
    assert len(expression) == 3
    c1 = expression.get("column1", None)
    c2 = expression.get("column2", None)
    expr = expression.get("criteria", None)
    assert expr in filter_ops

    v1 = expression.get("value1", None)
    v2 = expression.get("value2", None)

    if c1 is not None:
        dset_A = np.load(c1, allow_pickle=True, fix_imports=False)
    else:  # v1 is active:
        dset_A = np.array([v1] * (end - start))

    if c2 is not None:
        dset_B = np.load(c2, allow_pickle=True, fix_imports=False)
    else:  # v2 is active:
        dset_B = np.array([v2] * (end - start))

    # Connect
    existing_shm = shared_memory.SharedMemory(name=bitmap_shm)
    result_array = np.ndarray(bitmap_shape, dtype=np.bool, buffer=existing_shm.buf)
    # Evaluate
    if expr == ">":
        result = dset_A > dset_B
    elif expr == ">=":
        result = dset_A >= dset_B
    elif expr == "==":
        result = dset_A == dset_B
    elif expr == "<":
        result = dset_A < dset_B
    elif expr == "<=":
        result = dset_A <= dset_B
    elif expr == "!=":
        result = dset_A != dset_B
    else:  # it's a python evaluations (slow)
        f = filter_ops.get(expr)
        assert callable(f)
        result = np.array([f(a, b) for a, b in zip(dset_A, dset_B)])
    result_array[bitmap_ix][start:end] = result
    # Disconnect
    existing_shm.close()


def filter(T, expressions, filter_type="all", tqdm=_tqdm):
    """
    enables filtering across columns for multiple criteria.

    expressions:

        str: Expression that can be compiled and executed row by row.
            exampLe: "all((A==B and C!=4 and 200<D))"

        list of dicts: (example):

            L = [
                {'column1':'A', 'criteria': "==", 'column2': 'B'},
                {'column1':'C', 'criteria': "!=", "value2": '4'},
                {'value1': 200, 'criteria': "<", column2: 'D' }
            ]

        accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

    filter_type: 'all' or 'any'
    """
    sub_cls_check(T, Table)
    if isinstance(expressions, str):
        return _filter(expressions)

    if not isinstance(expressions, list) and not isinstance(expressions, tuple):
        raise TypeError

    if len(T) == 0:
        return T.copy(), T.copy()

    for expression in expressions:
        if not isinstance(expression, dict):
            raise TypeError(f"invalid expression: {expression}")
        if not len(expression) == 3:
            raise ValueError(f"expected 3 items, got {expression}")
        x = {"column1", "column2", "criteria", "value1", "value2"}
        if not set(expression.keys()).issubset(x):
            raise ValueError(f"got unknown key: {set(expression.keys()).difference(x)}")

        if expression["criteria"] not in filter_ops:
            raise ValueError(f"criteria missing from {expression}")

        c1 = expression.get("column1", None)
        if c1 is not None and c1 not in T.columns:
            raise ValueError(f"no such column: {c1}")

        v1 = expression.get("value1", None)
        if v1 is not None and c1 is not None:
            raise ValueError("filter can only take 1 left expr element. Got 2.")

        c2 = expression.get("column2", None)
        if c2 is not None and c2 not in T.columns:
            raise ValueError(f"no such column: {c2}")

        v2 = expression.get("value2", None)
        if v2 is not None and c2 is not None:
            raise ValueError("filter can only take 1 right expression element. Got 2.")

    if not isinstance(filter_type, str):
        raise TypeError()
    if filter_type not in {"all", "any"}:
        raise ValueError(f"filter_type: {filter_type} not in ['all', 'any']")

    # EVALUATION....
    # 1. setup a rectangular bitmap for evaluations
    shape = (len(expressions), len(T))
    arr = np.zeros(shape=shape, dtype=bool)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # shm name
    _ = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

    # 2. create tasks for evaluations
    table = {}
    for name, col in T.columns.items():
        table[name] = {}
        for ix, page in enumerate(col.pages):
            table[name][ix] = str(page.path)

    tasks = []
    for bitmap_ix, expression in enumerate(expressions):
        # one expression can AT MOST contain 2 columns.
        c1 = expression.get("column1", None)
        c2 = expression.get("column2", None)
        start = 0
        for end in range(Config.PAGE_SIZE, len(T) + 1, Config.PAGE_SIZE):
            eval_task = Task(_mp_filter_evaluation_task, shm, shape, bitmap_ix, start, end, c1, c2, expression)
            tasks.append(eval_task)
            end = start

    # 3. execute tasks.
    cpus = max(psutil.cpu_count(), 1)
    if cpus < 2 or Config.MULTIPROCESSING_MODE == Config.FALSE:
        for t in tasks:
            r = t.execute()
            if r is not None:
                raise r
            pbar.update(1)
    else:
        with TaskManager(cpus) as tm:
            errs = tm.execute(tasks, pbar=pbar)
            if any(errs):
                raise Exception(errs)

    f = np.all if filter_type == "all" else np.any
    mask = f(arr, axis=1)
    # 4. The mask is now created and is no longer needed.
    shm.close()

    # 5. MERGE...
    trues = type(T)()
    for name in T.columns:
        trues[name] = np.compress(mask, T[name][:])

    falses = type(T)()
    for name in T.columns:
        falses[name] = np.compress(np.invert(mask), T[name][:])
    # 6. RETURN
    return trues, falses


def _select_compress_method(fields):
    """selects method for processing the join

    Args:
        fields (int): number of fields in the join.

    Returns:
        callable: _sp or _mp join.
    """
    type_check(fields, int)
    if psutil.cpu_count() <= 1:
        f = _sp_compress
    elif Config.MULTIPROCESSING_MODE == Config.FALSE:
        f = _sp_compress
    elif Config.MULTIPROCESSING_MODE == Config.FORCE:
        f = _mp_compress
    elif fields < Config.SINGLE_PROCESSING_LIMIT:
        f = _sp_compress
    else:  # use_mp:
        f = _mp_compress
    return f


def all(T, **kwargs):
    """
    returns Table for rows where ALL kwargs match
    :param kwargs: dictionary with headers and values / boolean callable

    Examples:

        t = Table()
        t['a'] = [1,2,3,4]
        t['b'] = [10,20,30,40]

        def f(x):
            return x == 4
        def g(x):
            return x < 20

        t2 = t.any( **{"a":f, "b":g})
        assert [r for r in t2.rows] == [[1, 10], [4, 40]]

        t2 = t.any(a=f,b=g)
        assert [r for r in t2.rows] == [[1, 10], [4, 40]]

        def h(x):
            return x>=2

        def i(x):
            return x<=30

        t2 = t.all(a=h,b=i)
        assert [r for r in t2.rows] == [[2,20], [3, 30]]


    """
    sub_cls_check(T, Table)

    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")
    if not all(k in T.columns for k in kwargs):
        raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in T.columns]}")

    ixs = None
    for k, v in kwargs.items():
        col = T[k][:]
        if ixs is None:  # first header generates base set.
            if callable(v):
                ix2 = {ix for ix, i in enumerate(col) if v(i)}
            else:
                ix2 = {ix for ix, i in enumerate(col) if v == i}

        else:  # remaining headers reduce the base set.
            if callable(v):
                ix2 = {ix for ix in ixs if v(col[ix])}
            else:
                ix2 = {ix for ix in ixs if v == col[ix]}

        if not isinstance(ixs, set):
            ixs = ix2
        else:
            ixs = ixs.intersection(ix2)

        if not ixs:  # There are no matches.
            break

    mask = np.array([True if i in ixs else False for i in range(len(T))], dtype=bool)
    ixs.clear()
    f = _select_compress_method(len(T) * len(T.columns))
    return f(T, mask)


def any(T, **kwargs):
    """
    returns Table for rows where ANY kwargs match
    :param kwargs: dictionary with headers and values / boolean callable
    """
    sub_cls_check(T, Table)
    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")

    ixs = set()
    for k, v in kwargs.items():
        col = T[k][:]
        if callable(v):
            ix2 = {ix for ix, r in enumerate(col) if v(r)}
        else:
            ix2 = {ix for ix, r in enumerate(col) if v == r}
        ixs.update(ix2)

    mask = np.array([True if i in ixs else False for i in range(len(T))], dtype=bool)
    ixs.clear()
    f = _select_compress_method(len(T) * len(T.columns))
    return f(T, mask)


def _sp_compress(T, mask):
    sub_cls_check(T, Table)
    type_check(mask, np.ndarray)

    t = type(T)()
    for col_name in T.columns:
        t[col_name] = np.compress(mask, T[col_name][:])
    return t


def _mp_compress(T, mask):
    """
    helper for `any` and `all` that performs compression of the table self according to mask
    using multiprocessing.
    """
    # NOTE FOR DEVELOPERS:
    # _sp_compress is so fast that the overhead of multiprocessing doesn't pay off.
    return _sp_compress(T, mask)
