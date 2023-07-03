from tablite.base import Table
import numpy as np
from tablite.utils import sub_cls_check, type_check, expression_interpreter
from tablite.mp_utils import filter_ops
from tablite.datatypes import list_to_np_array
from tablite.config import Config
from tqdm import tqdm as _tqdm


def _filter_using_expression(T, expression):
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
        _f = expression_interpreter(expression, list(T.columns))
    except Exception as e:
        raise ValueError(f"Expression could not be compiled: {expression}:\n{e}")

    req_columns = [i for i in T.columns if i in expression]
    return np.array([bool(_f(*r)) for r in T[req_columns].rows], dtype=bool)


def _filter_using_list_of_dicts(T, expressions, filter_type, tqdm=_tqdm):
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
    bitmap = np.empty(shape=(len(expressions), len(T)), dtype=bool)
    # 2. create tasks for evaluations
    for bit_index, expression in enumerate(expressions):
        assert isinstance(expression, dict)
        assert len(expression) == 3
        c1 = expression.get("column1", None)
        c2 = expression.get("column2", None)
        expr = expression.get("criteria", None)
        assert expr in filter_ops
        v1 = expression.get("value1", None)
        v2 = expression.get("value2", None)

        for start, end in Config.page_steps(len(T)):
            if c1 is not None:
                dset_A = T[c1][start:end]
            else:  # v1 is active:
                dset_A = np.array([v1] * (end - start))

            if c2 is not None:
                dset_B = T[c2][start:end]
            else:  # v2 is active:
                dset_B = np.array([v2] * (end - start))

            if len(dset_A) != len(dset_B):
                raise ValueError(
                    f"Assymmetric dataset: {c1} has {len(dset_A)} values, whilst {c2} has {len(dset_B)} values."
                )
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
                result = list_to_np_array([f(a, b) for a, b in zip(dset_A, dset_B)])
            bitmap[bit_index, start:end] = result

    f = np.all if filter_type == "all" else np.any
    mask = f(bitmap, axis=0)
    # 4. The mask is now created and is no longer needed.
    return mask


def filter_all(T, **kwargs):
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
    if not all([k in T.columns for k in kwargs]):
        raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in T.columns]}")

    mask = np.full((len(T),), True)
    for k, v in kwargs.items():
        col = T[k]
        for start, end, data in col.iter_by_page():
            if callable(v):
                vf = np.frompyfunc(v, 1, 1)
                mask[start:end] = mask[start:end] & np.apply_along_axis(vf, 0, data)
            else:
                mask[start:end] = mask[start:end] & (data == v)

    return _compress_one(T, mask)


def drop(T, *args):
    """drops all rows that contain args

    Args:
        T (Table):
    """
    sub_cls_check(T, Table)
    mask = np.full((len(T),), False)
    for name in T.columns:
        col = T[name]
        for start, end, data in col.iter_by_page():
            for arg in args:
                mask[start:end] = mask[start:end] | (data == arg)

    mask = np.invert(mask)
    return _compress_one(T, mask)


def filter_any(T, **kwargs):
    """
    returns Table for rows where ANY kwargs match
    :param kwargs: dictionary with headers and values / boolean callable
    """
    sub_cls_check(T, Table)
    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")

    mask = np.full((len(T),), False)
    for k, v in kwargs.items():
        col = T[k]
        for start, end, data in col.iter_by_page():
            if callable(v):
                vf = np.frompyfunc(v, 1, 1)
                mask[start:end] = mask[start:end] | np.apply_along_axis(vf, 0, data)
            else:
                mask[start:end] = mask[start:end] | (v == data)

    return _compress_one(T, mask)


def _compress_one(T, mask):
    # NOTE FOR DEVELOPERS:
    # np.compress is so fast that the overhead of multiprocessing doesn't pay off.
    cls = type(T)
    new = cls()
    for name in T.columns:
        new.add_columns(name)
        col = new[name]  # fetch the col to avoid doing it in the loop below

        # prevent OOMError by slicing the getitem ops
        for start, end in Config.page_steps(len(T)):
            col.extend(np.compress(mask[start:end], T[name][start:end]))  # <-- getitem ops
    return new


def _compress_both(T, mask):
    # NOTE FOR DEVELOPERS:
    # np.compress is so fast that the overhead of multiprocessing doesn't pay off.
    cls = type(T)
    true, false = cls(), cls()

    for name in T.columns:
        true.add_column(name)
        false.add_column(name)
        true_col = true[name]  # fetch the col to avoid doing it in the loop below
        false_col = false[name]
        # prevent OOMError by slicing the getitem ops
        for start, end in Config.page_steps(len(T)):
            data = T[name][start:end]
            true_col.extend(np.compress(mask[start:end], data))
            false_col.extend(np.compress(np.invert(mask)[start:end], data))
    return true, false


def filter(T, expressions, filter_type="all", tqdm=_tqdm):
    """filters table


    Args:
        T (Table subclass): Table.
        expressions (list or str):
            str:
                filters based on an expression, such as:
                "all((A==B, C!=4, 200<D))"
                which is interpreted using python's compiler to:

                def _f(A,B,C,D):
                    return all((A==B, C!=4, 200<D))

            list of dicts: (example):

            L = [
                {'column1':'A', 'criteria': "==", 'column2': 'B'},
                {'column1':'C', 'criteria': "!=", "value2": '4'},
                {'value1': 200, 'criteria': "<", column2: 'D' }
            ]

        accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

        filter_type (str, optional): Ignored if expressions is str.
            'all' or 'any'. Defaults to "all".
        tqdm (tqdm, optional): progressbar. Defaults to _tqdm.

    Returns:
        2xTables: trues, falses
    """
    # determine method
    sub_cls_check(T, Table)
    if len(T) == 0:
        return T.copy(), T.copy()

    if isinstance(expressions, str):
        mask = _filter_using_expression(T, expressions)
    elif isinstance(expressions, list):
        mask = _filter_using_list_of_dicts(T, expressions, filter_type, tqdm)
    else:
        raise TypeError
    # create new tables
    return _compress_both(T, mask)
