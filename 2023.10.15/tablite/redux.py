from tablite.base import BaseTable
import numpy as np
from tablite.utils import sub_cls_check, type_check, expression_interpreter
from tablite.mp_utils import filter_ops
from tablite.datatypes import list_to_np_array
from tablite.config import Config
from tablite.nimlite import filter as _filter_using_list_of_dicts
from tqdm import tqdm as _tqdm


def _filter_using_expression(T, expression):
    """
    filters based on an expression, such as:

        "all((A==B, C!=4, 200<D))"

    which is interpreted using python's compiler to:

        def _f(A,B,C,D):
            return all((A==B, C!=4, 200<D))
    """
    sub_cls_check(T, BaseTable)
    type_check(expression, str)

    try:
        _f = expression_interpreter(expression, list(T.columns))
    except Exception as e:
        raise ValueError(f"Expression could not be compiled: {expression}:\n{e}")

    req_columns = [i for i in T.columns if i in expression]
    return np.array([bool(_f(*r)) for r in T[req_columns].rows], dtype=bool)

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
    sub_cls_check(T, BaseTable)

    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")
    if not all([k in T.columns for k in kwargs]):
        raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in T.columns]}")

    mask = np.full((len(T),), True)
    for k, v in kwargs.items():
        col = T[k]
        for start, end, page in col.iter_by_page():
            data = page.get()
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
    sub_cls_check(T, BaseTable)
    mask = np.full((len(T),), False)
    for name in T.columns:
        col = T[name]
        for start, end, page in col.iter_by_page():
            data = page.get()
            for arg in args:
                mask[start:end] = mask[start:end] | (data == arg)

    mask = np.invert(mask)
    return _compress_one(T, mask)


def filter_any(T, **kwargs):
    """
    returns Table for rows where ANY kwargs match
    :param kwargs: dictionary with headers and values / boolean callable
    """
    sub_cls_check(T, BaseTable)
    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")

    mask = np.full((len(T),), False)
    for k, v in kwargs.items():
        col = T[k]
        for start, end, page in col.iter_by_page():
            data = page.get()
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


def _compress_both(T, mask, pbar: _tqdm):
    # NOTE FOR DEVELOPERS:
    # np.compress is so fast that the overhead of multiprocessing doesn't pay off.
    cls = type(T)
    true, false = cls(), cls()

    pbar_div = (len(T.columns) * len(list(Config.page_steps(len(T)))) - 1)
    pbar_step = (10 / pbar_div) if pbar_div != 0 else 0

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
            pbar.update(pbar_step)
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
    sub_cls_check(T, BaseTable)
    if len(T) == 0:
        return T.copy(), T.copy()

    if isinstance(expressions, str):
        with tqdm(desc="filter", total=20) as pbar:
            # TODO: make parser for expressions and use the nim implement
            mask = _filter_using_expression(T, expressions)
            pbar.update(10)
            res = _compress_both(T, mask, pbar=pbar)
            pbar.update(pbar.total - pbar.n)
    elif isinstance(expressions, list):
        return _filter_using_list_of_dicts(T, expressions, filter_type, tqdm)
    else:
        raise TypeError
        # create new tables

    return res
