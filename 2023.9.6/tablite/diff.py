import numpy as np
from tablite.config import Config
from tablite.base import BaseTable
from tablite.utils import sub_cls_check, unique_name
import difflib


def diff(T, other, columns=None):
    """compares table self with table other

    Args:
        self (Table): Table
        other (Table): Table
        columns (List, optional): list of column names to include in comparison. Defaults to None.

    Returns:
        Table: diff of self and other with diff in columns 1st and 2nd.
    """
    sub_cls_check(T, BaseTable)
    sub_cls_check(other, BaseTable)
    if columns is None:
        columns = [name for name in T.columns if name in other.columns]
    elif isinstance(columns, list) and all(isinstance(i, str) for i in columns):
        for name in columns:
            if name not in T.columns:
                raise ValueError(f"column '{name}' not found")
            if name not in other.columns:
                raise ValueError(f"column '{name}' not found")
    else:
        raise TypeError("Expected list of column names")

    t1 = T[columns]
    if issubclass(type(t1), BaseTable):
        t1 = [tuple(r) for r in T.rows]
    else:
        t1 = list(T)
    t2 = other[columns]
    if issubclass(type(t2), BaseTable):
        t2 = [tuple(r) for r in other.rows]
    else:
        t2 = list(other)

    sm = difflib.SequenceMatcher(None, t1, t2)
    new = type(T)()
    first = unique_name("1st", columns)
    second = unique_name("2nd", columns)
    new.add_columns(*columns + [first, second])

    news = {n: [] for n in new.columns}  # Cache for Work in progress.

    for opc, t1a, t1b, t2a, t2b in sm.get_opcodes():
        if opc == "insert":
            for name, col in zip(columns, zip(*t2[t2a:t2b])):
                news[name].extend(col)
            news[first] += ["-"] * (t2b - t2a)
            news[second] += ["+"] * (t2b - t2a)

        elif opc == "delete":
            for name, col in zip(columns, zip(*t1[t1a:t1b])):
                news[name].extend(col)
            news[first] += ["+"] * (t1b - t1a)
            news[second] += ["-"] * (t1b - t1a)

        elif opc == "equal":
            for name, col in zip(columns, zip(*t2[t2a:t2b])):
                news[name].extend(col)
            news[first] += ["="] * (t2b - t2a)
            news[second] += ["="] * (t2b - t2a)

        elif opc == "replace":
            for name, col in zip(columns, zip(*t2[t2a:t2b])):
                news[name].extend(col)
            news[first] += ["r"] * (t2b - t2a)
            news[second] += ["r"] * (t2b - t2a)

        else:
            pass

        # Clear cache to free up memory.
        if len(news[first]) > Config.PAGE_SIZE == 0:
            for name, L in news.items():
                new[name].extend(np.array(L))
                L.clear()

    for name, L in news.items():
        new[name].extend(np.array(L))
        L.clear()
    return new
