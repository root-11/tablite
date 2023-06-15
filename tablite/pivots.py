from collections import defaultdict
import numpy as np

from tablite.base import Table
from tablite.utils import unique_name, sub_cls_check
from tablite.groupbys import groupby
from tablite.config import Config

from tqdm import tqdm as _tqdm


def pivot(T, rows, columns, functions, values_as_rows=True, tqdm=_tqdm, pbar=None):
    """
    param: rows: column names to keep as rows
    param: columns: column names to keep as columns
    param: functions: aggregation functions from the Groupby class as

    example:

    t.show()
    # +=====+=====+=====+
    # |  A  |  B  |  C  |
    # | int | int | int |
    # +-----+-----+-----+
    # |    1|    1|    6|
    # |    1|    2|    5|
    # |    2|    3|    4|
    # |    2|    4|    3|
    # |    3|    5|    2|
    # |    3|    6|    1|
    # |    1|    1|    6|
    # |    1|    2|    5|
    # |    2|    3|    4|
    # |    2|    4|    3|
    # |    3|    5|    2|
    # |    3|    6|    1|
    # +=====+=====+=====+

    t2 = t.pivot(rows=['C'], columns=['A'], functions=[('B', gb.sum)])
    t2.show()
    # +===+===+========+=====+=====+=====+
    # | # | C |function|(A=1)|(A=2)|(A=3)|
    # |row|int|  str   |mixed|mixed|mixed|
    # +---+---+--------+-----+-----+-----+
    # |0  |  6|Sum(B)  |    2|None |None |
    # |1  |  5|Sum(B)  |    4|None |None |
    # |2  |  4|Sum(B)  |None |    6|None |
    # |3  |  3|Sum(B)  |None |    8|None |
    # |4  |  2|Sum(B)  |None |None |   10|
    # |5  |  1|Sum(B)  |None |None |   12|
    # +===+===+========+=====+=====+=====+

    """
    sub_cls_check(T, Table)

    if isinstance(rows, str):
        rows = [rows]
    if not all(isinstance(i, str) for i in rows):
        raise TypeError(f"Expected rows as a list of column names, not {[i for i in rows if not isinstance(i,str)]}")

    if isinstance(columns, str):
        columns = [columns]
    if not all(isinstance(i, str) for i in columns):
        raise TypeError(
            f"Expected columns as a list of column names, not {[i for i in columns if not isinstance(i, str)]}"
        )

    if not isinstance(values_as_rows, bool):
        raise TypeError(f"expected sum_on_rows as boolean, not {type(values_as_rows)}")

    keys = rows + columns
    assert isinstance(keys, list)

    extra_steps = 2

    if pbar is None:
        total = extra_steps

        if len(functions) == 0:
            total = total + len(keys)
        else:
            total = total + len(T)

        pbar = tqdm(total=total, desc="pivot")

    grpby = groupby(T, keys, functions, tqdm=tqdm, pbar=pbar)

    if len(grpby) == 0:  # return empty table. This must be a test?
        pbar.update(extra_steps)
        return Table()

    # split keys to determine grid dimensions
    row_key_index = {}
    col_key_index = {}

    r = len(rows)
    c = len(columns)
    g = len(functions)

    records = defaultdict(dict)

    for row in grpby.rows:
        row_key = tuple(row[:r])
        col_key = tuple(row[r : r + c])
        func_key = tuple(row[r + c :])

        if row_key not in row_key_index:
            row_key_index[row_key] = len(row_key_index)  # Y

        if col_key not in col_key_index:
            col_key_index[col_key] = len(col_key_index)  # X

        rix = row_key_index[row_key]
        cix = col_key_index[col_key]
        if cix in records:
            if rix in records[cix]:
                raise ValueError("this should be empty.")
        records[cix][rix] = func_key

    pbar.update(1)
    result = type(T)()

    if values_as_rows:  # ---> leads to more rows.
        # first create all columns left to right

        n = r + 1  # rows keys + 1 col for function values.
        cols = [[] for _ in range(n)]
        for row, ix in row_key_index.items():
            for col_name, f in functions:
                cols[-1].append(f"{f.__name__}({col_name})")
                for col_ix, v in enumerate(row):
                    cols[col_ix].append(v)

        for col_name, values in zip(rows + ["function"], cols):
            col_name = unique_name(col_name, result.columns)
            result[col_name] = values
        col_length = len(cols[0])
        cols.clear()

        # then populate the sparse matrix.
        for col_key, c in col_key_index.items():
            col_name = "(" + ",".join([f"{col_name}={value}" for col_name, value in zip(columns, col_key)]) + ")"
            col_name = unique_name(col_name, result.columns)
            L = [None for _ in range(col_length)]
            for r, funcs in records[c].items():
                for ix, f in enumerate(funcs):
                    L[g * r + ix] = f
            result[col_name] = L

    else:  # ---> leads to more columns.
        n = r
        cols = [[] for _ in range(n)]
        for row in row_key_index:
            for col_ix, v in enumerate(row):
                cols[col_ix].append(v)  # write key columns.

        for col_name, values in zip(rows, cols):
            result[col_name] = values

        col_length = len(row_key_index)

        # now populate the sparse matrix.
        for col_key, c in col_key_index.items():  # select column.
            cols, names = [], []

            for f, v in zip(functions, func_key):
                agg_col, func = f
                terms = ",".join([agg_col] + [f"{col_name}={value}" for col_name, value in zip(columns, col_key)])
                col_name = f"{func.__name__}({terms})"
                col_name = unique_name(col_name, result.columns)
                names.append(col_name)
                cols.append([None for _ in range(col_length)])
            for r, funcs in records[c].items():
                for ix, f in enumerate(funcs):
                    cols[ix][r] = f
            for name, col in zip(names, cols):
                result[name] = col

    pbar.update(1)

    return result


def transpose(T, tqdm=_tqdm):
    """performs a CCW matrix rotation of the table."""
    sub_cls_check(T, Table)

    if len(T.columns) == 0:
        return type(T)()

    assert isinstance(T, Table)
    new = type(T)()
    L = list(T.columns)
    new[L[0]] = L[1:]
    for row in tqdm(T.rows, desc="table transpose"):
        new[row[0]] = row[1:]
    return new


def pivot_transpose(T, columns, keep=None, column_name="transpose", value_name="value", tqdm=_tqdm):
    """Transpose a selection of columns to rows.

    Args:
        columns (list of column names): column names to transpose
        keep (list of column names): column names to keep (repeat)

    Returns:
        Table: with columns transposed to rows

    Example:
        transpose columns 1,2 and 3 and transpose the remaining columns, except `sum`.

    Input:

    | col1 | col2 | col3 | sun | mon | tue | ... | sat | sum  |
    |------|------|------|-----|-----|-----|-----|-----|------|
    | 1234 | 2345 | 3456 | 456 | 567 |     | ... |     | 1023 |
    | 1244 | 2445 | 4456 |     |   7 |     | ... |     |    7 |
    | ...  |      |      |     |     |     |     |     |      |

    t.transpose(keep=[col1, col2, col3], transpose=[sun,mon,tue,wed,thu,fri,sat])`

    Output:

    |col1| col2| col3| transpose| value|
    |----|-----|-----|----------|------|
    |1234| 2345| 3456| sun      |   456|
    |1234| 2345| 3456| mon      |   567|
    |1244| 2445| 4456| mon      |     7|

    """
    sub_cls_check(T, Table)

    if not isinstance(columns, list):
        raise TypeError
    for i in columns:
        if not isinstance(i, str):
            raise TypeError
        if i not in T.columns:
            raise ValueError

    if keep is None:
        keep = []
    for i in keep:
        if not isinstance(i, str):
            raise TypeError
        if i not in T.columns:
            raise ValueError

    if column_name in keep + columns:
        column_name = unique_name(column_name, set_of_names=keep + columns)
    if value_name in keep + columns + [column_name]:
        value_name = unique_name(value_name, set_of_names=keep + columns)

    new = type(T)()
    new.add_columns(*keep + [column_name, value_name])
    news = {name: [] for name in new.columns}

    n = len(keep)

    with tqdm(total=len(T), desc="transpose", disable=Config.TQDM_DISABLE) as pbar:
        for ix, row in enumerate(T[keep + columns].rows, start=1):
            keeps = row[:n]
            transposes = row[n:]

            for name, value in zip(keep, keeps):
                news[name].extend([value] * len(transposes))
            for name, value in zip(columns, transposes):
                news[column_name].append(name)
                news[value_name].append(value)

            if ix % Config.SINGLE_PROCESSING_LIMIT == 0:
                for name, values in news.items():
                    new[name].extend(values)
                    values.clear()

            pbar.update(1)

    for name, values in news.items():
        new[name].extend(np.array(values))
        values.clear()
    return new
