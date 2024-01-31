from collections import defaultdict

from tablite.config import Config
from tablite.base import BaseTable, Column
from tablite.groupby_utils import GroupBy, GroupbyFunction
from tablite.utils import unique_name

from tqdm import tqdm as _tqdm


def groupby(
    T, keys, functions, tqdm=_tqdm, pbar=None
):  # TODO: This is single core code.
    """
    keys: column names for grouping.
    functions: [optional] list of column names and group functions (See GroupyBy class)
    returns: table

    Example:
    ```
    >>> t = Table()
    >>> t.add_column('A', data=[1, 1, 2, 2, 3, 3] * 2)
    >>> t.add_column('B', data=[1, 2, 3, 4, 5, 6] * 2)
    >>> t.add_column('C', data=[6, 5, 4, 3, 2, 1] * 2)
    >>> t.show()
    +=====+=====+=====+
    |  A  |  B  |  C  |
    | int | int | int |
    +-----+-----+-----+
    |    1|    1|    6|
    |    1|    2|    5|
    |    2|    3|    4|
    |    2|    4|    3|
    |    3|    5|    2|
    |    3|    6|    1|
    |    1|    1|    6|
    |    1|    2|    5|
    |    2|    3|    4|
    |    2|    4|    3|
    |    3|    5|    2|
    |    3|    6|    1|
    +=====+=====+=====+
    >>> g = t.groupby(keys=['A', 'C'], functions=[('B', gb.sum)])
    >>> g.show()
    +===+===+===+======+
    | # | A | C |Sum(B)|
    |row|int|int| int  |
    +---+---+---+------+
    |0  |  1|  6|     2|
    |1  |  1|  5|     4|
    |2  |  2|  4|     6|
    |3  |  2|  3|     8|
    |4  |  3|  2|    10|
    |5  |  3|  1|    12|
    +===+===+===+======+
    ```

    Cheat sheet:

    list of unique values
    ```
    >>> g1 = t.groupby(keys=['A'], functions=[])
    >>> g1['A'][:]
    [1,2,3]
    ```
    alternatively:
    ```
    >>> t['A'].unique()
    [1,2,3]
    ```
    list of unique values, grouped by longest combination.
    ```
    >>> g2 = t.groupby(keys=['A', 'B'], functions=[])
    >>> g2['A'][:], g2['B'][:]
    ([1,1,2,2,3,3], [1,2,3,4,5,6])
    ```
    alternatively use:
    ```
    >>> list(zip(*t.index('A', 'B').keys()))
    [(1,1,2,2,3,3) (1,2,3,4,5,6)]
    ```

    A key (unique values) and count hereof.
    ```
    >>> g3 = t.groupby(keys=['A'], functions=[('A', gb.count)])
    >>> g3['A'][:], g3['Count(A)'][:]
    ([1,2,3], [4,4,4])
    ```
    alternatively use:
    ```
    >>> t['A'].histogram()
    ([1,2,3], [4,4,4])
    ```
    for more examples see: https://github.com/root-11/tablite/blob/master/tests/test_groupby.py

    """
    if not isinstance(keys, list):
        raise TypeError("expected keys as a list of column names")

    if keys:
        if len(set(keys)) != len(keys):
            duplicates = [k for k in keys if keys.count(k) > 1]
            s = "" if len(duplicates) > 1 else "s"
            raise ValueError(
                f"duplicate key{s} found across rows and columns: {duplicates}"
            )

    if not isinstance(functions, list):
        raise TypeError(
            f"Expected functions to be a list of tuples. Got {type(functions)}"
        )

    if not keys + functions:
        raise ValueError("No keys or functions?")

    if not all(len(i) == 2 for i in functions):
        raise ValueError(
            f"Expected each tuple in functions to be of length 2. \nGot {functions}"
        )

    if not all(isinstance(a, str) for a, _ in functions):
        L = [(a, type(a)) for a, _ in functions if not isinstance(a, str)]
        raise ValueError(
            f"Expected column names in functions to be strings. Found: {L}"
        )

    if not all(
        issubclass(b, GroupbyFunction) and b in GroupBy.functions for _, b in functions
    ):
        L = [b for _, b in functions if b not in GroupBy._functions]
        if len(L) == 1:
            singular = f"function {L[0]} is not in GroupBy.functions"
            raise ValueError(singular)
        else:
            plural = f"the functions {L} are not in GroupBy.functions"
            raise ValueError(plural)

    # only keys will produce unique values for each key group.
    if keys and not functions:
        cols = list(zip(*T.index(*keys)))
        result = T.__class__()

        pbar = tqdm(total=len(keys), desc="groupby") if pbar is None else pbar

        for col_name, col in zip(keys, cols):
            result[col_name] = col

            pbar.update(1)
        return result

    # grouping is required...
    # 1. Aggregate data.
    aggregation_functions = defaultdict(dict)
    cols = keys + [col_name for col_name, _ in functions]
    seen, L = set(), []
    for c in cols:  # maintains order of appearance.
        if c not in seen:
            seen.add(c)
            L.append(c)

    # there's a table of values.
    data = T[L]
    if isinstance(data, Column):
        tbl = BaseTable()
        tbl[L[0]] = data
    else:
        tbl = data

    pbar = (
        tqdm(desc="groupby", total=len(tbl), disable=Config.TQDM_DISABLE)
        if pbar is None
        else pbar
    )

    for row in tbl.rows:
        d = {col_name: value for col_name, value in zip(L, row)}
        key = tuple([d[k] for k in keys])
        agg_functions = aggregation_functions.get(key)
        if not agg_functions:
            aggregation_functions[key] = agg_functions = [
                (col_name, f()) for col_name, f in functions
            ]
        for col_name, f in agg_functions:
            f.update(d[col_name])

        pbar.update(1)

    # 2. make dense table.
    cols = [[] for _ in cols]
    for key_tuple, funcs in aggregation_functions.items():
        for ix, key_value in enumerate(key_tuple):
            cols[ix].append(key_value)
        for ix, (_, f) in enumerate(funcs, start=len(keys)):
            cols[ix].append(f.value)

    new_names = keys + [f"{f.__name__}({col_name})" for col_name, f in functions]
    result = type(T)()  # New Table.
    for ix, (col_name, data) in enumerate(zip(new_names, cols)):
        revised_name = unique_name(col_name, result.columns)
        result[revised_name] = data
    return result
