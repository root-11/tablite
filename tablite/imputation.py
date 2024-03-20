import math

from tablite.base import BaseTable, Column
from tablite.utils import sub_cls_check, summary_statistics
from tablite.config import Config
from tablite import sort_utils

from tablite.nimlite import nearest_neighbour
from tqdm import tqdm as _tqdm


def imputation(T, targets, missing=None, method="carry forward", sources=None, tqdm=_tqdm, pbar=None):
    """
    In statistics, imputation is the process of replacing missing data with substituted values.

    See more: https://en.wikipedia.org/wiki/Imputation_(statistics)

    Args:
        table (Table): source table.

        targets (str or list of strings): column names to find and
            replace missing values

        missing (None or iterable): values to be replaced.

        method (str): method to be used for replacement. Options:

            'carry forward':
                takes the previous value, and carries forward into fields
                where values are missing.
                +: quick. Realistic on time series.
                -: Can produce strange outliers.

            'mean':
                calculates the column mean (exclude `missing`) and copies
                the mean in as replacement.
                +: quick
                -: doesn't work on text. Causes data set to drift towards the mean.

            'mode':
                calculates the column mode (exclude `missing`) and copies
                the mean in as replacement.
                +: quick
                -: most frequent value becomes over-represented in the sample

            'nearest neighbour':
                calculates normalised distance between items in source columns
                selects nearest neighbour and copies value as replacement.
                +: works for any datatype.
                -: computationally intensive (e.g. slow)

        sources (list of strings): NEAREST NEIGHBOUR ONLY
            column names to be used during imputation.
            if None or empty, all columns will be used.

    Returns:
        table: table with replaced values.
    """
    sub_cls_check(T, BaseTable)

    if isinstance(targets, str) and targets not in T.columns:
        targets = [targets]
    if isinstance(targets, list):
        for name in targets:
            if not isinstance(name, str):
                raise TypeError(f"expected str, not {type(name)}")
            if name not in T.columns:
                raise ValueError(f"target item {name} not a column name in T.columns:\n{T.columns}")
    else:
        raise TypeError("Expected source as list of column names")

    if missing is None:
        missing = {None}
    else:
        missing = set(missing)

    if method == "nearest neighbour":
        if sources in (None, []):
            sources = list(T.columns)
        if isinstance(sources, str):
            sources = [sources]
        if isinstance(sources, list):
            for name in sources:
                if not isinstance(name, str):
                    raise TypeError(f"expected str, not {type(name)}")
                if name not in T.columns:
                    raise ValueError(f"source item {name} not a column name in T.columns:\n{T.columns}")
        else:
            raise TypeError("Expected source as list of column names")

    methods = ["nearest neighbour", "mean", "mode", "carry forward"]

    if method == "carry forward":
        return carry_forward(T, targets, missing, tqdm=tqdm, pbar=pbar)
    elif method in {"mean", "mode"}:
        return stats_method(T, targets, missing, method, tqdm=tqdm, pbar=pbar)
    elif method == "nearest neighbour":
        return nearest_neighbour(T, sources, missing, targets, tqdm=tqdm)
    else:
        raise ValueError(f"method {method} not recognised amonst known methods: {list(methods)})")


def carry_forward(T, targets, missing, tqdm=_tqdm, pbar=None):
    assert isinstance(missing, set)

    if pbar is None:
        total = len(targets) * len(T)
        pbar = tqdm(total=total, desc="imputation.carry_forward", disable=Config.TQDM_DISABLE)

    new = T.copy()
    for name in T.columns:
        if name in targets:
            data = T[name][:]  # create copy
            last_value = None
            for ix, v in enumerate(data):
                if v in missing:  # perform replacement
                    data[ix] = last_value
                else:  # keep last value.
                    last_value = v
                pbar.update(1)
            new[name] = data
        else:
            new[name] = T[name]

    return new


def stats_method(T, targets, missing, method, tqdm=_tqdm, pbar=None):
    assert isinstance(missing, set)

    if pbar is None:
        total = len(targets)
        pbar = tqdm(total=total, desc=f"imputation.{method}", disable=Config.TQDM_DISABLE)

    new = T.copy()
    for name in T.columns:
        if name in targets:
            col = T.columns[name]
            assert isinstance(col, Column)

            hist_values, hist_counts = col.histogram()

            for m in missing:
                try:
                    idx = hist_values.index(m)
                    hist_counts[idx] = 0
                except ValueError:
                    pass

            stats = summary_statistics(hist_values, hist_counts)

            new_value = stats[method]
            col.replace(mapping={m: new_value for m in missing})
            new[name] = col
            pbar.update(1)
        else:
            new[name] = T[name]  # no entropy, keep as is.
        
    return new