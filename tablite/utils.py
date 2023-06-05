from collections import defaultdict
import math
import ast
from datetime import datetime, date, time, timedelta, timezone  # noqa
from itertools import compress


def type_check(var, kind):
    if not isinstance(var, kind):
        raise TypeError(f"Expected {kind}, not {type(var)}")


def sub_cls_check(c, kind):
    if not issubclass(type(c), kind):
        raise TypeError(f"Expected {kind}, not {type(c)}")


def unique_name(wanted_name, set_of_names):
    """
    returns a wanted_name as wanted_name_i given a list of names
    which guarantees unique naming.
    """
    if not isinstance(set_of_names, set):
        set_of_names = set(set_of_names)
    name, i = wanted_name, 1
    while name in set_of_names:
        name = f"{wanted_name}_{i}"
        i += 1
    return name


def expression_interpreter(expression, columns):
    """
    Interprets valid expressions such as:

        "all((A==B, C!=4, 200<D))"

    as:
        def _f(A,B,C,D):
            return all((A==B, C!=4, 200<D))

    using python's compiler.
    """
    if not isinstance(expression, str):
        raise TypeError(f"`{expression}` is not a str")
    if not isinstance(columns, list):
        raise TypeError
    if not all(isinstance(i, str) for i in columns):
        raise TypeError

    req_columns = ", ".join(i for i in columns if i in expression)
    script = f"def f({req_columns}):\n    return {expression}"
    tree = ast.parse(script)
    code = compile(tree, filename="blah", mode="exec")
    namespace = {}
    exec(code, namespace)
    f = namespace["f"]
    if not callable(f):
        raise ValueError(f"The expression could not be parse: {expression}")
    return f


def intercept(A, B):
    """Enables calculation of the intercept of two range objects.
    Used to determine if a datablock contains a slice.

    Args:
        A: range
        B: range

    Returns:
        range: The intercept of ranges A and B.
    """
    type_check(A, range)
    type_check(B, range)

    if A.step < 1:
        A = range(A.stop + 1, A.start + 1, 1)
    if B.step < 1:
        B = range(B.stop + 1, B.start + 1, 1)

    if len(A) == 0:
        return range(0)
    if len(B) == 0:
        return range(0)

    if A.stop <= B.start:
        return range(0)
    if A.start >= B.stop:
        return range(0)

    if A.start <= B.start:
        if A.stop <= B.stop:
            start, end = B.start, A.stop
        elif A.stop > B.stop:
            start, end = B.start, B.stop
        else:
            raise ValueError("bad logic")
    elif A.start < B.stop:
        if A.stop <= B.stop:
            start, end = A.start, A.stop
        elif A.stop > B.stop:
            start, end = A.start, B.stop
        else:
            raise ValueError("bad logic")
    else:
        raise ValueError("bad logic")

    a_steps = math.ceil((start - A.start) / A.step)
    a_start = (a_steps * A.step) + A.start

    b_steps = math.ceil((start - B.start) / B.step)
    b_start = (b_steps * B.step) + B.start

    if A.step == 1 or B.step == 1:
        start = max(a_start, b_start)
        step = max(A.step, B.step)
        return range(start, end, step)
    elif A.step == B.step:
        a, b = min(A.start, B.start), max(A.start, B.start)
        if (b - a) % A.step != 0:  # then the ranges are offset.
            return range(0)
        else:
            return range(b, end, step)
    else:
        # determine common step size:
        step = max(A.step, B.step) if math.gcd(A.step, B.step) != 1 else A.step * B.step
        # examples:
        # 119 <-- 17 if 1 != 1 else 119 <-- max(7, 17) if math.gcd(7, 17) != 1 else 7 * 17
        #  30 <-- 30 if 3 != 1 else 90 <-- max(3, 30) if math.gcd(3, 30) != 1 else 3*30
        if A.step < B.step:
            for n in range(a_start, end, A.step):  # increment in smallest step to identify the first common value.
                if n < b_start:
                    continue
                elif (n - b_start) % B.step == 0:
                    return range(n, end, step)  # common value found.
        else:
            for n in range(b_start, end, B.step):
                if n < a_start:
                    continue
                elif (n - a_start) % A.step == 0:
                    return range(n, end, step)

        return range(0)


# This list is the contract:
required_keys = {
    "min",
    "max",
    "mean",
    "median",
    "stdev",
    "mode",
    "distinct",
    "iqr_low",
    "iqr_high",
    "iqr",
    "sum",
    "summary type",
    "histogram",
}


def summary_statistics(values, counts):
    """
    values: any type
    counts: integer

    returns dict with:
    - min (int/float, length of str, date)
    - max (int/float, length of str, date)
    - mean (int/float, length of str, date)
    - median (int/float, length of str, date)
    - stdev (int/float, length of str, date)
    - mode (int/float, length of str, date)
    - distinct (number of distinct values)
    - iqr (int/float, length of str, date)
    - sum (int/float, length of str, date)
    - histogram (2 arrays: values, count of each values)
    """
    # determine the dominant datatype:
    dtypes = defaultdict(int)
    most_frequent, most_frequent_dtype = 0, int
    for v, c in zip(values, counts):
        dtype = type(v)
        total = dtypes[dtype] + c
        dtypes[dtype] = total
        if total > most_frequent:
            most_frequent_dtype = dtype
            most_frequent = total

    if most_frequent == 0:
        return {}

    most_frequent_dtype = max(dtypes, key=dtypes.get)
    mask = [type(v) == most_frequent_dtype for v in values]
    v = list(compress(values, mask))
    c = list(compress(counts, mask))

    f = summary_methods.get(most_frequent_dtype, int)
    result = f(v, c)
    result["distinct"] = len(values)
    result["summary type"] = most_frequent_dtype.__name__
    result["histogram"] = [values, counts]
    assert set(result.keys()) == required_keys, "Key missing!"
    return result


def _numeric_statistics_summary(v, c):
    VC = [[v, c] for v, c in zip(v, c)]
    VC.sort()

    total_val, mode, median, total_cnt = 0, None, None, sum(c)

    max_cnt, cnt_n = -1, 0
    mn, cstd = 0, 0.0
    iqr25 = total_cnt * 1 / 4
    iqr50 = total_cnt * 1 / 2
    iqr75 = total_cnt * 3 / 4
    iqr_low, iqr_high = 0, 0
    vx_0 = None
    vmin, vmax = VC[0][0], VC[-1][0]

    for vx, cx in VC:
        cnt_0 = cnt_n
        cnt_n += cx

        if cnt_0 < iqr25 < cnt_n:  # iqr 25%
            iqr_low = vx
        elif cnt_0 == iqr25:
            _, delta = divmod(1 * (total_cnt - 1), 4)
            iqr_low = (vx_0 * (4 - delta) + vx * delta) / 4

        # median calculations
        if cnt_n - cx < iqr50 < cnt_n:
            median = vx
        elif cnt_0 == iqr50:
            _, delta = divmod(2 * (total_cnt - 1), 4)
            median = (vx_0 * (4 - delta) + vx * delta) / 4

        if cnt_0 < iqr75 < cnt_n:  # iqr 75%
            iqr_high = vx
        elif cnt_0 == iqr75:
            _, delta = divmod(3 * (total_cnt - 1), 4)
            iqr_high = (vx_0 * (4 - delta) + vx * delta) / 4

        # stdev calulations
        dt = cx * (vx - mn)  # dt = value - self.mean
        mn += dt / cnt_n  # self.mean += dt / self.count
        cstd += dt * (vx - mn)  # self.c += dt * (value - self.mean)

        # mode calculations
        if cx > max_cnt:
            mode, max_cnt = vx, cx

        total_val += vx * cx
        vx_0 = vx

    var = cstd / (cnt_n - 1) if cnt_n > 1 else 0
    stdev = var ** (1 / 2) if cnt_n > 1 else 0

    d = {
        "min": vmin,
        "max": vmax,
        "mean": total_val / (total_cnt if total_cnt >= 1 else None),
        "median": median,
        "stdev": stdev,
        "mode": mode,
        "iqr_low": iqr_low,
        "iqr_high": iqr_high,
        "iqr": iqr_high - iqr_low,
        "sum": total_val,
    }
    return d


def _none_type_summary(v, c):
    return {k: "n/a" for k in required_keys}


def _boolean_statistics_summary(v, c):
    v = [int(vx) for vx in v]
    d = _numeric_statistics_summary(v, c)
    for k, v in d.items():
        if k in {"mean", "stdev", "sum", "iqr_low", "iqr_high", "iqr"}:
            continue
        elif v == 1:
            d[k] = True
        elif v == 0:
            d[k] = False
        else:
            pass
    return d


def _timedelta_statistics_summary(v, c):
    v = [vx.days + v.seconds / (24 * 60 * 60) for vx in v]
    d = _numeric_statistics_summary(v, c)
    for k in d.keys():
        d[k] = timedelta(d[k])
    return d


def _datetime_statistics_summary(v, c):
    v = [vx.timestamp() for vx in v]
    d = _numeric_statistics_summary(v, c)
    for k in d.keys():
        if k in {"stdev", "iqr", "sum"}:
            d[k] = f"{d[k]/(24*60*60)} days"
        else:
            d[k] = datetime.fromtimestamp(d[k])
    return d


def _time_statistics_summary(v, c):
    v = [sum(t.hour * 60 * 60, t.minute * 60, t.second, t.microsecond / 1e6) for t in v]
    d = _numeric_statistics_summary(v, c)
    for k in d.keys():
        if k in {"min", "max", "mean", "median"}:
            timestamp = d[k]
            hours = timestamp // (60 * 60)
            timestamp -= hours * 60 * 60
            minutes = timestamp // 60
            timestamp -= minutes * 60
            d[k] = time.fromtimestamp(hours, minutes, timestamp)
        elif k in {"stdev", "iqr", "sum"}:
            d[k] = f"{d[k]} seconds"
        else:
            pass
    return d


def _date_statistics_summary(v, c):
    v = [datetime(d.year, d.month, d.day, 0, 0, 0).timestamp() for d in v]
    d = _numeric_statistics_summary(v, c)
    for k in d.keys():
        if k in {"min", "max", "mean", "median"}:
            d[k] = date(*datetime.fromtimestamp(d[k]).timetuple()[:3])
        elif k in {"stdev", "iqr", "sum"}:
            d[k] = f"{d[k]/(24*60*60)} days"
        else:
            pass
    return d


def _string_statistics_summary(v, c):
    vx = [len(x) for x in v]
    d = _numeric_statistics_summary(vx, c)

    vc_sorted = sorted(zip(v, c), key=lambda t: t[1], reverse=True)
    mode, _ = vc_sorted[0]

    for k in d.keys():
        d[k] = f"{d[k]} characters"

    d["mode"] = mode

    return d


summary_methods = {
    bool: _boolean_statistics_summary,
    int: _numeric_statistics_summary,
    float: _numeric_statistics_summary,
    str: _string_statistics_summary,
    date: _date_statistics_summary,
    datetime: _datetime_statistics_summary,
    time: _time_statistics_summary,
    timedelta: _timedelta_statistics_summary,
    type(None): _none_type_summary,
}


def date_range(start, stop, step):
    if not isinstance(start, datetime):
        raise TypeError("start is not datetime")
    if not isinstance(stop, datetime):
        raise TypeError("stop is not datetime")
    if not isinstance(step, timedelta):
        raise TypeError("step is not timedelta")
    n = (stop - start) // step
    return [start + step * i for i in range(n)]


def dict_to_rows(d):
    type_check(d, dict)
    rows = []
    max_length = max(len(i) for i in d.values())
    order = list(d.keys())
    rows.append(order)
    for i in range(max_length):
        row = [d[k][i] for k in order]
        rows.append(row)
    return rows
