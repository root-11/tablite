import statistics
from tablite.utils import intercept, summary_statistics, date_range
from tablite.utils import expression_interpreter, dict_to_rows, sub_cls_check
from datetime import date, time, datetime, timedelta
from itertools import chain


def test_range_intercept():
    A = range(500, 700, 3)
    B = range(520, 700, 3)
    C = range(10, 1000, 30)

    assert intercept(A, C) == range(0)
    assert set(intercept(B, C)) == set(B).intersection(set(C))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1000)

    assert set(intercept(A, B)) == set(A).intersection(set(B))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1)

    assert set(intercept(A, B)) == set(A).intersection(set(B))

    A = range(0, 2, 1)
    B = range(0, 2, 1)
    assert set(intercept(A, B)) == set(A).intersection(set(B))


def test_normalize_slice():
    assert (0, 10, 1) == slice(0, None, 1).indices(10)


def test_intercept():
    r = range

    assert intercept(r(1, 3, 1), r(0, 4, 1)) == r(1, 3, 1)
    assert intercept(r(0, 5, 1), r(3, 7, 1)) == r(3, 5, 1)
    assert intercept(r(3, 7, 1), r(0, 5, 1)) == r(3, 5, 1)
    assert intercept(r(0, 1, 1), r(0, 1, 1)) == r(0, 1, 1)

    assert intercept(r(4, 20, 1), r(0, 16, 2)) == r(4, 16, 2)
    assert intercept(r(4, 20, 1), r(0, 16, 1)) == r(4, 16, 1)
    assert intercept(r(4, 20, 3), r(0, 16, 2)) == r(4, 16, 6)

    assert intercept(r(9, 0, -1), r(7, 10, 1)) == r(7, 10, 1)

    slc = slice(-1, 0, -1)
    A = r(*slc.indices(10))
    B = r(1, 11, 1)
    assert intercept(A, B) == r(1, 10, 1)

    A = range(-5, 5, 1)
    B = range(6, -6, -1)
    assert intercept(A, B) == range(-5, 5, 1)


def test_dict_to_rows():
    d = {"A": [1, 2, 3], "B": [10, 20, 30], "C": [100, 200, 300]}
    rows = dict_to_rows(d)
    assert rows == [["A", "B", "C"], [1, 10, 100], [2, 20, 200], [3, 30, 300]]


def test_subcls_check():
    class MyList(list):
        pass

    instance = MyList()

    sub_cls_check(instance, list)
    try:
        sub_cls_check(instance, int)
    except TypeError:
        pass


def test_interpreter():
    s = "all((A==B, C!=4, 200<D))"

    f = expression_interpreter(s, list("ABCDEF"))
    assert f(1, 2, 3, 4) is False
    assert f(10, 10, 0, 201) is True

    s2 = "any((A==B, C!=4, 200<D))"

    f = expression_interpreter(s2, list("ABCDEF"))
    assert f(1, 2, 4, 4) is False
    assert f(10, 10, 0, 201) is True


def test_summary_statistics_even_ints():
    V, C = [1, 2, 3, 4], [2, 3, 4, 5]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L)
    assert low < mid < high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_even_ints_equal():
    V, C = [1, 2, 3, 4], [2, 2, 2, 2]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L, method="inclusive")
    assert low < mid < high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_odd_ints():
    V, C = [1, 2, 3, 4, 5], [2, 3, 4, 5, 6]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L, method="inclusive")
    assert low < mid < high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_odd_ints_equal():
    V, C = [1, 2, 3, 4, 5], [2, 2, 2, 2, 2]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L, method="inclusive")
    assert low < mid < high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_min_data():
    V, C = [1], [2]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L, method="inclusive")
    assert low <= mid <= high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_min_data2():
    V, C = [1], [1]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    assert d["sum"] == sum(L)


def test_summary_statistics_even_floats():
    V, C = [1.1, 2.2, 3.3, 4.4], [2, 3, 4, 5]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == statistics.mean(L)
    assert d["median"] == statistics.median(L)
    assert d["stdev"] == statistics.stdev(L)
    assert d["mode"] == statistics.mode(L)
    assert d["distinct"] == len(V)
    low, mid, high = statistics.quantiles(L)
    assert low < mid < high
    assert d["iqr"] == high - low
    assert d["sum"] == sum(L)


def test_summary_statistics_even_strings():
    V, C = ["a", "bb", "ccc", "dddd"], [2, 3, 4, 5]  # Value, Count

    d = summary_statistics(V, C)
    assert d["min"] == "1 characters"
    assert d["max"] == "4 characters"
    assert d["mean"] == "2.857142857142857 characters"
    assert d["median"] == "3 characters"
    assert d["stdev"] == "1.0994504121565505 characters"
    assert d["mode"] == "dddd"
    assert d["distinct"] == len(V)
    assert d["iqr"] == "2 characters"
    assert d["sum"] == "40 characters"


def test_summary_statistics_mixed_most_floats():
    V, C = ["a", None, 1, 1.1], [2, 3, 4, 5]  # Value, Count
    d = summary_statistics(V, C)
    assert d["min"] == 1.1
    assert d["max"] == 1.1
    assert d["mean"] == 1.1
    assert d["median"] == 1.1
    assert d["stdev"] == 0.0
    assert d["mode"] == 1.1
    assert d["distinct"] == len(V)
    # low,mid,high = statistics.quantiles(L)
    assert d["iqr"] == 0.0
    assert d["sum"] == 5.5


def test_summary_statistics_datetimes():
    V = [datetime(1999, 12, i, 23, 59, 59, 999999) for i in range(1, 5)]
    C = [2, 3, 4, 5]  # Value, Count
    L = list(chain(*([v] * c for v, c in zip(V, C))))

    d = summary_statistics(V, C)
    assert d["min"] == min(L)
    assert d["max"] == max(L)
    assert d["mean"] == datetime(1999, 12, 3, 20, 34, 17, 142856)
    assert d["median"] == datetime(1999, 12, 3, 23, 59, 59, 999999)
    assert d["stdev"] == "1.0994504121563666 days"
    assert d["mode"] == datetime(1999, 12, 4, 23, 59, 59, 999999)
    assert d["distinct"] == len(V)
    assert d["iqr"] == "2.0 days"
    assert d["sum"] == "153003.99999999985 days"


def test_summary_statistics_mixed_types():
    V = [
        "a",
        None,
        datetime(1999, 12, 12, 23, 59, 59, 999999),
        date(2045, 12, 12),
        time(12, 0, 0, 0),
        3.1459,
        True,
        False,
    ]
    C = [5, 5, 4, 3, 1, 3, 5, 1]  # Value, Count

    d = summary_statistics(V, C)
    assert d["min"] is False
    assert d["max"] is True
    assert d["mean"] == 0.8333333333333334  # 5/6 True
    assert d["median"] is True
    assert d["stdev"] == 0.40824829046386296
    assert d["mode"] is True
    assert d["distinct"] == len(V)
    assert d["iqr"] == 0
    # iqr high = True , iqr low = True, so True-True becomes 1-1 = 0.
    assert d["sum"] == 5


def test_date_range():
    start, stop = datetime(2022, 1, 1), datetime(2023, 1, 1)
    step = timedelta(days=1)
    dr = date_range(start, stop, step)
    assert min(dr) == start
    assert max(dr) == stop - step
    assert dr == [start + step * i for i in range(365)]

    start, stop = datetime(2022, 12, 31), datetime(2021, 12, 31)
    step = timedelta(days=-1)
    dr = date_range(start, stop, step)
    assert min(dr) == datetime(2022, 1, 1)
    assert max(dr) == start
    assert dr == [start + step * i for i in range(365)]
