from tablite.groupby_utils import *


def test_median():
    def median(values):
        dtype = float if isinstance(values[0], float) else int
        m = Median(dtype)
        for v in values:
            m.update(v)
        return m.value

    assert median([1, 2, 3, 4, 5]) == 3
    assert median([1, 2, 3, 6, 7, 8]) == 4.5
    assert median([3]) == 3
    assert median([3, 3]) == 3
    assert median([3, 3, 3]) == 3
    assert median([3, 3, 6, 6, 9, 9]) == 6
    assert median([3, 3, 3, 9, 9, 9]) == 6
    assert median([-1, -1, 0, 1, 1]) == 0
    assert median([-1, -1, 0, 0, 1, 1]) == 0
    assert median([5, 4, 6, 3, 7, 2, 8, 1, 9]) == 5
    assert median([i/10 for i in range(10)]) == 0.45
    assert median([i/10 for i in range(1,10)]) == 0.5


def test_more():
    GroupbyFunction()
    Limit()
    Max()
    Min()
    Sum()
    First()
    Last()
    Count()
    CountUnique()
    Average()
    StandardDeviation()
    Histogram()
    Median()
    Mode()
    raise NotImplementedError("the functions above need verification")
