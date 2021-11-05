from tablite.groupby_utils import Median


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
