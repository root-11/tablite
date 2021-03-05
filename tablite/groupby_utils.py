from collections import defaultdict

from tablite.datatypes import DataTypes


class GroupbyFunction(object):
    def __init__(self, datatype):
        hasattr(DataTypes, datatype.__name__)
        self.datatype = datatype


class Limit(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = None
        self.f = None

    def update(self, value):
        if value is None:
            pass
        elif self.value is None:
            self.value = value
        else:
            self.value = self.f((value, self.value))


class Max(Limit):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.f = max


class Min(Limit):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.f = min


class Sum(Limit):
    def __init__(self, datatype):
        if datatype in (DataTypes.date, DataTypes.date, DataTypes.time, DataTypes.str):
            raise ValueError(f"Sum of {datatype} doesn't make sense.")
        super().__init__(datatype)
        self.f = sum


class First(GroupbyFunction):
    empty = (None, )
    # we will never receive a tuple, so using (None,) as the initial
    # value will assure that IF None is the first value, then it can
    # be captured correctly.

    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = self.empty

    def update(self, value):
        if self.value is First.empty:
            self.value = value


class Last(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.value = None

    def update(self, value):
        self.value = value


class Count(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=int)  # datatype will be int no matter what type is given.
        self.value = 0

    def update(self, value):
        self.value += 1


class CountUnique(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype=int)  # datatype will be int no matter what type is given.
        self.items = set()
        self.value = None

    def update(self, value):
        self.items.add(value)
        self.value = len(self.items)


class Average(GroupbyFunction):
    def __init__(self, datatype):
        if datatype in (DataTypes.date, DataTypes.date, DataTypes.time, DataTypes.str):
            raise ValueError(f"Average of {datatype} doesn't make sense.")
        super().__init__(datatype=float)  # datatype will be float no matter what type is given.
        self.sum = 0
        self.count = 0
        self.value = 0

    def update(self, value):
        if value is not None:
            self.sum += value
            self.count += 1
            self.value = self.sum / self.count


class StandardDeviation(GroupbyFunction):
    """
    Uses J.P. Welfords (1962) algorithm.
    For details see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """
    def __init__(self, datatype):
        if datatype in (DataTypes.date, DataTypes.date, DataTypes.time, DataTypes.str):
            raise ValueError(f"Std.dev. of {datatype} doesn't make sense.")
        super().__init__(datatype=float)  # datatype will be float no matter what type is given.
        self.count = 0
        self.mean = 0
        self.c = 0.0

    def update(self, value):
        if value is not None:
            self.count += 1
            dt = value - self.mean
            self.mean += dt / self.count
            self.c += dt * (value - self.mean)

    @property
    def value(self):
        if self.count <= 1:
            return 0.0
        variance = self.c / (self.count - 1)
        return variance ** (1 / 2)


class Histogram(GroupbyFunction):
    def __init__(self, datatype):
        super().__init__(datatype)
        self.hist = defaultdict(int)

    def update(self, value):
        self.hist[value] += 1


class Median(Histogram):
    def __init__(self, datatype):
        super().__init__(datatype)

    @property
    def value(self):
        midpoint = sum(self.hist.values()) / 2
        total = 0
        for k, v in self.hist.items():
            total += v
            if total > midpoint:
                return k


class Mode(Histogram):
    def __init__(self, datatype):
        super().__init__(datatype)

    @property
    def value(self):
        L = [(v, k) for k, v in self.hist.items()]
        L.sort(reverse=True)
        frequency, most_frequent = L[0]  # top of the list.
        return most_frequent