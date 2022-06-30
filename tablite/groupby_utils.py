from collections import defaultdict
from datetime import date,time,datetime,timedelta
from tablite.datatypes import DataTypes


class GroupbyFunction(object):
    pass


class Limit(GroupbyFunction):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
        self.f = max


class Min(Limit):
    def __init__(self):
        super().__init__()
        self.f = min


class Sum(GroupbyFunction):
    def __init__(self):
        self.value = 0
    def update(self,value):
        if isinstance(value, (type(None), date, time, datetime, str)):
            raise ValueError(f"Sum of {type(value)} doesn't make sense.")
        self.value += value


class Product(GroupbyFunction):
    def __init__(self) -> None:
        self.value = 1
    def update(self,value):
        self.value *= value


class First(GroupbyFunction):
    empty = (None, )
    # we will never receive a tuple, so using (None,) as the initial
    # value will assure that IF None is the first value, then it can
    # be captured correctly.

    def __init__(self):
        self.value = self.empty

    def update(self, value):
        if self.value is First.empty:
            self.value = value


class Last(GroupbyFunction):
    def __init__(self):
        self.value = None

    def update(self, value):
        self.value = value


class Count(GroupbyFunction):
    def __init__(self):
        self.value = 0

    def update(self, value):
        self.value += 1


class CountUnique(GroupbyFunction):
    def __init__(self):
        self.items = set()
        self.value = None

    def update(self, value):
        self.items.add(value)
        self.value = len(self.items)


class Average(GroupbyFunction):
    def __init__(self):
        self.sum = 0
        self.count = 0
        self.value = 0

    def update(self, value):
        if isinstance(value, (date,time,datetime,str)):
            raise ValueError(f"Sum of {type(value)} doesn't make sense.")
        if value is not None:
            self.sum += value
            self.count += 1
            self.value = self.sum / self.count


class StandardDeviation(GroupbyFunction):
    """
    Uses J.P. Welfords (1962) algorithm.
    For details see https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm
    """
    def __init__(self):
        self.count = 0
        self.mean = 0
        self.c = 0.0

    def update(self, value):
        if isinstance(value, (date,time,datetime,str)):
            raise ValueError(f"Std.dev. of {type(value)} doesn't make sense.")
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
    def __init__(self):
        self.hist = defaultdict(int)

    def update(self, value):
        self.hist[value] += 1


class Median(Histogram):
    def __init__(self):
        super().__init__()

    @property
    def value(self):
        if not self.hist:
            raise ValueError("No data.")

        keys = len(self.hist.keys())
        if keys == 1:
            for k in self.hist:
                return k
        elif keys % 2 == 0:
            A, B, total, midpoint = None, None, 0, sum(self.hist.values()) / 2
            for k, v in sorted(self.hist.items()):
                total += v
                A, B = B, k
                if total > midpoint:
                    return (A + B) / 2
        else:
            midpoint = sum(self.hist.values()) / 2
            total = 0
            for k, v in sorted(self.hist.items()):
                total += v
                if total > midpoint:
                    return k


class Mode(Histogram):
    def __init__(self):
        super().__init__()

    @property
    def value(self):
        L = [(v, k) for k, v in self.hist.items()]
        L.sort(reverse=True)
        _, most_frequent = L[0]  # top of the list.
        return most_frequent



class GroupBy(object):    
    max = Max  # shortcuts to avoid having to type a long list of imports.
    min = Min
    sum = Sum
    product = Product
    first = First
    last = Last
    count = Count
    count_unique = CountUnique
    avg = Average
    stdev = StandardDeviation
    median = Median
    mode = Mode

    functions = [
        Max, Min, Sum, First, Last, Product,
        Count, CountUnique,
        Average, StandardDeviation, Median, Mode
    ]

    function_names = {f.__name__: f for f in functions}

