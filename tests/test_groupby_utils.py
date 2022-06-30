from tablite.groupby_utils import GroupBy as gb
import statistics

def test_median():
    def median(values):
        m = gb.median()
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


def test_max():
    m = gb.max()
    for i in [-2,-1,0,1,2,3]:
        m.update(i)
    assert m.value == 3

def test_min():
    m = gb.min()
    for i in [-2,-1,0,1,2,3]:
        m.update(i)
    assert m.value == -2

def test_sum():
    m = gb.sum()
    L = [-2,-1,0,1,2,3]
    for i in L:
        m.update(i)
    assert sum(L) == m.value

def test_product():
    m = gb.product()
    L = [1,2,3,4,5]
    x = 1
    for i in L:
        m.update(i)
        x *= i
    assert x == m.value


def test_first_last():
    a = gb.first()
    b = gb.last()
    L = [-2,-1,0,1,2,3]
    for i in L:
        a.update(i)
        b.update(i)
    assert a.value == -2
    assert b.value == 3

def test_count():
    c = gb.count()
    cu = gb.count_unique()
    for i in [1,1,2,2]:
        c.update(i)
        cu.update(i)
    assert c.value == 4
    assert cu.value == 2

def test_average():
    avg = gb.avg()
    L = [-2,-1,0,1,2,3]
    for i in L:
        avg.update(i)
    assert avg.value == sum(L) / len(L)

def test_average2():
    avg = gb.avg()
    L = [0]
    for i in L:
        avg.update(i)
    assert avg.value == sum(L) / len(L)


def test_stdev():
    m = gb.stdev()
    L = [1,1]
    for i in L:
        m.update(i)
    assert m.value == 0

    m = gb.stdev()
    L = [1,1,2,2]
    for i in L:
        m.update(i)
    assert m.value == statistics.stdev(L)
    

def test_mode():

    def mode(values):
        m = gb.mode()
        for i in values:
            m.update(i)
        return m.value

    assert mode([1]) == 1
    assert mode([1,1,2]) == 1
    assert mode([1,1,2,3,3]) == 3
    assert mode([1,1,2,2,3,3]) == 3

    # raise NotImplementedError("the functions above need verification")
