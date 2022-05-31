from tablite.utils import intercept, summary_statistics
import statistics
from itertools import chain

def test_range_intercept():
    A = range(500,700,3)
    B = range(520,700,3)
    C = range(10,1000,30)

    assert intercept(A,C) == range(0)
    assert set(intercept(B,C)) == set(B).intersection(set(C))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1000)

    assert set(intercept(A,B)) == set(A).intersection(set(B))

    A = range(500_000, 700_000, 1)
    B = range(10, 10_000_000, 1)

    assert set(intercept(A,B)) == set(A).intersection(set(B))

    A = range(0,2,1)
    B = range(0,2,1)
    assert set(intercept(A,B)) == set(A).intersection(set(B))

def test_normalize_slice():
    assert (0,10,1) == slice(0,None,1).indices(10)


def test_summary_statistics_even_ints():
    V,C  = [1,2,3,4], [2,3,4,5]  # Value, Count
    L = list(chain(*([v]*c for v,c in zip(V,C))))
    
    d = summary_statistics(V,C)
    assert d['min'] == min(L)
    assert d['max'] == max(L)
    assert d['mean'] == statistics.mean(L)
    assert d['median'] == statistics.median(L)
    assert d['stdev'] == statistics.stdev(L)
    assert d['mode'] == statistics.mode(L)
    assert d['distinct'] == len(V)
    low,mid,high = statistics.quantiles(L)
    assert d['iqr'] == high-low
    assert d['sum'] == sum(L)


def test_summary_statistics_even_ints_equal():
    V,C  = [1,2,3,4], [2,2,2,2]  # Value, Count
    L = list(chain(*([v]*c for v,c in zip(V,C))))
    
    d = summary_statistics(V,C)
    assert d['min'] == min(L)
    assert d['max'] == max(L)
    assert d['mean'] == statistics.mean(L)
    assert d['median'] == statistics.median(L)
    assert d['stdev'] == statistics.stdev(L)
    assert d['mode'] == statistics.mode(L)
    assert d['distinct'] == len(V)
    low,mid,high = statistics.quantiles(L,method='inclusive')
    assert d['iqr'] == high-low
    assert d['sum'] == sum(L)


def test_summary_statistics_odd_ints():
    V,C  = [1,2,3,4,5], [2,3,4,5,6]  # Value, Count
    L = list(chain(*([v]*c for v,c in zip(V,C))))
    
    d = summary_statistics(V,C)
    assert d['min'] == min(L)
    assert d['max'] == max(L)
    assert d['mean'] == statistics.mean(L)
    assert d['median'] == statistics.median(L)
    assert d['stdev'] == statistics.stdev(L)
    assert d['mode'] == statistics.mode(L)
    assert d['distinct'] == len(V)
    low,mid,high = statistics.quantiles(L,method='inclusive')
    assert d['iqr'] == high-low
    assert d['sum'] == sum(L)


def test_summary_statistics_odd_ints_equal():
    V,C  = [1,2,3,4,5], [2,2,2,2,2]  # Value, Count
    L = list(chain(*([v]*c for v,c in zip(V,C))))
    
    d = summary_statistics(V,C)
    assert d['min'] == min(L)
    assert d['max'] == max(L)
    assert d['mean'] == statistics.mean(L)
    assert d['median'] == statistics.median(L)
    assert d['stdev'] == statistics.stdev(L)
    assert d['mode'] == statistics.mode(L)
    assert d['distinct'] == len(V)
    low,mid,high = statistics.quantiles(L,method='inclusive')
    assert d['iqr'] == high-low
    assert d['sum'] == sum(L)

