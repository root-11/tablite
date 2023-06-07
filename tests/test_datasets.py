from tablite.datasets import synthetic_order_data


def test01():
    length = 2_000
    t = synthetic_order_data(length)
    assert len(t) == length
