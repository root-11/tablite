from tablite.datasets import synthetic_order_data
from tablite.config import Config


def test01():
    old_page_size = Config.PAGE_SIZE
    Config.PAGE_SIZE = 100
    length = 250
    t = synthetic_order_data(length)
    assert len(t) == length
    assert t["#"] == list(range(1, length + 1))

    a = t["#"].types()
    t *= 3
    assert len(t) == length * 3
    b = t["#"].types()
    assert {k: v * 3 for k, v in a.items()} == b

    Config.PAGE_SIZE = old_page_size
