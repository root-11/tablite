from tablite.datasets import synthetic_order_data
from tablite.config import Config


def test01():
    old_page_size = Config.PAGE_SIZE
    Config.PAGE_SIZE = 100
    length = 250
    t = synthetic_order_data(length)
    assert len(t) == length
    assert t["#"] == list(range(1, length + 1))
    Config.PAGE_SIZE = old_page_size
