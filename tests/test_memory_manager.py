from tablite.memory_manager import timeout, MemoryManager
from tablite import Table
import time


@timeout
def sleepy(n):
    time.sleep(n / 10)
    return n


@timeout
def blowup():
    raise OSError()


def test_timeout_result():
    for i in [1, 3]:
        x = sleepy(i)  # tests that the timeout decorator works on sleepy
        assert x == i


def test_empty_table_creation():
    """
        Was failing when creating a table with some empty pages, however equivalent form presented below was passing.
        
        mem = MemoryManager()
        cols = { "A": [], "B": [], "C": [], } 
        new_table = Table.from_dict(cols) 
        t = Table.load(mem.path, new_table.key)
    """
    mem = MemoryManager()
    cols = { 
        "A": mem.mp_write_column(values=[]), 
        "B": mem.mp_write_column(values=[]), 
        "C": mem.mp_write_column(values=[]), 
    } 
    new_table_key = mem.new_id("/table") 
    mem.mp_write_table(new_table_key, columns=cols) 
    t = Table.load(mem.path, new_table_key) 
    assert len(t) == 0


def test_merge_columns_1():
    mem = MemoryManager()
    cols = [
        mem.mp_write_column(values=[1, 2, 3]),
    ]

    merged_column_key = mem.mp_merge_columns(cols)

    assert merged_column_key == cols[0]

    new_table_key = mem.new_id("/table")
    mem.mp_write_table(new_table_key, columns={"A": merged_column_key})
    t = Table.load(mem.path, new_table_key)
    assert len(t) == 3


def test_merge_columns_2():
    mem = MemoryManager()
    cols = [
        mem.mp_write_column(values=[1, 2, 3]),
        mem.mp_write_column(values=[4, 5, 6]),
    ]
    new_table_key = mem.new_id("/table")
    mem.mp_write_table(new_table_key, columns={"A": mem.mp_merge_columns(cols)})
    t = Table.load(mem.path, new_table_key)
    assert len(t) == 6


def test_merge_columns_3():
    mem = MemoryManager()
    cols = [
        mem.mp_write_column(values=[1, 2, 3]),
        mem.mp_write_column(values=[4, 5, 6]),
        mem.mp_write_column(values=[7, 8, 9]),
    ]
    new_table_key = mem.new_id("/table")
    mem.mp_write_table(new_table_key, columns={"A": mem.mp_merge_columns(cols)})
    t = Table.load(mem.path, new_table_key)
    assert len(t) == 9


def test_merge_columns_4():
    mem = MemoryManager()
    cols = [
        mem.mp_write_column(values=[1, 2, 3]),
    ]

    column_key = mem.new_id("/column")
    merged_column_key = mem.mp_merge_columns(cols, column_key)

    assert merged_column_key == column_key

    new_table_key = mem.new_id("/table")
    mem.mp_write_table(new_table_key, columns={"A": merged_column_key})
    t = Table.load(mem.path, new_table_key)
    assert len(t) == 3


def test_merge_columns_5():
    mem = MemoryManager()
    cols = [
        mem.mp_write_column(values=[1, 2, 3]),
        mem.mp_write_column(values=[4, 5, 6]),
    ]

    column_key = mem.new_id("/column")
    merged_column_key = mem.mp_merge_columns(cols, column_key)

    assert merged_column_key == column_key

    new_table_key = mem.new_id("/table")
    mem.mp_write_table(new_table_key, columns={"A": merged_column_key})
    t = Table.load(mem.path, new_table_key)
    assert len(t) == 6
