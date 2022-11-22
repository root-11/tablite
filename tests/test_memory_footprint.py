import psutil
from tablite import Table
import gc
import time
from time import process_time
import os


def test_recreate_readme_comparison():
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss

    digits = 1_000_000

    records = Table()
    records.add_column("method")
    records.add_column("memory")
    records.add_column("time")

    # Let's now use the common and convenient "row" based format:

    start = process_time()
    L = []
    for _ in range(digits):
        L.append(tuple([11 for _ in range(10)]))
    end = process_time()

    # go and check taskmanagers memory usage.
    # At this point we're using ~154.2 Mb to store 1 million lists with 10 items.
    records.add_rows(("1e6 lists w. 10 integers", process.memory_info().rss - baseline_memory, round(end - start, 4)))

    L.clear()
    gc.collect()
    time.sleep(1)

    # Let's now use a columnar format instead:
    start = process_time()
    L = [[11 for i in range(digits)] for _ in range(10)]
    end = process_time()

    # go and check taskmanagers memory usage.
    # at this point we're using ~98.2 Mb to store 10 lists with 1 million items.
    records.add_rows(("10 lists with 1e6 integers", process.memory_info().rss - baseline_memory, round(end - start, 4)))
    L.clear()
    gc.collect()
    time.sleep(1)

    # We've thereby saved 50 Mb by avoiding the overhead from managing 1 million lists.

    # Q: But why didn't I just use an array? It would have even lower memory footprint.
    # A: First, array's don't handle None's and we get that frequently in dirty csv data.
    # Second, Table needs even less memory.

    # Let's start with an array:

    import array

    start = process_time()
    L = [array.array("i", [11 for _ in range(digits)]) for _ in range(10)]
    end = process_time()
    # go and check taskmanagers memory usage.
    # at this point we're using 60.0 Mb to store 10 lists with 1 million integers.

    records.add_rows(
        ("10 lists with 1e6 integers in arrays", process.memory_info().rss - baseline_memory, round(end - start, 4))
    )
    L.clear()
    gc.collect()
    time.sleep(1)

    # Now let's use Table:

    start = process_time()
    t = Table()
    for i in range(10):
        t.add_column(str(i), data=[11 for _ in range(digits)])
    end = process_time()

    records.add_rows(
        ("Table with 10 columns with 1e6 integers", process.memory_info().rss - baseline_memory, round(end - start, 4))
    )

    start = process_time()
    _ = t.copy()
    end = process_time()

    records.add_rows(
        (
            "2 Tables with 10 columns with 1e6 integers each",
            process.memory_info().rss - baseline_memory,
            round(end - start, 4),
        )
    )

    # go and check taskmanagers memory usage.
    # At this point we're using  24.5 Mb to store 10 columns with 1 million integers.
    # Only the metadata remains in pythons memory.

    records.show()
    # +===+===============================================+=======++==+======+
    # | # |                     method                    |   memory  | time |
    # |row|                      str                      |    int    |float |
    # +---+-----------------------------------------------+-----------+------+
    # |0  |1e6 lists w. 10 integers                       |141,307,904|0.6562|
    # |1  |10 lists with 1e6 integers                     | 84,103,168|0.5625|
    # |2  |10 lists with 1e6 integers in arrays           | 44,027,904|0.6719|
    # |3  |Table with 10 columns with 1e6 integers        |  3,203,072|1.6094|
    # |4  |2 Tables with 10 columns with 1e6 integers each|  3,846,144|0.0781|
    # +===+===============================================+====++=====+======+
