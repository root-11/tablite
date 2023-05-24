import time
import os
import gc
import psutil
from pathlib import Path
from tempfile import tempdir

from tablite.datasets import synthetic_order_data


def test01():
    process = psutil.Process(os.getpid())
    baseline_memory = process.memory_info().rss

    length = 2_000_000
    fn = Path(tempdir) / "synth data.tpz"
    start = time.process_time()
    t = synthetic_order_data(length)
    created = time.process_time()
    while gc.collect():
        pass

    memory_after_table_creation = process.memory_info().rss
    memory_for_t = memory_after_table_creation - baseline_memory

    print(f"creating {t} took {created-start} sec")
    print(f"current RAM for table: {memory_for_t//1_000_000}Mb")
    print(t.show())

    t.save(fn)
    saved = time.process_time()
    print(f"saving {t} took {saved-created} sec  {os.path.getsize(fn)//1000:,}kb")

    t2 = t.load(fn)
    loaded = time.process_time()
    print(f"loading {t} took {loaded-saved} sec")

    del t2
    while gc.collect():
        pass
    os.remove(fn)

    start = time.process_time()
    t3 = t.copy()
    end = time.process_time()
    print(f"t.copy took {end-start} sec")
    del t3

    start = time.process_time()
    assert 1 in t["#"]
    short = time.process_time()
    assert length in t["#"]
    long = time.process_time()
    print(f"t.__contains__ took {short-start} / {long-short} (best-/worst case) sec")

    start = time.process_time()
    d = t.types()
    end = time.process_time()
    print(f"t.types() took {end-start} sec")

    length = len(t)
    for name in t.columns:
        start = time.process_time()
        L = t[name].unique()
        end = time.process_time()
        print(f"t.unique({name}) took {end-start} sec.")
        assert 0 < len(L) <= length

    for name in t.columns:
        start = time.process_time()
        _ = t.index(name)
        end = time.process_time()
        print(f"t.index({name}) took {end-start} sec.")

    names = ["4", "7", "8", "9"]
    for ix in range(1, 5):
        start = time.process_time()
        t.index(*names[:ix])
        end = time.process_time()
        print(f"t.index({names[:ix]}) took {end-start}")
