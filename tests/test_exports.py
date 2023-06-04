import pathlib
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from tablite import Table


def test_exporters():
    now = datetime.now().replace(microsecond=0)

    t = Table()
    t["A"] = [-1, 1]
    t["B"] = [None, 1]
    t["C"] = [-1.1, 1.1]
    t["D"] = ["", "1000"]
    t["E"] = [None, "1"]
    t["F"] = [False, True]
    t["G"] = [now, now]
    t["H"] = [now.date(), now.date()]
    t["I"] = [now.time(), now.time()]
    t["J"] = [timedelta(1), timedelta(2, 400)]
    t["K"] = ["b", "嗨"]  # utf-32
    t["L"] = [-(10**23), 10**23]  # int > int64.
    t["M"] = [float("inf"), float("-inf")]
    t["O"] = [np.int32(11), np.int64(-11)]

    test_dir = pathlib.Path(tempfile.gettempdir()) / "junk_test"
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(exist_ok=False)

    p = test_dir / "1.h5"
    t.to_hdf5(p)
    assert p.exists()

    p = test_dir / "2.txt"
    t.to_ascii(p)
    assert p.exists()

    p = test_dir / "3.csv"
    t.to_csv(p)
    assert p.exists()

    d = t.to_dict()
    assert isinstance(d, dict)

    p = test_dir / "4.html"
    t.to_html(p)
    assert p.exists()

    s = t.to_json()
    assert s is not None

    p = test_dir / "5.ods"
    t.to_ods(p)
    assert p.exists()
    p = test_dir / "6.txt"
    t.to_text(p)
    assert p.exists()
    p = test_dir / "7.tsv"
    t.to_tsv(p)
    assert p.exists()

    shutil.rmtree(test_dir)
