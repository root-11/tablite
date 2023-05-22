import pathlib
import numpy as np
from datetime import datetime, timedelta
import tempfile
import shutil

from tablite import Table
from tablite.export_utils import exporters


import pytest


@pytest.fixture(autouse=True)  # this resets the HDF5 file for every test.
def refresh():
    Table.reset_storage()
    yield


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

    for suffix in exporters:
        path = test_dir / f"myfile.{suffix}"
        t.export(path)
        assert path.exists()

    shutil.rmtree(test_dir)
