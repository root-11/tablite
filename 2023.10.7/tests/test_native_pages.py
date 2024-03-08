import numpy as np
from pathlib import Path
from tablite.nimlite import read_page

dirpages = Path(__file__).parent / "data" / "pages"


def _test_cmp(path: Path):
    # validates that our nim loader has parity with np.load
    arr = read_page(path).tolist()
    res = np.load(path, allow_pickle=True).tolist()

    assert arr == res


def test_load_bool(): _test_cmp(dirpages / "boolean.npy")
def test_load_bool_nones(): _test_cmp(dirpages / "boolean_nones.npy")
def test_load_int(): _test_cmp(dirpages / "int.npy")
def test_load_int_nones(): _test_cmp(dirpages / "int_nones.npy")
def test_load_float(): _test_cmp(dirpages / "float.npy")
def test_load_float_nones(): _test_cmp(dirpages / "float_nones.npy")
def test_load_str(): _test_cmp(dirpages / "str.npy")
def test_load_str_nones(): _test_cmp(dirpages / "str_nones.npy")
def test_load_date(): _test_cmp(dirpages / "date.npy")
def test_load_date_nones(): _test_cmp(dirpages / "date_nones.npy")
def test_load_time(): _test_cmp(dirpages / "time.npy")
def test_load_time_nones(): _test_cmp(dirpages / "time_nones.npy")
def test_load_datetime(): _test_cmp(dirpages / "datetime.npy")
def test_load_datetime_nones(): _test_cmp(dirpages / "datetime_nones.npy")
def test_load_mixed(): _test_cmp(dirpages / "mixed.npy")
def test_load_scalar(): _test_cmp(dirpages / "scalar.npy")
