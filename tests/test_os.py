import time
import h5py
import pathlib
import tempfile


from tablite.config import TEMPDIR


def test01():
    assert TEMPDIR.exists()


def test2():
    t = tempfile.gettempdir()
    path = pathlib.Path(t) / "my.h5"
    if path.exists():
        path.unlink()

    key = "/table/1"
    with h5py.File(path, "a") as h5:
        dset = h5.create_dataset(name=key, dtype=h5py.Empty("f"))
        dset.attrs["now"] = time.time()

    with h5py.File(path, "w") as h5:
        assert list(h5.keys()) == []

    with h5py.File(path, "r+") as h5:
        assert key not in h5.keys()
        dset = h5.create_dataset(name=key, dtype=h5py.Empty("f"))
        dset.attrs["now"] = time.time()

    path.unlink()
