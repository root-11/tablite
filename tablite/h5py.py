from h5py import Empty
from h5py import string_dtype
from h5py import File as _File
from time import sleep, time as now
from contextlib import contextmanager
from tablite.memory_manager import TIMEOUT


@contextmanager
def File(*args, **kwargs):
    f = None
    timout_time = now() + TIMEOUT / 1000

    while f is None:
        try:
            f = _File(*args, **kwargs)
        except BlockingIOError:
            if timout_time < now():
                raise OSError(f"couldn't write to disk (slept {TIMEOUT} msec")

            sleep(0.01)

    try:
        yield f
    finally:
        f.close()
