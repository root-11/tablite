import os
import pathlib
import tempfile


class Config(object):
    # The default location for the storage
    workdir = pathlib.Path(os.environ.get("TABLITE_TMPDIR", f"{tempfile.gettempdir()}/tablite-tmp"))
    workdir.mkdir(parents=True, exist_ok=True)
    # to overwrite first import the config class:
    # >>> from tablite import config
    # >>> from pathlib import Path
    # >>> config.workdir = Path("/this/new/location")
    # for every new table or record this path will be used.

    PAGE_SIZE = 1_000_000  # sets the page size limit.
    ENCODING = "UTF-8"  # sets the page encoding when using bytes
    DISK_LIMIT = 10e9  # 10e9 (10Gb) on 100 Gb disk means raise at 90 Gb disk usage.

    SINGLE_PROCESSING_LIMIT = 1_000_000
    # when the number of fields (rows x columns)
    # exceed this value, multiprocessing is used.

    PROCESSING_PRIORITY = "auto"

    MULTIPROCESSING_ENABLED = True

    @classmethod
    def reset(cls):
        for k, v in _default_values.items():
            setattr(Config, k, v)


_default_values = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") or callable(v)}
