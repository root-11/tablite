import os
import pathlib
import tempfile


class Config(object):
    """Config class for Tablite Tables.

    The default location for the storage is loaded as

    Config.workdir = pathlib.Path(os.environ.get("TABLITE_TMPDIR", f"{tempfile.gettempdir()}/tablite-tmp"))

    to overwrite, first import the config class, then set the new workdir.
    >>> from tablite import config
    >>> from pathlib import Path
    >>> config.workdir = Path("/this/new/location")
    for every new table or record this path will be used.

    PAGE_SIZE = 1_000_000 sets the page size limit.

    Multiprocessing is enabled in one of three modes:
    AUTO = "auto"
    FALSE = "sp"
    FORCE = "mp"

    MULTIPROCESSING_MODE = AUTO  is default.

    SINGLE_PROCESSING_LIMIT = 1_000_000
    when the number of fields (rows x columns) exceed this value,
    multiprocessing is used.
    """

    workdir = pathlib.Path(os.environ.get("TABLITE_TMPDIR", f"{tempfile.gettempdir()}/tablite-tmp"))
    workdir.mkdir(parents=True, exist_ok=True)

    PAGE_SIZE = 1_000_000  # sets the page size limit.
    ENCODING = "UTF-8"  # sets the page encoding when using bytes

    DISK_LIMIT = int(10e9)  # 10e9 (10Gb) on 100 Gb disk means raise at
    # 90 Gb disk usage.
    # if DISK_LIMIT <= 0, the check is turned off.

    SINGLE_PROCESSING_LIMIT = 1_000_000
    # when the number of fields (rows x columns)
    # exceed this value, multiprocessing is used.

    AUTO = "auto"
    FALSE = "sp"
    FORCE = "mp"
    MULTIPROCESSING_MODE = AUTO
    # Usage example (from import_utils in text_reader)
    # if cpu_count < 2 or Config.MULTIPROCESSING_MODE == Config.FALSE:
    #         for task in tasks:
    #             task.execute()
    #             pbar.update(dump_size)
    #     else:
    #         with TaskManager(cpu_count - 1) as tm:
    #             errors = tm.execute(tasks, pbar=PatchTqdm())  # I expects a list of None's if everything is ok.
    #             if any(errors):
    #                 raise Exception("\n".join(e for e in errors if e))

    TQDM_DISABLE = False  # set to True to disable tqdm

    @classmethod
    def reset(cls):
        """Resets the config class to original values."""
        for k, v in _default_values.items():
            setattr(Config, k, v)

    @classmethod
    def page_steps(cls, length):
        """an iterator that yield start and end in page sizes

        Yields:
            tuple: start:int, end:int
        """
        start, end = 0, 0
        for _ in range(0, length + 1, cls.PAGE_SIZE):
            start, end = end, min(end + cls.PAGE_SIZE, length)
            yield start, end
            if end == length:
                return


_default_values = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") or callable(v)}
