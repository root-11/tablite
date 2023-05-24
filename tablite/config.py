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

    PAGE_FRAGMENTATION_CHECK = False
    PAGE_FRAGMENTATION_TOLERANCE = 2
    # at 2 pages per 1M fields, pages will be consolidated.
    PAGE_FRAGMENTATION_INSPECTION_RATE = 50
    # After 50 inserts/updates in a column fragmentation
    # a check is performed. If the number of pages exceed
    # the fragmentation tolerance, the data will be are
    # repaginated to PAGE_SIZE.

    ENCODING = "UTF-8"  # sets the page encoding when using bytes
    DISK_LIMIT = 10e9  # 10e9 (10Gb) on 100 Gb disk means raise at
    # 90 Gb disk usage.

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

    @classmethod
    def reset(cls):
        """Resets the config class to original values."""
        for k, v in _default_values.items():
            setattr(Config, k, v)


_default_values = {k: v for k, v in Config.__dict__.items() if not k.startswith("__") or callable(v)}


class FragmentationMonitor(object):  # TODO: To be added to Column.__setitem__(...)
    """decorator to monitor for page fragmentation."""
    def __init__(self, arg) -> None:
        self._arg = arg
        self._change_counter = 0

    def __call__(self, *args: Any) -> Any:
        result = self._arg(*args)
        if Config.PAGE_FRAGMENTATION_CHECK:
            self._change_counter += 1
            if self._change_counter >= Config.PAGE_FRAGMENTATION_INSPECTION_RATE:
                self._change_counter = 0

                column = args[0]
                assert isinstance(column, Column)
                n_pages = math.ceil(len(column) / Config.PAGE_SIZE)
                if len(column.pages) / n_pages > Config.PAGE_FRAGMENTATION_TOLERANCE:
                    column.repaginate()

        return result