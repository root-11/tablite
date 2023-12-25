import os
import pathlib
import tempfile
import platform
from mplite import TaskManager as _TaskManager
from mplite import Task as _Task

from dotenv import load_dotenv

load_dotenv()


class Config(object):
    """Config class for Tablite Tables.

    The default location for the storage is loaded as
    ```
    Config.workdir = pathlib.Path(os.environ.get("TABLITE_TMPDIR", f"{tempfile.gettempdir()}/tablite-tmp"))
    ```
    to overwrite, first import the config class, then set the new workdir.
    ```
    >>> from tablite import config
    >>> from pathlib import Path
    >>> config.workdir = Path("/this/new/location")
    ```
    the new path will now be used for every new table.

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

    USE_NIMPORTER = os.environ.get("USE_NIMPORTER", "true").lower() in ["1","t","true","y","yes"]
    ALLOW_CSV_READER_FALLTHROUGH = os.environ.get("ALLOW_CSV_READER_FALLTHROUGH", "true").lower() in ["1", "t", "true", "y", "yes"]

    NIM_SUPPORTED_CONV_TYPES = ["Windows-1252", "ISO-8859-1"]

    workdir = pathlib.Path(os.environ.get("TABLITE_TMPDIR", f"{tempfile.gettempdir()}/tablite-tmp"))
    workdir.mkdir(parents=True, exist_ok=True)

    pid = f"pid-{os.getpid()}"

    PAGE_SIZE = 1_000_000  # sets the page size limit.
    ENCODING = "UTF-8"  # sets the page encoding when using bytes

    DISK_LIMIT = int(10e9)
    """ 
    10e9 (10Gb) on 100 Gb disk means raise at 90 Gb disk usage.
    if DISK_LIMIT <= 0, the check is turned off.
    """

    SINGLE_PROCESSING_LIMIT = 1_000_000
    """
    when the number of fields (rows x columns)
    exceed this value, multiprocessing is used.
    """
    vpus = max(os.cpu_count() - 1, 1)
    AUTO = "auto"
    FALSE = "sp"
    FORCE = "mp"
    MULTIPROCESSING_MODE = AUTO
    
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


class Task(_Task):
    def gets(self, *args):
        """helper to get kwargs of a task

        *Args:
            names from kwargs to retrieve.

        Returns:
            tuple: tuple with kw-values in same order as args

        Examples:

        Verbose way:
        ```
        >>> col = task.kwarg.get("left")
        >>> right = task.kwarg.get("right")
        >>> end = task.kwarg.get("end")
        ```
        Compact way:
        ```
        >>> col, start, end = task.gets("left", "start", "end")
        ```
        """
        result = tuple()
        for arg in args:
            result += self.kwargs.get(arg)
        return result


global_task_manager = None


class TaskManager(object):
    """
    A long lived task manager so that subprocesses aren't killed.
    """

    def __enter__(cls):
        if Config.MULTIPROCESSING_MODE == Config.FALSE:
            return cls

        global global_task_manager
        if global_task_manager is None:
            global_task_manager = _TaskManager(cpu_count=Config.vpus)
            global_task_manager.start()
        return cls

    def __exit__(cls, exc_type, exc_val, exc_tb):
        if exc_tb is not None:
            global global_task_manager
            if global_task_manager is not None:
                global_task_manager.stop()
                global_task_manager = None
        else:
            pass  # keep the task manager running.

    def execute(cls, tasks, pbar):
        if Config.MULTIPROCESSING_MODE == Config.FORCE:
            _mp = True
        elif Config.MULTIPROCESSING_MODE == Config.FALSE:
            _mp = False
        elif Config.vpus == 1:
            _mp = False
        elif Config.MULTIPROCESSING_MODE == Config.AUTO:
            if len(tasks) <= 1:
                _mp = False
            else:
                _mp = True
        else:
            _mp = True

        if _mp:
            results = global_task_manager.execute(tasks, pbar=pbar)
        else:
            pbar.update(0)
            results = []
            for task in tasks:
                result = task.execute()
                results.append(result)
                pbar.update(1 / len(tasks))
        return results


import atexit


def stop():
    global global_task_manager
    if global_task_manager is not None:
        global_task_manager.stop()
        global_task_manager = None


atexit.register(stop)
