import copy
import io
import traceback
from itertools import count


POOL = None


class TaskManager(object):
    def __init__(self) -> None:
        pass
        if __name__ == "__main__" and POOL is None:
            global POOL
            POOL = [Worker(str(i)) for i in range(n_cpus)]

    def __enter__(self):
        pass
    def __exit__(self):
        pass


class Worker(object):
    pass


class Task(ABC):
    """
    Generic Task class for tasks.
    """
    ids = count(start=1)
    def __init__(self, f, *args, **kwargs) -> None:
        """
        f: callable 
        *args: arguments for f
        **kwargs: keyword arguments for f.
        """
        if not callable(f):
            raise TypeError
        self.task_id = next(self.ids)
        self.f = f
        self.args = copy.deepcopy(args)  # deep copy is slow unless the data is shallow.
        self.kwargs = copy.deepcopy(kwargs)
        self.result = None
        self.exception = None

    def __str__(self) -> str:
        if self.exception:
            return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Error: {self.exception}"
        else:
            return f"Call to {self.f.__name__}(*{self.args}, **{self.kwargs}) --> Result: {self.result}"

    def execute(self):
        """ The worker calls this function. """
        try:
            self.result = self.f(*self.args, **self.kwargs)
        except Exception as e:
            f = io.StringIO()
            traceback.print_exc(limit=3, file=f)
            f.seek(0)
            error = f.read()
            f.close()
            self.exception = error