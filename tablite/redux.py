from base import Table
from utils import sub_cls_check, type_check, expression_interpreter


def _filter(T, expression):
    """
    filters based on an expression, such as:

        "all((A==B, C!=4, 200<D))"

    which is interpreted using python's compiler to:

        def _f(A,B,C,D):
            return all((A==B, C!=4, 200<D))
    """
    sub_cls_check(T, Table)
    type_check(expression, str)

    try:
        _f = expression_interpreter(expression, T.columns)
    except Exception as e:
        raise ValueError(f"Expression could not be compiled: {expression}:\n{e}")

    req_columns = [i for i in self.columns if i in expression]
    bitmap = [bool(_f(*r)) for r in self.__getitem__(*req_columns).rows]
    inverse_bitmap = [not i for i in bitmap]

    if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
        true, false = Table(), Table()
        for col_name in self.columns:
            data = self[col_name][:]
            true[col_name] = list(itertools.compress(data, bitmap))
            false[col_name] = list(itertools.compress(data, inverse_bitmap))
        return true, false
    else:
        mask = np.array(bitmap, dtype=bool)
        return self._mp_compress(mask), self._mp_compress(np.invert(mask))  # true, false


def filter(self, expressions, filter_type="all", tqdm=_tqdm):
    """
    enables filtering across columns for multiple criteria.

    expressions:

        str: Expression that can be compiled and executed row by row.
            exampLe: "all((A==B and C!=4 and 200<D))"

        list of dicts: (example):

            L = [
                {'column1':'A', 'criteria': "==", 'column2': 'B'},
                {'column1':'C', 'criteria': "!=", "value2": '4'},
                {'value1': 200, 'criteria': "<", column2: 'D' }
            ]

        accepted dictionary keys: 'column1', 'column2', 'criteria', 'value1', 'value2'

    filter_type: 'all' or 'any'
    """
    if isinstance(expressions, str):
        return self._filter(expressions)

    if not isinstance(expressions, list) and not isinstance(expressions, tuple):
        raise TypeError

    if len(self) == 0:
        return self.copy(), self.copy()

    for expression in expressions:
        if not isinstance(expression, dict):
            raise TypeError(f"invalid expression: {expression}")
        if not len(expression) == 3:
            raise ValueError(f"expected 3 items, got {expression}")
        x = {"column1", "column2", "criteria", "value1", "value2"}
        if not set(expression.keys()).issubset(x):
            raise ValueError(f"got unknown key: {set(expression.keys()).difference(x)}")
        if expression["criteria"] not in filter_ops:
            raise ValueError(f"criteria missing from {expression}")

        c1 = expression.get("column1", None)
        if c1 is not None and c1 not in self.columns:
            raise ValueError(f"no such column: {c1}")
        v1 = expression.get("value1", None)
        if v1 is not None and c1 is not None:
            raise ValueError("filter can only take 1 left expr element. Got 2.")

        c2 = expression.get("column2", None)
        if c2 is not None and c2 not in self.columns:
            raise ValueError(f"no such column: {c2}")
        v2 = expression.get("value2", None)
        if v2 is not None and c2 is not None:
            raise ValueError("filter can only take 1 right expression element. Got 2.")

    if not isinstance(filter_type, str):
        raise TypeError()
    if filter_type not in {"all", "any"}:
        raise ValueError(f"filter_type: {filter_type} not in ['all', 'any']")

    # the results are to be gathered here:
    arr = np.zeros(shape=(len(expressions), len(self)), dtype=bool)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)
    _ = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)

    # the task manager enables evaluation of a column per core,
    # which is assembled in the shared array.
    max_task_size = math.floor(Config.SINGLE_PROCESSING_LIMIT / len(self.columns))
    # 1 million fields per core (best guess!)

    filter_tasks = []
    for ix, expression in enumerate(expressions):
        for step in range(0, len(self), max_task_size):
            config = {
                "table_key": self.key,
                "expression": expression,
                "shm_name": shm.name,
                "shm_index": ix,
                "shm_shape": arr.shape,
                "slice_": slice(step, min(step + max_task_size, len(self))),
            }
            task = Task(f=filter_evaluation_task, **config)
            filter_tasks.append(task)

    merge_tasks = []
    for step in range(0, len(self), max_task_size):
        config = {
            "table_key": self.key,
            "true_key": mem.new_id("/table"),
            "false_key": mem.new_id("/table"),
            "shm_name": shm.name,
            "shm_shape": arr.shape,
            "slice_": slice(step, min(step + max_task_size, len(self)), 1),
            "filter_type": filter_type,
        }
        task = Task(f=filter_merge_task, **config)
        merge_tasks.append(task)

    n_cpus = min(
        max(len(filter_tasks), len(merge_tasks)), psutil.cpu_count()
    )  # revise for case where memory footprint is limited to include zero subprocesses.

    with tqdm(total=len(filter_tasks) + len(merge_tasks), desc="filter") as pbar:
        if len(self) * (len(filter_tasks) + len(merge_tasks)) >= config.SINGLE_PROCESSING_LIMIT:
            with TaskManager(n_cpus) as tm:
                # EVALUATE
                errs = tm.execute(filter_tasks, pbar=pbar)
                # tm.execute returns the tasks with results, but we don't
                # really care as the result is in the result array.
                if any(errs):
                    raise Exception(errs)
                # MERGE RESULTS
                errs = tm.execute(merge_tasks, pbar=pbar)
                # tm.execute returns the tasks with results, but we don't
                # really care as the result is in the result array.
                if any(errs):
                    raise Exception(errs)
        else:
            for t in filter_tasks:
                r = t.f(*t.args, **t.kwargs)
                if r is not None:
                    raise r
                pbar.update(1)

            for t in merge_tasks:
                r = t.f(*t.args, **t.kwargs)
                if r is not None:
                    raise r
                pbar.update(1)

    cls = type(self)

    true = cls()
    true.add_columns(*self.columns)
    false = true.copy()

    for task in merge_tasks:
        tmp_true = cls.load(mem.path, key=task.kwargs["true_key"])
        if len(tmp_true):
            true += tmp_true
        else:
            pass

        tmp_false = cls.load(mem.path, key=task.kwargs["false_key"])
        if len(tmp_false):
            false += tmp_false
        else:
            pass
    return true, false


def all(self, **kwargs):
    """
    returns Table for rows where ALL kwargs match
    :param kwargs: dictionary with headers and values / boolean callable

    Examples:

        t = Table()
        t['a'] = [1,2,3,4]
        t['b'] = [10,20,30,40]

        def f(x):
            return x == 4
        def g(x):
            return x < 20

        t2 = t.any( **{"a":f, "b":g})
        assert [r for r in t2.rows] == [[1, 10], [4, 40]]

        t2 = t.any(a=f,b=g)
        assert [r for r in t2.rows] == [[1, 10], [4, 40]]

        def h(x):
            return x>=2

        def i(x):
            return x<=30

        t2 = t.all(a=h,b=i)
        assert [r for r in t2.rows] == [[2,20], [3, 30]]


    """
    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")
    if not all(k in self.columns for k in kwargs):
        raise ValueError(f"Unknown column(s): {[k for k in kwargs if k not in self.columns]}")

    ixs = None
    for k, v in kwargs.items():
        col = self._columns[k][:]
        if ixs is None:  # first header generates base set.
            if callable(v):
                ix2 = {ix for ix, i in enumerate(col) if v(i)}
            else:
                ix2 = {ix for ix, i in enumerate(col) if v == i}

        else:  # remaining headers reduce the base set.
            if callable(v):
                ix2 = {ix for ix in ixs if v(col[ix])}
            else:
                ix2 = {ix for ix in ixs if v == col[ix]}

        if not isinstance(ixs, set):
            ixs = ix2
        else:
            ixs = ixs.intersection(ix2)

        if not ixs:  # There are no matches.
            break

    if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
        cls = type(self)
        t = cls()
        for col_name in self.columns:
            data = self[col_name][:]
            t[col_name] = [data[i] for i in ixs]
        return t
    else:
        mask = np.array([True if i in ixs else False for i in range(len(self))], dtype=bool)
        return self._mp_compress(mask)


def any(self, **kwargs):
    """
    returns Table for rows where ANY kwargs match
    :param kwargs: dictionary with headers and values / boolean callable
    """
    redux.any(self, **kwargs)
    if not isinstance(kwargs, dict):
        raise TypeError("did you forget to add the ** in front of your dict?")

    ixs = set()
    for k, v in kwargs.items():
        col = self._columns[k][:]
        if callable(v):
            ix2 = {ix for ix, r in enumerate(col) if v(r)}
        else:
            ix2 = {ix for ix, r in enumerate(col) if v == r}
        ixs.update(ix2)

    if len(self) * len(self.columns) < config.SINGLE_PROCESSING_LIMIT:
        cls = type(self)

        t = cls()
        for col_name in self.columns:
            data = self[col_name][:]
            t[col_name] = [data[i] for i in ixs]
        return t
    else:
        mask = np.array([i in ixs for i in range(len(self))], dtype=bool)
        return self._mp_compress(mask)


def _mp_compress(self, mask):
    """
    helper for `any` and `all` that performs compression of the table self according to mask
    using multiprocessing.
    """
    arr = np.zeros(shape=(len(self),), dtype=bool)
    shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
    compresssion_mask = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
    compresssion_mask[:] = mask

    t = Table()
    tasks = []
    columns_refs = {}
    for name in self.columns:
        col = self[name]
        d_key = mem.new_id("/column")
        columns_refs[name] = d_key
        t = Task(compress_task, source_key=col.key, destination_key=d_key, shm_index_name=shm.name, shape=arr.shape)
        tasks.append(t)

    with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
        results = tm.execute(tasks)
        if any(r is not None for r in results):
            for r in results:
                print(r)
            raise Exception("!")

    with h5py.File(mem.path, "r+") as h5:
        table_key = mem.new_id("/table")
        dset = h5.create_dataset(name=f"/table/{table_key}", dtype=h5py.Empty("f"))
        dset.attrs["columns"] = json.dumps(columns_refs)
        dset.attrs["saved"] = False

    shm.close()
    shm.unlink()

    cls = type(self)

    t = cls.load(path=mem.path, key=table_key)
    return t
