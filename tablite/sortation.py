def sort_index(self, sort_mode="excel", tqdm=_tqdm, pbar=None, **kwargs):
    """
    helper for methods `sort` and `is_sorted`

    param: sort_mode: str: "alphanumeric", "unix", or, "excel" (default)
    param: **kwargs: sort criteria. See Table.sort()
    """
    logging.info("Table.sort_index running 1 core")  # This is single core code.

    if not isinstance(kwargs, dict):
        raise ValueError("Expected keyword arguments, did you forget the ** in front of your dict?")
    if not kwargs:
        kwargs = {c: False for c in self.columns}

    for k, v in kwargs.items():
        if k not in self.columns:
            raise ValueError(f"no column {k}")
        if not isinstance(v, bool):
            raise ValueError(f"{k} was mapped to {v} - a non-boolean")

    if sort_mode not in sortation.modes:
        raise ValueError(f"{sort_mode} not in list of sort_modes: {list(sortation.Sortable.modes.modes)}")

    rank = {i: tuple() for i in range(len(self))}  # create index and empty tuple for sortation.

    _pbar = tqdm(total=len(kwargs.items()), desc="creating sort index") if pbar is None else pbar

    for key, reverse in kwargs.items():
        col = self._columns[key][:]
        col = col.tolist() if isinstance(col, np.ndarray) else col
        ranks = sortation.rank(values=set(col), reverse=reverse, mode=sort_mode)
        assert isinstance(ranks, dict)
        for ix, v in enumerate(col):
            rank[ix] += (ranks[v],)  # add tuple

        _pbar.update(1)

    new_order = [(r, i) for i, r in rank.items()]  # tuples are listed and sort...
    rank.clear()  # free memory.
    new_order.sort()
    sorted_index = [i for _, i in new_order]  # new index is extracted.
    new_order.clear()
    return sorted_index


def reindex(self, index):
    """
    index: list of integers that declare sort order.

    Examples:

        Table:  ['a','b','c','d','e','f','g','h']
        index:  [0,2,4,6]
        result: ['b','d','f','h']

        Table:  ['a','b','c','d','e','f','g','h']
        index:  [0,2,4,6,1,3,5,7]
        result: ['a','c','e','g','b','d','f','h']

    """
    if index is not None:
        if not isinstance(index, list):
            raise TypeError
        if max(index) >= len(self):
            raise IndexError("index out of range: max(index) > len(self)")
        if min(index) < -len(self):
            raise IndexError("index out of range: min(index) < -len(self)")
        if not all(isinstance(i, int) for i in index):
            raise TypeError

    if (
        len(self) * len(self.columns) < Config.SINGLE_PROCESSING_LIMIT
    ):  # the task is so small that multiprocessing doesn't make sense.
        t = Table()
        for col_name, col in self._columns.items():  # this LOOP can be done with TaskManager
            data = list(col[:])
            t.add_column(col_name, data=[data[ix] for ix in index])
        return t

    else:
        arr = np.zeros(shape=(len(index),), dtype=np.int64)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
        sort_index = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        sort_index[:] = index

        tasks = []
        columns_refs = {}
        for name in self.columns:
            col = self[name]
            columns_refs[name] = d_key = mem.new_id("/column")
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=shm.name,
                    shape=arr.shape,
                )
            )

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            errs = tm.execute(tasks)
            if any(errs):
                raise Exception("\n".join(filter(lambda x: x is not None, errs)))

        table_key = mem.new_id("/table")
        mem.create_table(key=table_key, columns=columns_refs)

        shm.close()
        shm.unlink()
        t = Table.load(path=mem.path, key=table_key)
        return t


def sort(self, sort_mode="excel", **kwargs):
    """Perform multi-pass sorting with precedence given order of column names.
    sort_mode: str: "alphanumeric", "unix", or, "excel"
    kwargs:
        keys: columns,
        values: 'reverse' as boolean.

    examples:
    Table.sort('A'=False) means sort by 'A' in ascending order.
    Table.sort('A'=True, 'B'=False) means sort 'A' in descending order, then (2nd priority)
    sort B in ascending order.
    """
    if (
        len(self) * len(self.columns) < Config.SINGLE_PROCESSING_LIMIT
    ):  # the task is so small that multiprocessing doesn't make sense.
        sorted_index = self.sort_index(sort_mode=sort_mode, **kwargs)

        t = Table()
        for col_name, col in self._columns.items():
            data = list(col[:])
            t.add_column(col_name, data=[data[ix] for ix in sorted_index])
        return t
    else:
        arr = np.zeros(shape=(len(self),), dtype=np.int64)
        shm = shared_memory.SharedMemory(create=True, size=arr.nbytes)  # the co_processors will read this.
        sort_index = np.ndarray(arr.shape, dtype=arr.dtype, buffer=shm.buf)
        sort_index[:] = self.sort_index(sort_mode=sort_mode, **kwargs)

        tasks = []
        columns_refs = {}
        for name in self.columns:
            col = self[name]
            columns_refs[name] = d_key = mem.new_id("/column")
            tasks.append(
                Task(
                    indexing_task,
                    source_key=col.key,
                    destination_key=d_key,
                    shm_name_for_sort_index=shm.name,
                    shape=arr.shape,
                )
            )

        with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
            errs = tm.execute(tasks)
            if any(errs):
                raise Exception("\n".join(filter(lambda x: x is not None, errs)))

        table_key = mem.new_id("/table")
        mem.create_table(key=table_key, columns=columns_refs)

        shm.close()
        shm.unlink()
        t = type(self).load(path=mem.path, key=table_key)
        return t


def is_sorted(self, **kwargs):
    """Performs multi-pass sorting check with precedence given order of column names.
    **kwargs: optional: sort criteria. See Table.sort()
    :return bool
    """
    logging.info("Table.is_sorted running 1 core")  # TODO: This is single core code.
    sorted_index = self.sort_index(**kwargs)
    if any(ix != i for ix, i in enumerate(sorted_index)):
        return False
    return True
