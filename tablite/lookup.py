


def lookup(self, other, *criteria, all=True, tqdm=_tqdm):  # TODO: This is single core code.
    """function for looking up values in `other` according to criteria in ascending order.
    :param: other: Table sorted in ascending search order.
    :param: criteria: Each criteria must be a tuple with value comparisons in the form:
        (LEFT, OPERATOR, RIGHT)
    :param: all: boolean: True=ALL, False=Any

    OPERATOR must be a callable that returns a boolean
    LEFT must be a value that the OPERATOR can compare.
    RIGHT must be a value that the OPERATOR can compare.

    Examples:
            ('column A', "==", 'column B')  # comparison of two columns
            ('Date', "<", DataTypes.date(24,12) )  # value from column 'Date' is before 24/12.
            f = lambda L,R: all( ord(L) < ord(R) )  # uses custom function.
            ('text 1', f, 'text 2')
            value from column 'text 1' is compared with value from column 'text 2'
    """
    assert isinstance(self, Table)
    assert isinstance(other, Table)

    all = all
    any = not all

    ops = lookup_ops

    functions, left_criteria, right_criteria = [], set(), set()

    for left, op, right in criteria:
        left_criteria.add(left)
        right_criteria.add(right)
        if callable(op):
            pass  # it's a custom function.
        else:
            op = ops.get(op, None)
            if not callable(op):
                raise ValueError(f"{op} not a recognised operator for comparison.")

        functions.append((op, left, right))
    left_columns = [n for n in left_criteria if n in self.columns]
    right_columns = [n for n in right_criteria if n in other.columns]

    results = []
    lru_cache = {}
    left = self.__getitem__(*left_columns)
    if isinstance(left, Column):
        tmp, left = left, Table()
        left[left_columns[0]] = tmp
    right = other.__getitem__(*right_columns)
    if isinstance(right, Column):
        tmp, right = right, Table()
        right[right_columns[0]] = tmp
    assert isinstance(left, Table)
    assert isinstance(right, Table)

    for row1 in tqdm(left.rows, total=self.__len__()):
        row1_tup = tuple(row1)
        row1d = {name: value for name, value in zip(left_columns, row1)}
        row1_hash = hash(row1_tup)

        match_found = True if row1_hash in lru_cache else False

        if not match_found:  # search.
            for row2ix, row2 in enumerate(right.rows):
                row2d = {name: value for name, value in zip(right_columns, row2)}

                evaluations = {op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions}
                # The evaluations above does a neat trick:
                # as L is a dict, L.get(left, L) will return a value
                # from the columns IF left is a column name. If it isn't
                # the function will treat left as a value.
                # The same applies to right.
                A = all and (False not in evaluations)
                B = any and True in evaluations
                if A or B:
                    match_found = True
                    lru_cache[row1_hash] = row2ix
                    break

        if not match_found:  # no match found.
            lru_cache[row1_hash] = None

        results.append(lru_cache[row1_hash])

    result = self.copy()

    if tcfg.PROCESSING_PRIORITY == "sp":
        return self._sp_lookup(other, result, results)
    elif tcfg.PROCESSING_PRIORITY == "mp":
        return self._mp_lookup(other, result, results)
    else:
        if len(self) * len(other.columns) < Config.SINGLE_PROCESSING_LIMIT:
            return self._sp_lookup(other, result, results)
        else:
            return self._mp_lookup(other, result, results)

def _sp_lookup(self, other, result, results):
    for col_name in other.columns:
        col_data = other[col_name][:]
        revised_name = unique_name(col_name, result.columns)
        result[revised_name] = [col_data[k] if k is not None else None for k in results]
    return result

def _mp_lookup(self, other, result, results):
    # 1. create shared memory array.
    RIGHT_NONE_MASK = _maskify(results)
    right_arr, right_shm = _share_mem(results, np.int64)
    _, right_msk_shm = _share_mem(RIGHT_NONE_MASK, np.bool8)

    # 2. create tasks
    tasks = []
    columns_refs = {}

    for name in other.columns:
        revised_name = unique_name(name, result.columns + list(columns_refs.keys()))
        col = other[name]
        columns_refs[revised_name] = d_key = mem.new_id("/column")
        tasks.append(
            Task(
                indexing_task,
                source_key=col.key,
                destination_key=d_key,
                shm_name_for_sort_index=right_shm.name,
                shm_name_for_sort_index_mask=right_msk_shm.name,
                shape=right_arr.shape,
            )
        )

    # 3. let task manager handle the tasks
    with TaskManager(cpu_count=min(psutil.cpu_count(), len(tasks))) as tm:
        errs = tm.execute(tasks)
        if any(errs):
            raise Exception("\n".join(filter(lambda x: x is not None, errs)))

    # 4. update the result table.
    with h5py.File(mem.path, "r+") as h5:
        dset = h5[f"/table/{result.key}"]
        columns = json.loads(dset.attrs["columns"])
        columns.update(columns_refs)
        dset.attrs["columns"] = json.dumps(columns)
        dset.attrs["saved"] = False

    # 5. close the share memory and deallocate
    right_shm.close()
    right_shm.unlink()

    right_msk_shm.close()
    right_msk_shm.unlink()

    # 6. reload the result table
    t = Table.load(path=mem.path, key=result.key)

    return t

