import psutil
import math
import numpy as np
from tablite.base import Table, Column
from tablite.utils import sub_cls_check, unique_name
from tablite.mp_utils import lookup_ops, share_mem, map_task, select_processing_method
from mplite import Task, TaskManager


from tqdm import tqdm as _tqdm


def lookup(T, other, *criteria, all=True, tqdm=_tqdm):
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
    sub_cls_check(T, Table)
    sub_cls_check(other, Table)

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
    left_columns = [n for n in left_criteria if n in T.columns]
    right_columns = [n for n in right_criteria if n in other.columns]

    result_index = np.empty(shape=(len(T)), dtype=np.int64)
    cache = {}
    left = T.__getitem__(*left_columns)
    if isinstance(left, Column):
        tmp, left = left, Table()
        left[left_columns[0]] = tmp
    right = other.__getitem__(*right_columns)
    if isinstance(right, Column):
        tmp, right = right, Table()
        right[right_columns[0]] = tmp
    assert isinstance(left, Table)
    assert isinstance(right, Table)

    for ix, row1 in tqdm(enumerate(left.rows), total=len(T)):
        row1_tup = tuple(row1)
        row1d = {name: value for name, value in zip(left_columns, row1)}
        row1_hash = hash(row1_tup)

        match_found = True if row1_hash in cache else False

        if not match_found:  # search.
            for row2ix, row2 in enumerate(right.rows):
                row2d = {name: value for name, value in zip(right_columns, row2)}

                evaluations = {op(row1d.get(left, left), row2d.get(right, right)) for op, left, right in functions}
                # The evaluations above does a neat trick:
                # as L is a dict, L.get(left, L) will return a value
                # from the columns IF left is a column name. If it isn't
                # the function will treat left as a value.
                # The same applies to right.
                all_ = all and (False not in evaluations)
                any_ = any and True in evaluations
                if all_ or any_:
                    match_found = True
                    cache[row1_hash] = row2ix
                    break

        if not match_found:  # no match found.
            cache[row1_hash] = -1  # -1 is replacement for None in the index as numpy can't handle Nones.

        result_index[ix] = cache[row1_hash]

    f = select_processing_method(2 * max(len(T), len(other)), _sp_lookup, _mp_lookup)
    return f(T, other, result_index)


def _sp_lookup(T, other, index):
    result = T.copy()
    for col_name in other.columns:
        col_data = other[col_name][:]
        revised_name = unique_name(col_name, result.columns)
        # 1/3 reindex but well knowing that -1 in the index will be wrong.
        reindexed = np.take(col_data, index)
        # 2/3 prepare an array of nones
        nones = np.empty(shape=index.shape, dtype=object)
        # 3/3 merge reindexed array with None, whenever the original index is -1.
        result[revised_name] = np.where(index == -1, nones, reindexed)
        # the result of the three steps above are the same as in python below:
        # result[revised_name] = [col_data[k] if k != -1 else None for k in index]
    return result


def _mp_lookup(T, other, index):
    result = T.copy()
    cpus = max(psutil.cpu_count(logical=False), 1)
    step_size = math.ceil(len(T) / cpus)

    with TaskManager(cpu_count=cpus) as tm:  # keeps the CPU pool alive during the whole join.
        # for table, columns, side in ([T, left_columns, LEFT], [other, right_columns, RIGHT]):

        index, index_shm = share_mem(index, np.int64)  # <-- this is index
        # As all indices in `index` are positive, -1 is used as replacement for None.

        for name in other.columns:
            data = other[name][:]
            # TODO         ^---- determine how much memory is free and then decide
            # either to mmap the source or keep it in RAM.

            data, data_shm = share_mem(data, data.dtype)  # <-- this is source
            destination, dest_shm = share_mem(np.ndarray(shape=data.shape), data.dtype)  # <--this is destination.

            tasks = []
            start, end = 0, step_size
            for _ in range(cpus):
                tasks.append(
                    Task(map_task, data_shm.name, index_shm.name, dest_shm.name, data.shape, data.dtype, start, end)
                )
                start, end = end, end + step_size
            # All CPUS now work on the same column and memory footprint is predetermined.
            results = tm.execute(tasks)
            if any(i is not None for i in results):
                raise Exception("\n".join(filter(lambda x: x is not None, results)))

            # As the data and index no longer is needed, then can be closed.
            data_shm.close()
            data_shm.unlink()

            # As all the tasks have been completed, the Column can handle the pagination at once.
            name = unique_name(name, set(result.columns))

            # deal with Nones, before storing.
            nones = np.empty(shape=destination.shape, dtype=object)
            result[name] = np.where(index == -1, nones, destination)

            dest_shm.close()  # finally close dest.
            dest_shm.unlink()
        index_shm.close()
        index_shm.unlink()

    return result
