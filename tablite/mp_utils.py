import operator
import numpy as np
from multiprocessing import shared_memory


def not_in(a, b):
    return not operator.contains(str(a), str(b))


def _in(a, b):
    """
    enables filter function 'in'
    """
    return str(a) in str(b)
    return operator.contains(str(a), str(b))  # TODO : check which method is faster


lookup_ops = {
    "in": _in,
    "not in": not_in,
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
    "!=": operator.ne,
    "==": operator.eq,
}


filter_ops = {
    ">": operator.gt,
    ">=": operator.ge,
    "==": operator.eq,
    "<": operator.lt,
    "<=": operator.le,
    "!=": operator.ne,
    "in": _in,
}

filter_ops_from_text = {"gt": ">", "gteq": ">=", "eq": "==", "lt": "<", "lteq": "<=", "neq": "!=", "in": _in}


def filter_evaluation_task(table_key, expression, shm_name, shm_index, shm_shape, slice_):
    """
    multiprocessing tasks for evaluating Table.filter
    """
    assert isinstance(table_key, str)  # 10 --> group = '/table/10'
    assert isinstance(expression, dict)
    assert len(expression) == 3
    assert isinstance(shm_name, str)
    assert isinstance(shm_index, int)
    assert isinstance(shm_shape, tuple)
    assert isinstance(slice_, slice)
    c1 = expression.get("column1", None)
    c2 = expression.get("column2", None)
    c = expression.get("criteria", None)
    assert c in filter_ops
    f = filter_ops.get(c)
    assert callable(f)
    v1 = expression.get("value1", None)
    v2 = expression.get("value2", None)

    columns = mem.mp_get_columns(table_key)
    if c1 is not None:
        column_key = columns[c1]
        dset_A = mem.get_data(f"/column/{column_key}", slice_)
    else:  # v1 is active:
        dset_A = np.array([v1] * (slice_.stop - slice_.start))

    if c2 is not None:
        column_key = columns[c2]
        dset_B = mem.get_data(f"/column/{column_key}", slice_)
    else:  # v2 is active:
        dset_B = np.array([v2] * (slice_.stop - slice_.start))

    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    result_array[shm_index][slice_] = np.array([f(a, b) for a, b in zip(dset_A, dset_B)])  # Evaluate
    existing_shm.close()  # disconnect


def filter_merge_task(table_key, true_key, false_key, shm_name, shm_shape, slice_, filter_type):
    """
    multiprocessing task for merging data after the filter task has been completed.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name)  # connect
    result_array = np.ndarray(shm_shape, dtype=np.bool, buffer=existing_shm.buf)
    mask_source = result_array

    if filter_type == "any":
        true_mask = np.any(mask_source, axis=0)
    else:
        true_mask = np.all(mask_source, axis=0)
    true_mask = true_mask[slice_]
    false_mask = np.invert(true_mask)

    # 2. load source
    columns = mem.mp_get_columns(table_key)

    true_columns, false_columns = {}, {}
    for col_name, column_key in columns.items():
        col = Column(key=column_key)
        slize = col.get_numpy(slice_)
        true_values = slize[true_mask]
        if np.any(true_mask):
            true_columns[col_name] = mem.mp_write_column(true_values)
        false_values = slize[false_mask]
        if np.any(false_mask):
            false_columns[col_name] = mem.mp_write_column(false_values)

    mem.mp_write_table(true_key, true_columns)
    mem.mp_write_table(false_key, false_columns)

    existing_shm.close()  # disconnect


def indexing_task(
    source_key, destination_key, shm_name_for_sort_index, shape, slice_=slice(None), shm_name_for_sort_index_mask=None
):
    """
    performs the creation of a column sorted by sort_index (shared memory object).
    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shm_name_for_sort_index_mask: sort index' shm.name created by main containing None mask.
    shape: shm array shape.

    *used by sort and all join functions.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index)  # connect
    sort_index = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    sort_slice = sort_index[slice_]

    if shm_name_for_sort_index_mask is not None:
        existing_mask_shm = shared_memory.SharedMemory(name=shm_name_for_sort_index_mask)  # connect
        sort_index_mask = np.ndarray(shape, dtype=np.int8, buffer=existing_mask_shm.buf)
    else:
        sort_index_mask = None

    data = mem.get_data(f"/column/{source_key}", slice(None))  # --- READ!

    values = [None] * len(sort_slice)

    start_offset = 0 if slice_.start is None else slice_.start

    for i, (j, ix) in enumerate(enumerate(sort_slice, start_offset)):
        if sort_index_mask is not None and sort_index_mask[j] == 1:
            values[i] = None
        else:
            values[i] = data[ix]

    existing_shm.close()  # disconnect
    if sort_index_mask is not None:
        existing_mask_shm.close()
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


def compress_task(source_key, destination_key, shm_index_name, shape):
    """
    compresses the source using boolean mask from shared memory

    source_key: column to read
    destination_key: column to write
    shm_name_for_sort_index: sort index' shm.name created by main.
    shape: shm array shape.
    """
    existing_shm = shared_memory.SharedMemory(name=shm_index_name)  # connect
    mask = np.ndarray(shape, dtype=np.int64, buffer=existing_shm.buf)

    data = mem.get_data(f"/column/{source_key}", slice(None))  # --- READ!
    values = np.compress(mask, data)

    existing_shm.close()  # disconnect
    mem.mp_write_column(values, column_key=destination_key)  # --- WRITE!


def maskify(arr):
    none_mask = [False] * len(arr)  # Setting the default

    for i in range(len(arr)):
        if arr[i] is None:  # Check if our value is None
            none_mask[i] = True
            arr[i] = 0  # Remove None from the original array

    return none_mask


def share_mem(inp_arr, dtype):
    len_ = len(inp_arr)
    size = np.dtype(dtype).itemsize * len_
    shape = (len_,)

    out_shm = shared_memory.SharedMemory(create=True, size=size)  # the co_processors will read this.
    out_arr_index = np.ndarray(shape, dtype=dtype, buffer=out_shm.buf)
    out_arr_index[:] = inp_arr

    return out_arr_index, out_shm


def map_task(data, index, destination, start, end):
    # connect
    shared_data = shared_memory.SharedMemory(name=data)
    shared_index = shared_memory.SharedMemory(name=index)
    shared_target = shared_memory.SharedMemory(name=destination)
    # work
    shared_target[start:end] = np.take(shared_data[start:end], shared_index[start:end])
    # disconnect
    shared_data.close()
    shared_index.close()
    shared_target.close()
