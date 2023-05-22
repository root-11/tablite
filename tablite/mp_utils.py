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
    #return operator.contains(str(a), str(b))  # TODO : check which method is faster


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
