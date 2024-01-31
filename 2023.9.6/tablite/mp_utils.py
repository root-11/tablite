import operator
import numpy as np
from tablite.utils import load_numpy
from multiprocessing import shared_memory
import psutil
from tablite.config import Config


def not_in(a, b):
    return not operator.contains(str(a), str(b))


def _in(a, b):
    """
    enables filter function 'in'
    """
    return str(a) in str(b)
    # return operator.contains(str(a), str(b))  # TODO : check which method is faster


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


def select_processing_method(fields, sp, mp):
    """

    Args:
        fields (int): number of fields
        sp (callable): method for single processing
        mp (callable): method for multiprocessing

    Returns:
        _type_: _description_
    """
    if Config.MULTIPROCESSING_MODE == Config.FORCE:
        m = mp
    elif Config.MULTIPROCESSING_MODE == Config.FALSE:
        m = sp
    elif fields < Config.SINGLE_PROCESSING_LIMIT:
        m = sp
    elif max(psutil.cpu_count(logical=False), 1) < 2:
        m = sp
    else:
        m = mp
    return m


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


def map_task(data_shm_name, index_shm_name, destination_shm_name, shape, dtype, start, end):
    # connect
    shared_data = shared_memory.SharedMemory(name=data_shm_name)
    data = np.ndarray(shape, dtype=dtype, buffer=shared_data.buf)

    shared_index = shared_memory.SharedMemory(name=index_shm_name)
    index = np.ndarray(shape, dtype=np.int64, buffer=shared_index.buf)

    shared_target = shared_memory.SharedMemory(name=destination_shm_name)
    target = np.ndarray(shape, dtype=dtype, buffer=shared_target.buf)
    # work
    target[start:end] = np.take(data[start:end], index[start:end])
    # disconnect
    shared_data.close()
    shared_index.close()
    shared_target.close()


def reindex_task(src, dst, index_shm, shm_shape, start, end):
    # connect
    existing_shm = shared_memory.SharedMemory(name=index_shm)
    shared_index = np.ndarray(shm_shape, dtype=np.int64, buffer=existing_shm.buf)
    # work
    array = load_numpy(src)
    new = np.take(array, shared_index[start:end])
    np.save(dst, new, allow_pickle=True, fix_imports=False)
    # disconnect
    existing_shm.close()
