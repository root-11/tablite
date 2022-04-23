import operator
from collections import defaultdict
import pathlib


import numpy as np
import h5py

from tablite2.settings import HDF5_IMPORT_ROOT
from tablite2.column import Column, Page
from tablite2.table import Table
from tablite2.utils import normalize_slice, intercept


def _in(a,b):
    """
    enables filter function 'in'
    """
    return a.decode('utf-8') in b.decode('utf-8')


filter_ops = {
            ">": operator.gt,
            ">=": operator.ge,
            "==": operator.eq,
            "<": operator.lt,
            "<=": operator.le,
            "!=": operator.ne,
            "in": _in
        }

filter_ops_from_text = {
    "gt": ">",
    "gteq": ">=",
    "eq": "==",
    "lt": "<",
    "lteq": "<=",
    "neq": "!=",
    "in": _in
}

def filter(source1, criteria, source2, destination, destination_index, slice_):
    """ PARALLEL TASK FUNCTION
    source1: list of addresses
    criteria: logical operator
    source1: list of addresses
    destination: shm address name.
    destination_index: integer.
    """    
    # 1. access the data sources.
    if isinstance(source1, list):
        A = Column()
        for address in source1:
            datablock = Page.from_address(address)
            A.extend(datablock)
        sliceA = A[slice_]

        A_is_data = True
    else:
        A_is_data = False  # A is value
    
    if isinstance(source2, list):
        B = Column()
        for address in source2:
            datablock = Page.from_address(address)
            B.extend(datablock)
        sliceB = B[slice_]

        B_is_data = True
    else:
        B_is_data = False  # B is a value.

    assert isinstance(destination, SharedMemoryAddress)
    handle, data = destination.to_shm()  # the handle is required to sit idle as gc otherwise deletes it.
    assert destination_index < len(data),  "len of data is the number of evaluations, so the destination index must be within this range."
    
    # ir = range(*normalize_slice(length, slice_))
    # di = destination_index
    # if length_A is None:
    #     if length_B is None:
    #         result = criteria(source1,source2)
    #         result = np.ndarray([result for _ in ir], dtype='bool')
    #     else:  # length_B is not None
    #         sliceA = np.array([source1] * length_B)
    # else:
    #     if length_B is None:
    #         B = np.array([source2] * length_A)
    #     else:  # A & B is not None
    #         pass
    
    if A_is_data and B_is_data:
        result = eval(f"sliceA {criteria} sliceB")
    if A_is_data or B_is_data:
        if A_is_data:
            sliceB = np.array([source2] * len(sliceA))
        else:
            sliceA = np.array([source1] * len(sliceB))
    else:
        v = criteria(source1,source2)
        length = slice_.stop - slice_.start 
        ir = range(*normalize_slice(length, slice_))
        result = np.ndarray([v for _ in ir], dtype='bool')

    if criteria == "in":
        result = np.ndarray([criteria(a,b) for a, b in zip(sliceA, sliceB)], dtype='bool')
    else:
        result = eval(f"sliceA {criteria} sliceB")  # eval is evil .. blah blah blah... Eval delegates to optimized numpy functions.        

    data[destination_index][slice_] = result


def merge(source, mask, filter_type, slice_):
    """ PARALLEL TASK FUNCTION
    creates new tables from combining source and mask.
    """
    if not isinstance(source, dict):
        raise TypeError
    for L in source.values():
        if not isinstance(L, list):
            raise TypeError
        if not all(isinstance(sma, SharedMemoryAddress) for sma in L):
            raise TypeError

    if not isinstance(mask, SharedMemoryAddress):
        raise TypeError
    if not isinstance(filter_type, str) and filter_type in {'any', 'all'}:
        raise TypeError
    if not isinstance(slice_, slice):
        raise TypeError
    
    # 1. determine length of Falses and Trues
    f = any if filter_type == 'any' else all
    handle, mask = mask.to_shm() 
    if len(mask) == 1:
        true_mask = mask[0][slice_]
    else:
        true_mask = [f(c[i] for c in mask) for i in range(slice_.start, slice_.stop)]
    false_mask = np.invert(true_mask)

    t1 = Table.from_address(source)  # 2. load Table.from_shm(source)
    # 3. populate the tables
    
    true, false = Table(), Table()
    for name, mc in t1.columns.items():
        mc_unfiltered = np.array(mc[slice_])
        if any(true_mask):
            data = mc_unfiltered[true_mask]
            true.add_column(name, data)  # data = mc_unfiltered[new_mask]
        if any(false_mask):
            data = mc_unfiltered[false_mask]
            false.add_column(name, data)

    # 4. return table.to_shm()
    return true.key, false.key   


