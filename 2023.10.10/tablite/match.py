import numpy as np
from tablite.base import BaseTable
from tablite.utils import unique_name, type_check, name_check


def match(T, other, *criteria, keep_left=None, keep_right=None):  # lookup and filter combined - drops unmatched rows.
    """
    performs inner join where `T` matches `other` and removes rows that do not match.

    :param: T: Table
    :param: other: Table
    :param: criteria: Each criteria must be a tuple with value comparisons in the form:
        
        (LEFT, OPERATOR, RIGHT), where operator must be "=="

        Example:
            ('column A', "==", 'column B')

        This syntax follows the lookup syntax. See Lookup for details.

    :param: keep_left: list of columns to keep.
    :param: keep_right: list of right columns to keep.
    """
    assert isinstance(T, BaseTable)
    assert isinstance(other, BaseTable)
    if keep_left is None:
        keep_left = [n for n in T.columns]
    else:
        type_check(keep_left, list)
        name_check(T.columns, *keep_left)

    if keep_right is None:
        keep_right = [n for n in other.columns]
    else:
        type_check(keep_right, list)
        name_check(other.columns, *keep_right)

    indices = np.full(shape=(len(T),), fill_value=-1, dtype=np.int64)
    for arg in criteria:
        b,_,a = arg
        if _ != "==":
            raise ValueError("match requires A == B. For other logic visit `lookup`")
        if b not in T.columns:
            raise ValueError(f"Column {b} not found in T for criteria: {arg}")
        if a not in other.columns:
            raise ValueError(f"Column {a} not found in T for criteria: {arg}")
        
        index_update = find_indices(other[a][:], T[b][:], fill_value=-1)
        indices = merge_indices(indices, index_update)

    cls = type(T)
    new = cls()
    for name in T.columns:
        if name in keep_left:
            new[name] = np.compress(indices != -1, T[name][:])
    
    for name in other.columns:
        if name in keep_right:
            new_name = unique_name(name, new.columns)
            primary = np.compress(indices != -1, indices)
            new[new_name] = np.take(other[name][:], primary)
        
    return new


def find_indices(x,y, fill_value=-1):  # fast.
    """
    finds index of y in x
    """
    # disassembly of numpy:
    # import numpy as np
    # x = np.array([3, 5, 7,  1,   9, 8, 6, 6])
    # y = np.array([2, 1, 5, 10, 100, 6])
    index = np.argsort(x)  # array([3, 0, 1, 6, 7, 2, 5, 4])
    sorted_x = x[index]  # array([1, 3, 5, 6, 6, 7, 8, 9])
    sorted_index = np.searchsorted(sorted_x, y)  # array([1, 0, 2, 8, 8, 3])
    yindex = np.take(index, sorted_index, mode="clip")  # array([0, 3, 1, 4, 4, 6])
    mask = x[yindex] != y  # array([ True, False, False,  True,  True, False])
    indices = np.ma.array(yindex, mask=mask, fill_value=fill_value)  
    # masked_array(data=[--, 3, 1, --, --, 6], mask=[ True, False, False,  True,  True, False], fill_value=999999)
    # --: y[0] not in x
    # 3 : y[1] == x[3]
    # 1 : y[2] == x[1]
    # --: y[3] not in x
    # --: y[4] not in x
    # --: y[5] == x[6]
    result = np.where(~indices.mask, indices.data, -1)  
    return result  # array([-1,  3,  1, -1, -1,  6])


def merge_indices(x1, *args, fill_value=-1):
    """
    merges x1 and x2 where 
    """
    # dis:
    # >>> AA = array([-1,  3, -1, 5])
    # >>> BB = array([-1, -1,  4, 5])
    new = x1[:]  # = AA
    for arg in args:
        mask = (new == fill_value)  # array([True, False, True, False])
        new = np.where(mask, arg, new)  # array([-1, 3, 4, 5])
    return new   # array([-1, 3, 4, 5])

