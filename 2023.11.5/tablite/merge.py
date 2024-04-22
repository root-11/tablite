import numpy as np
from tablite.base import BaseTable
from tablite.config import Config
from tablite.utils import unique_name, type_check


def where(T, criteria, left, right, new):
    """ takes from LEFT where criteria is True else RIGHT 
    and creates a single new column.
    
    :param: T: Table
    :param: criteria: np.array(bool): 
            if True take left column
            else take right column
    :param left: (str) column name
    :param right: (str) column name
    :param new: (str) new name

    :returns: T
    """
    type_check(T, BaseTable)
    if isinstance(criteria, np.ndarray):
        if not criteria.dtype == "bool":
            raise TypeError
    else:
        criteria = np.array(criteria, dtype='bool')
    
    new_uq = unique_name(new, list(T.columns))
    T.add_column(new_uq)
    col = T[new_uq]
    
    for start,end in Config.page_steps(len(criteria)):
        left_values = T[left][start:end]
        right_values = T[right][start:end]
        new_values = np.where(criteria, left_values, right_values)
        col.extend(new_values)

    if new == right:
        T[right] = T[new_uq]  # keep column order
        del T[new_uq]
        del T[left]
    elif new == left:
        T[left] = T[new_uq]  # keep column order
        del T[new_uq]
        del T[right]
    else:
        T[new] = T[new_uq]
        del T[left]
        del T[right]
    return T
